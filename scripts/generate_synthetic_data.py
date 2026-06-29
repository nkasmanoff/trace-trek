#!/usr/bin/env python
"""
Synthetic trajectory generator for opencode-sft dataset augmentation (v2).

Design goals (vs. the original one-shot generator):

1. Turn-by-turn agent loop instead of "emit the whole conversation in one JSON
   blob". The *teacher policy* only ever produces ONE assistant turn at a time;
   this code assembles the canonical wire schema (ids, tool_call_id pairing,
   parallel calls). That makes malformed final schema essentially impossible
   because the model never writes the envelope.

2. Decoupled environment simulator. Tool OUTPUTS come from a separate LLM call
   that does not see the desired narrative conclusion, only the transcript so
   far + the specific tool call. With probability `p_deadend` it returns a
   realistic failure (missing file, empty grep, command error, red herring, ...)
   that the agent then has to recover from. The simulator is fed the full
   running transcript so its fabricated outputs stay self-consistent.

   NOTE: outputs are still fabricated, not read from a real filesystem. This is
   a mitigation, not grounding. The judge stage (below) is therefore load-bearing.

3. Exact schema match to the canonical replay trace:
     assistant tool call : {"id","type":"function","function":{"name","arguments"}}
                           arguments is a JSON *string*. Parallel calls = multiple
                           entries in ONE assistant message.
     tool result         : {"role":"tool","content","tool_call_id"}  (keyed by id,
                           no "tool" name field)

4. Strict validation: jsonschema per message + referential integrity
   (every tool result answers a real prior call exactly once, correct grouping).

5. LLM judge gate: coherence / tool-use / faithfulness / recovery. Bottom slice
   discarded.

6. Diversity seeding: structured scenario sampler (repo archetype x task type x
   env x dead-end profile) + lexical near-duplicate filtering at the end.
"""

import argparse
import functools
import json
import os
import random
import string
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from urllib.parse import quote

try:
    import jsonschema
except ImportError:  # pragma: no cover
    print(
        "Missing dependency: pip install jsonschema --break-system-packages",
        file=sys.stderr,
    )
    raise

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # allows importing this module for tests without the SDK


# =============================================================================
# Canonical system prompt for the *trajectory* (the data we emit).
#
# We do NOT hand-maintain a condensed approximation. Instead we load opencode's
# REAL system prompt, harvested verbatim from proxy logs of live opencode
# sessions into viewer/server/fixtures/opencode-harness.json (see
# scripts/harvest-harness.js). The harvested template has three per-session
# holes that we refill exactly the way opencode does at runtime (this mirrors
# viewer/server/harness.js):
#   {{POWERED_BY}} -> the model-identity line
#   {{ENV}}        -> the <env> working-directory / platform / date block
#   {{CWD_URI}}    -> the built-in skill's file:// location
#
# This guarantees the SFT system messages match deployment byte-for-byte, except
# for the per-scenario env values which we vary deliberately for diversity.
# =============================================================================
HARNESS_FIXTURE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "viewer",
    "server",
    "fixtures",
    "opencode-harness.json",
)


@functools.lru_cache(maxsize=1)
def _load_opencode_prompts() -> dict:
    with open(HARNESS_FIXTURE) as f:
        return json.load(f)["prompts"]


def select_prompt_role(model_id: str) -> str:
    """Pick the opencode prompt role for a model id (mirrors harness.js).

    Claude/Anthropic teachers run opencode's frontier "beast" prompt; every
    other model (including GLM) uses the standard "build" prompt.
    """
    m = (model_id or "").lower()
    if "claude" in m or "anthropic" in m:
        return "beast"
    return "build"


def _powered_by_line(model_id: str) -> str:
    mid = model_id or "unknown"
    return f"You are powered by the model named {mid}. The exact model ID is {mid}"


def _env_block(scenario: "Scenario") -> str:
    return "\n".join(
        [
            "Here is some useful information about the environment you are running in:",
            "<env>",
            f"  Working directory: {scenario.cwd}",
            f"  Workspace root folder: {scenario.cwd}",
            f"  Is directory a git repo: {'yes' if scenario.is_git else 'no'}",
            f"  Platform: {scenario.platform}",
            f"  Today's date: {scenario.date}",
            "</env>",
        ]
    )


def build_system_prompt(
    scenario: "Scenario", model_id: str, role_override: Optional[str] = None
) -> str:
    """Refill opencode's harvested system prompt for this scenario + model.

    `role_override` forces a specific harvested prompt role (e.g. "beast",
    "build", "explore") instead of auto-selecting from the model id.
    """
    prompts = _load_opencode_prompts()
    role = role_override or select_prompt_role(model_id)
    template = prompts.get(role)
    if not template:
        # Fall back to "build" (the standard agent prompt) but be loud about it,
        # since a missing role usually means stale/missing harvested fixtures.
        available = ", ".join(sorted(prompts)) or "<none>"
        if role != "build" and prompts.get("build"):
            print(
                f"  ! system-prompt role '{role}' not in opencode-harness.json "
                f"(have: {available}); falling back to 'build'.",
                file=sys.stderr,
            )
            role = "build"
            template = prompts.get("build")
    if not template:
        raise RuntimeError(
            f"opencode-harness.json has no usable prompt for role '{role}' "
            f"(available: {', '.join(sorted(prompts)) or '<none>'}). "
            "Re-run scripts/harvest-harness.js against fresh proxy logs."
        )
    content = (
        template.replace("{{POWERED_BY}}", _powered_by_line(model_id))
        .replace("{{ENV}}", _env_block(scenario))
        .replace("{{CWD_URI}}", "file://" + quote(scenario.cwd))
    )
    # Guard against a doubled scheme if the template kept its own file://.
    return content.replace("file://file://", "file://")


# =============================================================================
# Tool catalog the agent may use, plus output-format guidance for the simulator.
# Format templates are drawn from the canonical replay trace so simulated
# outputs are shaped like real tool results.
# =============================================================================
TOOLS = ["bash", "glob", "grep", "read", "write", "edit", "task", "webfetch"]

TOOL_OUTPUT_FORMAT_GUIDE = """Output format conventions (match these exactly):

- read on a FILE:
<path>{abs_path}</path>
<type>file</type>
<content>
1: first line of file
2: second line
...
(End of file - total N lines)
</content>

- read on a DIRECTORY:
<path>{abs_path}</path>
<type>directory</type>
<entries>
name1
subdir/
name2

(N entries)
</entries>

- grep / bash grep: raw stdout, one match per line, usually `line_number:matched text`.
- bash (other): raw stdout/stderr exactly as a shell would print it.
- glob: newline-separated absolute paths.

All file paths must be ABSOLUTE and consistent with the working directory and with paths already established earlier in the transcript."""


# =============================================================================
# Diversity seeding: structured scenario sampler.
# =============================================================================
REPO_ARCHETYPES = [
    "a meta-codegen project that reads its own source and rewrites modules (Python, AST parsing, Jinja2 templates)",
    "an AI agent loop that reflects on past trajectories and proposes prompt patches (Python, LLM API, JSON logging)",
    "a small knowledge-graph wiki for personal research notes (Python, SQLite, full-text search, markdown)",
    "a self-hosted astronomy observation planner (Python, astropy, ephem, matplotlib, Jupyter)",
    "a toolkit that scrapes NASA/ESA open data catalogs and builds light-curve plots (Python, pandas, FITS, HTTPX)",
    "an exoplanet transit detector that fits light curves from TESS public data (Python, NumPy, SciPy, batman)",
    "a personal CLI dashboard for weather, calendar, and habit tracking (Go, bubbletea, SQLite)",
    "a side-project game boy emulator in Rust (no_std, SDL2 bindings, CPU debugger)",
    "a Discord bot for a programming-language study group (Python, discord.py, aiosqlite, trivia)",
    "a static-site generator for a personal blog with Wiki-style backlinks (Python, markdown-it, Jinja2, CSS)",
    "a hobby compiler for a tiny lisp-like language (Rust, nom parser, LLVM-IR emit)",
    "a local-first note-taking app with full-text search and graph viz (TypeScript, React, SQLite WASM, D3)",
    "a recursive self-improvement loop: an agent that rewrites its own prompts based on eval results (Python, LLM API, git)",
    "a synthetic-data pipeline that bootstraps a fine-tuned model from a larger teacher (Python, HF datasets, LoRA)",
    "a personal finance tracker with ledger-style double-entry (Python, click, SQLite, plotly)",
    "a multiplayer roguelike written in Python (asyncio, tcod, websockets, pytest)",
]

TASK_TYPES = [
    "figure out why the self-improvement loop's eval score dropped after the latest prompt patch",
    "trace how the agent's reflection module feeds its output back into the system prompt template",
    "find where a knowledge-graph query is leaking memory on large wikis",
    "port the CLI observation planner from astropy to jplephem for better performance",
    "debug why the TESS light-curve downloader skips a third of the target IDs",
    "add fuzzy matching to the personal wiki search so typos still find pages",
    "understand how the emulator's CPU dispatch table is generated and extend it for one new opcode",
    "find the bottleneck in the Discord bot's message handler and suggest a fix",
    "port the static-site generator's backlink logic from Python to a build-time plugin",
    "add tail-call optimization to the lisp compiler's codegen pass",
    "investigate why the note-taking app's full-text search returns stale results after a sync",
    "trace where the synthetic-data pipeline drops examples that the judge didn't reject",
    "audit how the prompt-rewriting agent validates its own changes before committing them",
    "add a new celestial body type to the observation planner without breaking existing queries",
    "find the off-by-one in the roguelike's FOV raycasting code that causes flickering walls",
    "understand how the self-improvement loop selects which past trajectory to learn from",
    "track down why the exoplanet transit fitter converges slowly on grazing transits",
    "add persistent storage to the habit tracker so history survives a restart",
    "find and deduplicate the config-loading logic spread across three modules in the blog generator",
    "figure out why the weather CLI shows yesterday's forecast and where the cache invalidation breaks",
]

DEADEND_TYPES = [
    "file_not_found",  # agent reads a path that does not exist
    "empty_grep",  # search returns no matches
    "is_a_directory",  # agent reads something that turns out to be a dir
    "command_error",  # bash command fails with a nonzero exit / stderr
    "red_herring",  # output is real but points the agent the wrong way
    "truncated_output",  # output is cut off / too large, must be narrowed
    "permission_denied",  # cannot read the file
]

PLATFORMS = ["darwin", "linux"]


@dataclass
class Scenario:
    repo: str
    task_type: str
    cwd: str
    platform: str
    is_git: bool
    date: str
    deadend_profile: list[str]  # which dead-end types are eligible this run

    def describe(self) -> str:
        return (
            f"Repository: {self.repo}\n"
            f"Working directory: {self.cwd}\n"
            f"Platform: {self.platform}; git repo: {self.is_git}\n"
            f"User's underlying goal: {self.task_type}."
        )


def sample_scenario(rng: random.Random) -> Scenario:
    repo = rng.choice(REPO_ARCHETYPES)
    platform = rng.choice(PLATFORMS)
    proj = "".join(rng.choices(string.ascii_lowercase, k=6))
    if platform == "darwin":
        cwd = f"/Users/dev/projects/{proj}"
    else:
        cwd = f"/home/dev/projects/{proj}"
    # Each scenario gets a random subset of eligible dead-end types (or none).
    k = rng.choice([0, 1, 1, 2])  # weight toward 1 dead-end
    deadends = rng.sample(DEADEND_TYPES, k) if k else []
    return Scenario(
        repo=repo,
        task_type=rng.choice(TASK_TYPES),
        cwd=cwd,
        platform=platform,
        is_git=rng.random() < 0.85,
        date="Tue Jun 09 2026",
        deadend_profile=deadends,
    )


# =============================================================================
# LLM client wrapper. Provider-agnostic OpenAI-compatible client; works against
# the Modal-hosted GLM endpoint (Modal-Key/Secret headers) or OpenRouter
# (Bearer auth), both emitting json_object output.
# =============================================================================
MODAL_BASE_URL = "https://nkasmanoff--ep-glm-5-2-fp8-server.us-west.modal.direct/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_headers: Optional[dict] = None,
        reasoning: bool = False,
    ):
        if OpenAI is None:
            raise RuntimeError("openai SDK not installed")
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers=default_headers or None,
        )
        self._reasoning = reasoning

    @classmethod
    def modal(
        cls,
        key_id: str,
        key_secret: str,
        base_url: str = MODAL_BASE_URL,
        reasoning: bool = False,
    ) -> "LLMClient":
        return cls(
            base_url=base_url,
            api_key="unused",
            default_headers={"Modal-Key": key_id, "Modal-Secret": key_secret},
            reasoning=reasoning,
        )

    @classmethod
    def openrouter(
        cls,
        api_key: str,
        base_url: str = OPENROUTER_BASE_URL,
        reasoning: bool = False,
    ) -> "LLMClient":
        return cls(base_url=base_url, api_key=api_key, reasoning=reasoning)

    def complete(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        if self._reasoning:
            kwargs["extra_body"] = {"reasoning": {"enabled": True}}
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


def parse_json_lenient(raw: str) -> Any:
    """Parse JSON, tolerating accidental code fences."""
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        if "```json" in raw:
            raw = raw.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in raw:
            raw = raw.split("```", 1)[1].split("```", 1)[0]
        return json.loads(raw.strip())


def call_json(
    client: LLMClient,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    attempts: int = 3,
) -> Optional[dict]:
    """Call the model and parse a JSON object, retrying on transport/parse errors."""
    for attempt in range(attempts):
        try:
            raw = client.complete(model, system, user, temperature, max_tokens)
            obj = parse_json_lenient(raw)
            if isinstance(obj, dict):
                return obj
        except (
            Exception
        ) as e:  # noqa: BLE001 - we genuinely want to retry anything transient
            if attempt == attempts - 1:
                print(f"    call_json failed after {attempts} attempts: {e}")
            else:
                time.sleep(1.5 * (attempt + 1))
    return None


# =============================================================================
# ID generation (matches the trace style: toolu_ + base62).
# =============================================================================
_ID_ALPHABET = string.ascii_letters + string.digits


def gen_tool_id(rng: random.Random) -> str:
    return "toolu_" + "".join(rng.choices(_ID_ALPHABET, k=24))


# =============================================================================
# Agent step: ONE assistant turn from the teacher policy.
# =============================================================================
AGENT_SYSTEM = """You are role-playing as OpenCode, an expert coding agent, to generate ONE assistant turn of a synthetic tool-use trajectory for training data.

You are given: the trajectory's own system prompt, the scenario, and the conversation so far (user message + any previous assistant turns and tool results). Produce the NEXT assistant turn only.

Behave like a careful real agent:
- Investigate before concluding. Use tools to gather evidence.
- You MAY issue several tool calls at once when they are independent (parallel). Issue them sequentially when one depends on another's result.
- If a previous tool result was an error, empty, or misleading, DO NOT pretend otherwise. Adapt: different path, broader search, list the directory, read a related file. Recovery is expected and desirable.
- When (and only when) you have enough evidence, give a final answer with NO further tool calls. Reference code as file_path:line_number where relevant. Keep it concise and CLI-appropriate.

Available tools: bash, glob, grep, read, write, edit, task, webfetch.

Return ONLY a JSON object with this exact shape:
{
  "thought": "brief private reasoning for this turn (1-3 sentences)",
  "message": "assistant text for this turn; may be empty string when you are only making tool calls",
  "actions": [
    {"tool": "read", "args": {"filePath": "/abs/path"}},
    {"tool": "bash", "args": {"command": "grep -n foo src/"}}
  ],
  "final": false
}

Rules:
- "actions" is a list of tool calls to make THIS turn. Put independent calls together to represent parallel tool use.
- Set "final": true and "actions": [] when you are delivering the final answer (put it in "message").
- Tool args must be realistic for the tool (read -> {"filePath": ...}; bash -> {"command": ...}; grep -> {"pattern":..., "path":...}; glob -> {"pattern":...}).
- Use absolute paths under the working directory shown in the system prompt.
- Do not invent tool RESULTS; you only choose actions. Results are provided back to you on the next turn."""

# One real-schema few-shot exemplar of an assistant tool-call turn, shown as
# actual JSON so the model sees the target shape rather than a prose sketch.
AGENT_FEWSHOT = json.dumps(
    {
        "thought": "I should look at the directory first, then read the most relevant files in parallel.",
        "message": "",
        "actions": [
            {
                "tool": "read",
                "args": {"filePath": "/home/dev/projects/example/README.md"},
            },
            {
                "tool": "bash",
                "args": {
                    "command": "grep -rn 'def handle_retry' /home/dev/projects/example/src"
                },
            },
        ],
        "final": False,
    },
    indent=2,
)


def render_transcript_for_model(messages: list[dict]) -> str:
    """Render the wire-format transcript into a readable form for the teacher/simulator."""
    lines: list[str] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            lines.append("=== SYSTEM (trajectory) ===\n" + m["content"])
        elif role == "user":
            lines.append("=== USER ===\n" + m["content"])
        elif role == "assistant":
            if m.get("tool_calls"):
                calls = []
                for tc in m["tool_calls"]:
                    calls.append(
                        f"  - {tc['function']['name']}({tc['function']['arguments']})  [id={tc['id']}]"
                    )
                body = m.get("content") or "(no text; tool calls only)"
                lines.append(
                    "=== ASSISTANT ===\n" + body + "\nTOOL CALLS:\n" + "\n".join(calls)
                )
            else:
                lines.append(
                    "=== ASSISTANT (final/text) ===\n" + (m.get("content") or "")
                )
        elif role == "tool":
            lines.append(
                f"=== TOOL RESULT [id={m['tool_call_id']}] ===\n" + m["content"]
            )
    return "\n\n".join(lines)


def agent_step(
    client: LLMClient,
    model: str,
    scenario: Scenario,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    force_final: bool = False,
) -> Optional[dict]:
    transcript = render_transcript_for_model(messages)
    instruction = ""
    if force_final:
        instruction = (
            "\n\nIMPORTANT: You have investigated enough. Produce your FINAL answer now: "
            'set "final": true, "actions": [], and put the complete answer in "message".'
        )
    user = (
        f"SCENARIO:\n{scenario.describe()}\n\n"
        f"FORMAT EXEMPLAR (an assistant tool-call turn, for shape only):\n{AGENT_FEWSHOT}\n\n"
        f"CONVERSATION SO FAR:\n{transcript}\n\n"
        f"Produce the next assistant turn as a single JSON object.{instruction}"
    )
    obj = call_json(client, model, AGENT_SYSTEM, user, temperature, max_tokens)
    if obj is None:
        return None
    # Normalize.
    obj.setdefault("thought", "")
    obj.setdefault("message", "")
    obj.setdefault("actions", [])
    if not isinstance(obj["actions"], list):
        obj["actions"] = []
    obj["final"] = bool(obj.get("final")) or len(obj["actions"]) == 0
    if force_final:
        obj["final"] = True
        obj["actions"] = []
    return obj


# =============================================================================
# Environment simulator: a SEPARATE call that fabricates one tool result.
# Does not receive the desired conclusion. Injects dead-ends per profile.
# =============================================================================
SIM_SYSTEM = (
    """You simulate the OUTPUT of a developer tool (bash/grep/read/glob/etc.) inside a synthetic coding-agent trajectory. You are NOT the agent and you do not know what answer the agent is hoping to find. Your only job: given the conversation so far and ONE specific tool call, return the realistic output that tool would produce.

Hard requirements:
- Stay CONSISTENT with everything already established earlier in the transcript (file names, directory layout, line numbers, contents). If you described a file before, keep it the same now.
- Produce plausible, specific, technically coherent content appropriate to the repository described. Real code/config/log text, not placeholders.
- Match the tool's output FORMAT exactly (see the guide).
- Do NOT add commentary, explanations, or markdown fences. Return only the raw tool output text (inside the JSON field).

"""
    + TOOL_OUTPUT_FORMAT_GUIDE
)

DEADEND_INSTRUCTIONS = {
    "file_not_found": "Render a realistic 'file or directory not found' error for the path the agent requested (e.g. an error string the read/bash tool would emit). The path does NOT exist.",
    "empty_grep": "The search ran successfully but found ZERO matches. Return empty or a no-matches result as the tool would (e.g. empty stdout, or exit summary with no lines).",
    "is_a_directory": "The path the agent tried to read is actually a DIRECTORY, not a file. Return the directory-listing format (or an 'is a directory' error if bash).",
    "command_error": "The command FAILED. Return realistic stderr and a nonzero exit (e.g. command not found, syntax error, missing dependency, non-zero status).",
    "red_herring": "Return real, correctly-formatted output that is technically valid but MISLEADING for the agent's goal — it points toward the wrong file/module. Do not flag it as misleading; just return the plausible-but-unhelpful content.",
    "truncated_output": "The output is too large and gets TRUNCATED. Return a realistic partial result that ends with a truncation marker (e.g. '... (output truncated, NNN more lines)'), forcing the agent to narrow its query.",
    "permission_denied": "Return a realistic permission-denied error for the requested path.",
}


def simulate_tool(
    client: LLMClient,
    model: str,
    scenario: Scenario,
    messages: list[dict],
    action: dict,
    deadend_type: Optional[str],
    temperature: float,
    max_tokens: int,
) -> str:
    transcript = render_transcript_for_model(messages)
    tool = action.get("tool", "bash")
    args = action.get("args", {})
    deadend_note = ""
    if deadend_type:
        deadend_note = (
            f"\n\nINJECT A DEAD-END of type '{deadend_type}': "
            + DEADEND_INSTRUCTIONS[deadend_type]
            + " Keep it realistic and in the correct output format."
        )
    user = (
        f"SCENARIO:\n{scenario.describe()}\n\n"
        f"CONVERSATION SO FAR (for consistency):\n{transcript}\n\n"
        f"TOOL CALL TO SIMULATE:\n  tool: {tool}\n  args: {json.dumps(args, ensure_ascii=False)}\n"
        f"{deadend_note}\n\n"
        'Return a JSON object: {"output": "<the raw tool output text>"}'
    )
    obj = call_json(client, model, SIM_SYSTEM, user, temperature, max_tokens)
    if obj is None or "output" not in obj:
        # Fallback so the loop can still proceed; will likely be judged out.
        return f"<error>tool '{tool}' produced no output (simulation failure)</error>"
    out = obj["output"]
    return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)


# =============================================================================
# Trajectory generation: the turn loop. Assembles canonical wire schema.
# =============================================================================
@dataclass
class GenStats:
    deadends_injected: int = 0
    turns: int = 0


def generate_trajectory(
    client: LLMClient,
    scenario: Scenario,
    rng: random.Random,
    agent_model: str,
    sim_model: str,
    max_turns: int,
    p_deadend: float,
    temperature: float,
    sim_temperature: float,
    agent_max_tokens: int,
    sim_max_tokens: int,
    with_reasoning: bool,
    system_prompt_role: Optional[str] = None,
) -> Optional[dict]:
    # Seed the user task. Let the agent phrase nothing here; the user message is
    # a concrete request derived from the scenario.
    user_task = build_user_task(client, scenario, agent_model, temperature)
    if not user_task:
        return None

    messages: list[dict] = [
        {
            "role": "system",
            "content": build_system_prompt(
                scenario, agent_model, role_override=system_prompt_role
            ),
        },
        {"role": "user", "content": user_task},
    ]
    stats = GenStats()

    for turn in range(max_turns):
        force_final = turn == max_turns - 1
        step = agent_step(
            client,
            agent_model,
            scenario,
            messages,
            temperature,
            agent_max_tokens,
            force_final=force_final,
        )
        if step is None:
            return None
        stats.turns += 1

        if step["final"] or not step["actions"]:
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": step["message"],
            }
            if with_reasoning and step.get("thought"):
                assistant_msg["reasoning_content"] = step["thought"]
            messages.append(assistant_msg)
            return {"messages": messages, "_gen_stats": stats.__dict__}

        # Assemble an assistant turn with tool calls (parallel preserved).
        tool_calls = []
        ids: list[str] = []
        for a in step["actions"]:
            tid = gen_tool_id(rng)
            ids.append(tid)
            tool_calls.append(
                {
                    "id": tid,
                    "type": "function",
                    "function": {
                        "name": a.get("tool", "bash"),
                        # arguments MUST be a JSON string.
                        "arguments": json.dumps(a.get("args", {}), ensure_ascii=False),
                    },
                }
            )
        assistant_msg = {
            "role": "assistant",
            "content": step.get("message", ""),
            "tool_calls": tool_calls,
        }
        if with_reasoning and step.get("thought"):
            assistant_msg["reasoning_content"] = step["thought"]
        messages.append(assistant_msg)

        # Simulate each call -> one tool message per call, matched by id.
        for tid, a in zip(ids, step["actions"]):
            deadend_type = None
            if scenario.deadend_profile and rng.random() < p_deadend:
                deadend_type = rng.choice(scenario.deadend_profile)
                stats.deadends_injected += 1
            result = simulate_tool(
                client,
                sim_model,
                scenario,
                messages,
                a,
                deadend_type,
                sim_temperature,
                sim_max_tokens,
            )
            messages.append({"role": "tool", "content": result, "tool_call_id": tid})

    # Shouldn't reach here (last turn is forced final), but guard anyway.
    return None


USER_TASK_SYSTEM = """You write a single realistic user request to a CLI coding agent (OpenCode), as a developer working in the given repository would actually type it. One or two sentences, natural, possibly referencing files with @ syntax. It should require investigation (not answerable without using tools). Return JSON: {"task": "..."}."""


def build_user_task(
    client: LLMClient, scenario: Scenario, model: str, temperature: float
) -> Optional[str]:
    user = (
        f"SCENARIO:\n{scenario.describe()}\n\n"
        "Write the user's opening request. It must align with the underlying goal but be phrased "
        "naturally and concretely (mention plausible file/dir/test names from such a repo). "
        'Return JSON: {"task": "..."}.'
    )
    obj = call_json(client, model, USER_TASK_SYSTEM, user, temperature, max_tokens=1000)
    if obj and isinstance(obj.get("task"), str) and obj["task"].strip():
        return obj["task"].strip()
    return None


# =============================================================================
# Validation: jsonschema per message + referential integrity.
# =============================================================================
_SYSTEM_SCHEMA = {
    "type": "object",
    "required": ["role", "content"],
    "properties": {
        "role": {"const": "system"},
        "content": {"type": "string", "minLength": 1},
    },
    "additionalProperties": False,
}
_USER_SCHEMA = {
    "type": "object",
    "required": ["role", "content"],
    "properties": {
        "role": {"const": "user"},
        "content": {"type": "string", "minLength": 1},
    },
    "additionalProperties": False,
}
_ASSISTANT_TEXT_SCHEMA = {
    "type": "object",
    "required": ["role", "content"],
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string", "minLength": 1},
        "reasoning_content": {"type": "string"},
    },
    "additionalProperties": False,
}
_ASSISTANT_TOOLCALL_SCHEMA = {
    "type": "object",
    "required": ["role", "content", "tool_calls"],
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string"},  # may be empty for tool-only turns
        "reasoning_content": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["id", "type", "function"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "required": ["name", "arguments"],
                        "properties": {
                            "name": {"type": "string", "minLength": 1},
                            "arguments": {"type": "string"},  # JSON string
                        },
                        "additionalProperties": False,
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}
_TOOL_SCHEMA = {
    "type": "object",
    "required": ["role", "content", "tool_call_id"],
    "properties": {
        "role": {"const": "tool"},
        "content": {"type": "string"},
        "tool_call_id": {"type": "string", "minLength": 1},
    },
    "additionalProperties": False,
}


class TrajectoryError(Exception):
    pass


def validate_trajectory(messages: list[dict], max_length: int = 60) -> tuple[bool, str]:
    """Strict structural validation. Returns (ok, reason)."""
    if not isinstance(messages, list) or not (3 <= len(messages) <= max_length):
        return (
            False,
            f"bad length: {len(messages) if isinstance(messages, list) else 'n/a'}",
        )

    # First two roles fixed.
    try:
        jsonschema.validate(messages[0], _SYSTEM_SCHEMA)
        jsonschema.validate(messages[1], _USER_SCHEMA)
    except jsonschema.ValidationError as e:
        return False, f"system/user envelope: {e.message}"

    if messages[-1].get("role") != "assistant" or messages[-1].get("tool_calls"):
        return False, "final message must be an assistant text turn"

    # Per-message schema.
    for idx, m in enumerate(messages):
        role = m.get("role")
        try:
            if role == "system":
                jsonschema.validate(m, _SYSTEM_SCHEMA)
            elif role == "user":
                jsonschema.validate(m, _USER_SCHEMA)
            elif role == "assistant":
                if m.get("tool_calls"):
                    jsonschema.validate(m, _ASSISTANT_TOOLCALL_SCHEMA)
                    for tc in m["tool_calls"]:
                        try:
                            json.loads(tc["function"]["arguments"])
                        except json.JSONDecodeError:
                            return (
                                False,
                                f"msg {idx}: tool_call arguments not valid JSON string",
                            )
                else:
                    jsonschema.validate(m, _ASSISTANT_TEXT_SCHEMA)
            elif role == "tool":
                jsonschema.validate(m, _TOOL_SCHEMA)
            else:
                return False, f"msg {idx}: unknown role {role!r}"
        except jsonschema.ValidationError as e:
            return False, f"msg {idx} ({role}): {e.message}"

    # Referential integrity + grouping/ordering.
    ok, reason = _check_referential_integrity(messages)
    if not ok:
        return False, reason
    return True, "ok"


def _check_referential_integrity(messages: list[dict]) -> tuple[bool, str]:
    seen_ids: set[str] = set()
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            ids = [tc["id"] for tc in m["tool_calls"]]
            if len(ids) != len(set(ids)):
                return False, f"duplicate tool_call ids in msg {i}"
            for tid in ids:
                if tid in seen_ids:
                    return False, f"tool_call id reused: {tid}"
                seen_ids.add(tid)
            # The next len(ids) messages must be tool messages answering exactly these ids.
            following = messages[i + 1 : i + 1 + len(ids)]
            if len(following) != len(ids) or any(
                f.get("role") != "tool" for f in following
            ):
                return (
                    False,
                    f"msg {i}: tool calls not followed by matching tool results",
                )
            answered = {f["tool_call_id"] for f in following}
            if answered != set(ids):
                return (
                    False,
                    f"msg {i}: tool result ids {answered} != call ids {set(ids)}",
                )
            i += 1 + len(ids)
        elif role == "tool":
            # Any tool message reaching here was not consumed by a preceding group.
            return False, f"orphan tool message at index {i}"
        else:
            i += 1
    return True, "ok"


# =============================================================================
# LLM judge gate.
# =============================================================================
JUDGE_SYSTEM = """You are a strict reviewer of synthetic coding-agent trajectories used for supervised fine-tuning. The tool outputs in the trajectory were SIMULATED, so be alert to incoherence. Score the trajectory on 1-5 scales.

Criteria:
- coherence: do the agent's actions form a sensible investigation; does the conversation hang together?
- tool_use: are tool calls well-chosen, well-formed, and appropriately parallel/sequential?
- faithfulness: does the agent's reasoning and FINAL answer follow ONLY from what the tool outputs actually showed? Penalize claims not supported by any tool result (hallucinated facts).
- recovery: if any tool output was an error / empty / misleading dead-end, did the agent NOTICE it and adapt, rather than ignoring it or hallucinating through it? If there was no dead-end, set recovery to null.

Return JSON:
{"coherence":1-5,"tool_use":1-5,"faithfulness":1-5,"recovery":1-5 or null,"reasons":"one or two sentences"}"""


def judge_trajectory(
    client: LLMClient, model: str, messages: list[dict], temperature: float = 0.0
) -> Optional[dict]:
    transcript = render_transcript_for_model(messages)
    user = f"TRAJECTORY:\n{transcript}\n\nScore it. Return only the JSON object."
    obj = call_json(client, model, JUDGE_SYSTEM, user, temperature, max_tokens=2000)
    if obj is None:
        return None
    return obj


def judge_keep(scores: dict, threshold: float) -> bool:
    try:
        core = [
            float(scores["coherence"]),
            float(scores["tool_use"]),
            float(scores["faithfulness"]),
        ]
    except (KeyError, TypeError, ValueError):
        return False
    if min(core) < 2:  # any hard failure on a core axis
        return False
    rec = scores.get("recovery")
    if rec is not None:
        try:
            if (
                float(rec) <= 2
            ):  # a dead-end that wasn't recovered from is a bad example
                return False
        except (TypeError, ValueError):
            return False
    return (sum(core) / len(core)) >= threshold


# =============================================================================
# Near-duplicate filtering (lexical; no extra deps or API).
# For stronger semantic dedup, embed `dedup_signature` and cosine-filter instead.
# =============================================================================
def dedup_signature(traj: dict) -> str:
    msgs = traj["messages"]
    user_text = next((m["content"] for m in msgs if m["role"] == "user"), "")
    tool_seq = []
    for m in msgs:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tool_seq.extend(tc["function"]["name"] for tc in m["tool_calls"])
    return (user_text + " || " + ">".join(tool_seq)).lower()


def _shingles(text: str, k: int = 3) -> set[str]:
    toks = text.split()
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def filter_near_duplicates(trajectories: list[dict], threshold: float) -> list[dict]:
    kept: list[dict] = []
    kept_shingles: list[set[str]] = []
    for traj in trajectories:
        sh = _shingles(dedup_signature(traj))
        if any(jaccard(sh, prev) >= threshold for prev in kept_shingles):
            continue
        kept.append(traj)
        kept_shingles.append(sh)
    return kept


# =============================================================================
# Main.
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic opencode-sft trajectories (v2)."
    )
    parser.add_argument("--num-to-generate", type=int, default=20)
    parser.add_argument(
        "--output-file", type=str, default="synthetic_conversations.json"
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=None,
        help=(
            "JSONL file appended (and flushed) after each kept trajectory so progress "
            "survives an interruption. Defaults to '<output-file>.jsonl'. "
            "Pass '' to disable."
        ),
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["modal", "openrouter"],
        default="modal",
        help="Inference backend: 'modal' (GLM endpoint) or 'openrouter'.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override the provider base URL (defaults per --provider).",
    )
    parser.add_argument(
        "--agent-model",
        type=str,
        default=None,
        help=(
            "Teacher policy model. Defaults to 'zai-org/GLM-5.2-FP8' for modal "
            "and 'z-ai/glm-5.2' for openrouter."
        ),
    )
    parser.add_argument(
        "--sim-model",
        type=str,
        default=None,
        help="Environment simulator model (defaults to --agent-model).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Judge model (defaults to --agent-model). Prefer a strong, separate model.",
    )
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument(
        "--p-deadend",
        type=float,
        default=0.3,
        help="Per-tool-call probability of injecting a dead-end (only when scenario has an eligible profile).",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--sim-temperature", type=float, default=0.6)
    parser.add_argument("--agent-max-tokens", type=int, default=2500)
    parser.add_argument("--sim-max-tokens", type=int, default=3000)
    parser.add_argument("--judge-threshold", type=float, default=3.5)
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.8,
        help="Jaccard >= this on the signature => treated as a near-duplicate.",
    )
    parser.add_argument(
        "--with-reasoning",
        action="store_true",
        help="Emit reasoning_content on assistant turns. Off by default to match the canonical replay trace.",
    )
    parser.add_argument(
        "--no-judge", action="store_true", help="Skip the LLM judge gate."
    )
    parser.add_argument(
        "--system-prompt-role",
        type=str,
        default=None,
        help=(
            "Force a specific harvested opencode prompt role for the system message "
            "(e.g. build, beast, explore). Default: auto-select from the agent model "
            "(claude/anthropic -> beast, otherwise build)."
        ),
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Pass extra_body reasoning=enabled to the provider (model-specific).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--upload-to-hf",
        action="store_true",
        help="Upload the generated dataset to Hugging Face Hub after saving.",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="opencode-sft-synthetic",
        help="HF dataset repo name (default: opencode-sft-synthetic).",
    )
    args = parser.parse_args()

    # Resolve the agent model default per provider.
    if args.agent_model is None:
        args.agent_model = (
            "z-ai/glm-5.2" if args.provider == "openrouter" else "zai-org/GLM-5.2-FP8"
        )

    # Build the provider client.
    if args.provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("OPENROUTER_API_KEY not found in environment", file=sys.stderr)
            sys.exit(1)
        client = LLMClient.openrouter(
            api_key=api_key,
            base_url=args.base_url or OPENROUTER_BASE_URL,
            reasoning=args.reasoning,
        )
    else:
        modal_key_id = os.getenv("MODAL_PROXY_AUTH_TOKEN_ID")
        modal_key_secret = os.getenv("MODAL_PROXY_AUTH_TOKEN_SECRET")
        if not modal_key_id or not modal_key_secret:
            print(
                "MODAL_PROXY_AUTH_TOKEN_ID / MODAL_PROXY_AUTH_TOKEN_SECRET not found in environment",
                file=sys.stderr,
            )
            sys.exit(1)
        client = LLMClient.modal(
            key_id=modal_key_id,
            key_secret=modal_key_secret,
            base_url=args.base_url or MODAL_BASE_URL,
            reasoning=args.reasoning,
        )

    rng = random.Random(args.seed)
    sim_model = args.sim_model or args.agent_model
    judge_model = args.judge_model or args.agent_model

    # Resolve (and validate) the system-prompt role up front so a typo or stale
    # fixture fails loudly before we burn any API calls.
    try:
        available_roles = sorted(_load_opencode_prompts())
    except (OSError, json.JSONDecodeError, KeyError) as e:
        print(
            f"Could not load opencode prompt fixtures ({HARNESS_FIXTURE}): {e}\n"
            "Re-run scripts/harvest-harness.js to (re)generate them.",
            file=sys.stderr,
        )
        sys.exit(1)
    prompt_role = args.system_prompt_role or select_prompt_role(args.agent_model)
    if prompt_role not in available_roles:
        print(
            f"System-prompt role '{prompt_role}' not found in opencode-harness.json "
            f"(available: {', '.join(available_roles) or '<none>'}).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Provider        : {args.provider}{' (reasoning)' if args.reasoning else ''}")
    print(f"Agent model     : {args.agent_model}")
    print(f"Simulator model : {sim_model}")
    print(f"Judge model     : {judge_model}{' (DISABLED)' if args.no_judge else ''}")
    print(
        f"System prompt   : opencode '{prompt_role}' role"
        f"{' (forced)' if args.system_prompt_role else ' (auto)'}"
    )
    print(f"Generating up to {args.num_to_generate} trajectories...\n")

    # Incremental checkpoint: append each kept trajectory as JSONL so a long run
    # can be interrupted/resumed without losing progress. Default alongside the
    # output file; pass --checkpoint-file '' to disable.
    if args.checkpoint_file is None:
        checkpoint_path = args.output_file + ".jsonl"
    else:
        checkpoint_path = args.checkpoint_file or None
    checkpoint_fh = None
    if checkpoint_path:
        checkpoint_fh = open(checkpoint_path, "a", encoding="utf-8")
        print(f"Checkpoint      : appending kept trajectories to {checkpoint_path}\n")

    generated: list[dict] = []
    n_struct_fail = 0
    n_judge_fail = 0
    n_gen_fail = 0

    for i in range(args.num_to_generate):
        print(f"({i + 1}/{args.num_to_generate}) generating...")
        scenario = sample_scenario(rng)
        traj = generate_trajectory(
            client,
            scenario,
            rng,
            agent_model=args.agent_model,
            sim_model=sim_model,
            max_turns=args.max_turns,
            p_deadend=args.p_deadend,
            temperature=args.temperature,
            sim_temperature=args.sim_temperature,
            agent_max_tokens=args.agent_max_tokens,
            sim_max_tokens=args.sim_max_tokens,
            with_reasoning=args.with_reasoning,
            system_prompt_role=args.system_prompt_role,
        )
        if traj is None:
            n_gen_fail += 1
            print("  x generation failed")
            continue

        ok, reason = validate_trajectory(traj["messages"])
        if not ok:
            n_struct_fail += 1
            print(f"  x structural validation failed: {reason}")
            continue

        gen_stats = traj.pop("_gen_stats", {})
        if not args.no_judge:
            scores = judge_trajectory(client, judge_model, traj["messages"])
            if scores is None or not judge_keep(scores, args.judge_threshold):
                n_judge_fail += 1
                print(f"  x judged out: {scores}")
                continue
            traj["judge_scores"] = scores

        traj["source"] = "synthetic"
        traj["scenario"] = {
            "repo": scenario.repo,
            "task_type": scenario.task_type,
            "deadend_profile": scenario.deadend_profile,
            "deadends_injected": gen_stats.get("deadends_injected", 0),
        }
        generated.append(traj)
        if checkpoint_fh is not None:
            checkpoint_fh.write(json.dumps(traj, ensure_ascii=False) + "\n")
            checkpoint_fh.flush()
            os.fsync(checkpoint_fh.fileno())
        print(f"  ok (kept: {len(generated)})")
        time.sleep(0.3)

    if checkpoint_fh is not None:
        checkpoint_fh.close()

    before_dedup = len(generated)
    generated = filter_near_duplicates(generated, args.dedup_threshold)
    n_dupes = before_dedup - len(generated)

    attempted = args.num_to_generate
    print("\nSummary")
    print(f"  attempted        : {attempted}")
    print(f"  generation fails : {n_gen_fail}")
    print(f"  structural fails : {n_struct_fail}")
    print(f"  judged out       : {n_judge_fail}")
    print(f"  near-duplicates  : {n_dupes}")
    print(f"  kept             : {len(generated)}")
    if attempted:
        print(f"  yield            : {len(generated) / attempted * 100:.1f}%")

    if generated:
        with open(args.output_file, "w") as f:
            json.dump(generated, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(generated)} trajectories to {args.output_file}")

        if args.upload_to_hf:
            upload_script = os.path.join(os.path.dirname(__file__), "upload_to_hf.py")
            print(f"Uploading to HF repo '{args.hf_repo}' ...")
            # Convert JSON array to JSONL and pipe to upload_to_hf.py
            jsonl_bytes = "\n".join(json.dumps(t, ensure_ascii=False) for t in generated).encode("utf-8")
            proc = subprocess.run(
                [sys.executable, upload_script, "--repo", args.hf_repo],
                input=jsonl_bytes,
                capture_output=True,
                timeout=120,
            )
            result = json.loads(proc.stdout.decode())
            if result.get("ok"):
                print(f"  Uploaded {result['records']} records -> {result['url']}")
            else:
                print(f"  Upload failed: {result.get('error', proc.stderr.decode())}", file=sys.stderr)
    else:
        print("\nNothing kept; no file written.")


if __name__ == "__main__":
    main()
