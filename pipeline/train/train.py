#!/usr/bin/env python3
"""LoRA fine-tune a local code model on collected traces. Runs on a CUDA GPU.

Defaults to `Qwen/Qwen3.6-35B-A3B` (35B-total / 3B-active hybrid MoE). Tries
Unsloth first (handles the multimodal model's text-only path, the MoE kernels,
and gradient checkpointing); falls back to plain transformers + PEFT + TRL.

Loss is masked to assistant turns only. When a chat template marks assistant
spans with Jinja `{% generation %}` blocks (e.g. Laguna) we get an exact
assistant-token mask straight from `apply_chat_template(...,
return_assistant_tokens_mask=True)`. Qwen's template has no generation blocks,
so we fall back to masking between configured turn markers (`<|im_start|>
assistant\\n` … `<|im_end|>`).

Input:
    --data sft.jsonl (one {"messages": [...], "tools": [...]|null, "source": ...}
        per line) or --hf-repo to pull from HF. A single-split HF dataset (e.g.
        nkasmanoff/opencode-sft) is split into train/eval via --eval-frac.
Output: <out>/adapter/  (LoRA weights)
        <out>/merged/   (merged bf16 checkpoint, ready for GGUF conversion)

Usage (on the pod):
    # Pull + split a single-split HF dataset and train Qwen3.6:
    python train.py --hf-repo nkasmanoff/opencode-sft --out outputs

    # Local files (e.g. from pull_dataset.py / `make split`):
    python train.py --data dataset/sft.jsonl --eval-data dataset/sft-eval.jsonl \
        --out outputs --epochs 3 --lr 5e-5
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

# Default base. Qwen3.6-35B-A3B is a 35B-total / 3B-active hybrid MoE
# (~30 Gated DeltaNet linear-attention layers + ~10 standard attention layers +
# 256 experts). bf16 LoRA fits one 80GB H100 (~74GB), which is why the default
# is bf16 (`--quant none`): bitsandbytes 4-bit QLoRA is *not recommended* for
# this MoE family (large quantization error / instability), and the fused
# experts don't quantize cleanly anyway. An FP8 build exists for tighter VRAM.
QWEN_BASE = "Qwen/Qwen3.6-35B-A3B"
QWEN_FP8_BASE = "Qwen/Qwen3.6-35B-A3B-FP8"

# Legacy Laguna bases (still selectable via --base).
LAGUNA_BASE = "poolside/Laguna-XS.2"
# FP8 (compressed-tensors) build: the MoE experts are FP8-compressed while the
# attention projections stay bf16, so the model fits on one 80GB H100 and LoRA
# wraps the un-quantized attention linears. Plain bitsandbytes 4-bit can't
# quantize Laguna's fused `LagunaExperts` module (94% of params).
LAGUNA_FP8_BASE = "poolside/Laguna-XS.2-FP8"

# Only train on trajectories produced by strong teacher models. Rows whose
# producing model matches none of these substrings (case-insensitive) are
# dropped by default — local-model traces (laguna, north-mini, qwen, ...) are
# noisier and would teach the student its own bad habits. Override with
# --allow-models (comma-separated substrings, or "all" to disable filtering).
DEFAULT_ALLOWED_MODELS = ("big-pickle", "claude")

# Fallback for rows with no "model" field (build_dataset.py's quality filter
# strips it): the viewer's exporter labels rows by upstream provider —
# source="frontier" for openrouter/claude teachers, source="opencode" for the
# opencode provider (big-pickle). Local providers also land in "opencode", so
# this mapping is a heuristic; rows exported straight from the viewer carry the
# exact model id and never hit it.
_MODEL_SOURCE_FALLBACK = {"big-pickle": "opencode", "claude": "frontier"}

# Per-chat-format settings. `apply_chat_template(return_assistant_tokens_mask=
# True)` is preferred for masking; the marker fields are only used as a fallback
# when a template has no `{% generation %}` blocks (so no assistant mask).
CHAT_FORMATS = {
    "qwen": {
        # template reads reasoning straight from message.reasoning_content
        "rename_reasoning": False,
        # dataset stores tool_call arguments as JSON strings; the template
        # iterates arguments.items(), so they must be parsed back to dicts
        "normalize_tool_args": True,
        # enable_thinking keeps the <think> spans; preserve_thinking is the
        # Qwen3.6 flag that renders reasoning from *historical* assistant turns
        # too (not just the last one) — without it, earlier-turn reasoning is
        # dropped from the rendered text and never trained on.
        "template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
        # Qwen template has no {% generation %} blocks, so masking falls back to
        # these markers: an assistant turn opens with `<|im_start|>assistant\n`
        # and closes with `<|im_end|>`.
        "response_marker": "<|im_start|>assistant\n",
        "instruction_marker": "<|im_start|>user\n",
        "end_marker": "<|im_end|>",
        "preflight_needles": [
            ("assistant turn", "<|im_start|>assistant"),
            ("reasoning/think", "PREFLIGHT_THINK"),
            ("tool call", "<function=calc"),
        ],
    },
    "laguna": {
        # template reads reasoning straight from message.reasoning_content
        "rename_reasoning": False,
        # dataset stores tool_call arguments as JSON strings; the template
        # iterates arguments.items(), so they must be parsed back to dicts
        "normalize_tool_args": True,
        "template_kwargs": {"enable_thinking": True},
        "response_marker": "<assistant>\n",
        "instruction_marker": "<user>\n",
        "end_marker": "</assistant>",
        # (label, needle) pairs the preflight requires in a rendered sample
        "preflight_needles": [
            ("assistant turn", "<assistant>"),
            ("reasoning/think", "PREFLIGHT_THINK"),
            ("tool call", "<tool_call>calc"),
        ],
    },
    "cohere": {
        "rename_reasoning": True,
        "normalize_tool_args": False,
        "template_kwargs": {},
        "response_marker": "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
        "instruction_marker": "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
        "end_marker": "<|END_OF_TURN_TOKEN|>",
        "preflight_needles": [
            ("response marker", "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"),
            ("reasoning/thinking", "PREFLIGHT_THINK"),
            ("tool name", "calc"),
        ],
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=None,
                   help="local sft jsonl file (mutually exclusive with --hf-repo)")
    p.add_argument("--hf-repo", default=None,
                   help="HuggingFace dataset repo to pull from, e.g. "
                        "nkasmanoff/opencode-sft")
    p.add_argument("--eval-data", type=Path, default=None,
                   help="held-out sft jsonl; if set, eval loss is computed "
                        "every --eval-steps")
    p.add_argument("--eval-frac", type=float, default=0.1,
                   help="when --hf-repo has only a train split, hold out this "
                        "fraction for eval")
    p.add_argument("--eval-steps", type=int, default=20)
    p.add_argument("--no-eval", action="store_true",
                   help="skip held-out eval loss (avoids VRAM spike on long context)")
    p.add_argument("--allow-models", default=",".join(DEFAULT_ALLOWED_MODELS),
                   help="comma-separated substrings of teacher model ids to "
                        "train on (matched case-insensitively against each "
                        "row's 'model' field, falling back to its 'source' "
                        "label). Rows from other models are dropped. "
                        "Pass 'all' to disable filtering. "
                        f"Default: {','.join(DEFAULT_ALLOWED_MODELS)}")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--base", default=QWEN_BASE)
    p.add_argument("--quant", choices=["auto", "fp8", "4bit", "none"],
                   default="auto",
                   help="base-model quantization. auto: 'fp8' (native "
                         "compressed-tensors) for *-FP8 repos; 'none' (bf16 LoRA) "
                         "for Qwen MoE (4-bit QLoRA is discouraged for this MoE "
                         "family); '4bit' (bitsandbytes QLoRA) for dense models. "
                         "'none' bf16 fits one 80GB H100 for Qwen3.6-35B-A3B; "
                         "shards across GPUs otherwise.")
    p.add_argument("--chat-format", choices=["auto", "qwen", "laguna", "cohere"],
                   default="auto",
                   help="chat template family (auto-detected from --base)")
    # 65535 avoids Unsloth's patch_sdpa_bool_causal_mask edge at seq >= 2^16
    # (65536), which materializes a full SDPA mask and OOMs at long context.
    p.add_argument("--max-seq-len", type=int, default=65535,
                   help="max tokens per sample (default 65535; keep below 2^16 "
                        "because Unsloth's SDPA patch switches to a mask-"
                        "materializing path at seq >= 65536)")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument(
        "--include-mlp", action="store_true",
        help="also adapt expert FFN projections (more VRAM, more capacity)",
    )
    p.add_argument("--no-liger", action="store_true",
                   help="disable the Liger fused-linear cross-entropy patch "
                        "(it avoids materializing the [seq, vocab] logits "
                        "tensor, so longer context fits in memory)")
    p.add_argument("--no-unsloth", action="store_true",
                   help="use transformers + PEFT instead of Unsloth. Required "
                        "for Qwen3.6 at 64k: Unsloth's compiled linear-attn "
                        "falls back to a torch path that OOMs; transformers "
                        "uses flash-linear-attention when installed.")
    p.add_argument("--no-merge", action="store_true",
                   help="skip writing the merged bf16 checkpoint")
    p.add_argument("--wandb-project", default="qwen-sft",
                   help="W&B project; logging activates when WANDB_API_KEY "
                        "is set (use --no-wandb to force off)")
    p.add_argument("--run-name", default=None,
                   help="W&B run name (default: auto from data + params)")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--max-samples", type=int, default=0,
                   help="cap train samples (0 = all); for quick smoke tests")
    # Intermittent eval gate: stand up an in-process OpenAI server backed by the
    # live (in-training) weights and run eval/run_evals.py against it, logging
    # the section pass rates to W&B. See train/eval_server.py.
    p.add_argument("--eval-gate", action="store_true",
                   help="run eval/run_evals.py periodically during training "
                        "against the in-training model, logging to W&B")
    p.add_argument("--eval-gate-steps", type=int, default=20,
                   help="run the eval gate every N optimizer steps (0 = off)")
    p.add_argument("--eval-gate-epochs", type=int, default=0,
                   help="run the eval gate every N epochs (0 = off); set with "
                        "--eval-gate-steps 0 for an epoch-only cadence")
    p.add_argument("--eval-gate-at-start", action="store_true", default=True,
                   help="also run the gate once before training (step 0)")
    p.add_argument("--no-eval-gate-at-start", dest="eval_gate_at_start",
                   action="store_false")
    p.add_argument("--eval-gate-port", type=int, default=8848)
    p.add_argument("--eval-gate-max-new-tokens", type=int, default=2048)
    p.add_argument("--eval-gate-opencode", action="store_true",
                   help="include the (slow, streaming) opencode-tasks section; "
                        "off by default — tool-validity + chat-sanity only")
    p.add_argument("--eval-gate-problem-pack", action="store_true",
                   help="also run the agent-problem-pack benchmark against the "
                        "in-training model at each eval-gate cadence (logs "
                        "pack pass rate to W&B). Needs opencode + uv on PATH.")
    p.add_argument("--eval-gate-pack-subset", default="smoke",
                   choices=["smoke", "easy", "medium", "hard", "repair",
                            "comprehension", "all"],
                   help="which agent-problem-pack problems the gate runs "
                        "(default: a quick smoke subset)")
    p.add_argument("--eval-gate-pack-timeout", type=int, default=900,
                   help="per-problem opencode timeout for the pack gate")
    p.add_argument("--eval-gate-skip-core", action="store_true",
                   help="skip the run_evals.py core sections in the gate "
                        "(use with --eval-gate-problem-pack to run only the pack)")
    p.add_argument("--push-to-hub", default=None, metavar="REPO_ID",
                   help="after training, upload the LoRA adapter "
                        "(outputs/adapter/) to this HF repo, e.g. "
                        "user/laguna-opencode-lora. Uses HF_TOKEN.")
    p.add_argument("--push-merged", action="store_true",
                   help="also upload the merged bf16 checkpoint to "
                        "<repo>-merged (large; non-fp8 paths only)")
    p.add_argument("--hub-private", action="store_true",
                   help="create the pushed HF repo(s) as private")
    return p.parse_args()


def load_dotenv(start: Path) -> None:
    import os
    for d in [start, *start.parents]:
        env = d / ".env"
        if env.is_file():
            for line in env.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
            print(f"loaded env from {env}")
            return


def normalize_text_tokenizer(tokenizer):
    """Qwen loads as a multimodal Processor; SFT needs the text tokenizer."""
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None and hasattr(inner, "apply_chat_template"):
        return inner
    return tokenizer


def resolve_quant(args) -> str:
    if args.quant != "auto":
        return args.quant
    base = (args.base or "").lower()
    if "fp8" in base or "compressed" in base:
        return "fp8"
    # bf16 LoRA for Qwen MoE (A3B): 4-bit QLoRA is discouraged for the MoE family.
    # Dense Qwen (e.g. Qwen3.6-27B) needs 4-bit at 64k on 140GB — bf16 OOMs.
    if "qwen" in base and ("a3b" in base or "moe" in base):
        return "none"
    return "4bit"


def resolve_chat_format(args) -> str:
    if args.chat_format != "auto":
        return args.chat_format
    base = (args.base or "").lower()
    if "qwen" in base:
        return "qwen"
    if "laguna" in base:
        return "laguna"
    if any(k in base for k in ("cohere", "command", "north-mini", "north_mini")):
        return "cohere"
    # default base is Qwen; assume qwen for unknown ids
    return "qwen"


def setup_wandb(args, fmt: str, n_train: int, n_eval: int) -> str | None:
    """Init W&B if a key is available. Returns run name or None."""
    import os
    if args.no_wandb or not os.environ.get("WANDB_API_KEY"):
        return None
    import time
    import wandb
    run_name = args.run_name or (
        f"sft-{fmt}-{n_train}samp-r{args.lora_r}-lr{args.lr:g}"
        f"-{time.strftime('%m%d-%H%M')}")
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "base_model": args.base,
            "chat_format": fmt,
            "allow_models": args.allow_models,
            "n_train_samples": n_train,
            "n_eval_samples": n_eval,
            "max_seq_len": args.max_seq_len,
            "epochs": args.epochs,
            "lr": args.lr,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "include_mlp": args.include_mlp,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
        },
    )
    return run_name


# Per-format LoRA target modules. PEFT/Unsloth match these as module-name
# suffixes. Router/gate module names are excluded on purpose — adapting the MoE
# router destabilizes training.
_ATTN_TARGETS = {
    "laguna": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "cohere": ["q_proj", "k_proj", "v_proj", "o_proj"],
    # Qwen3.6 is hybrid: ~30 Gated DeltaNet (linear-attention) layers + ~10
    # standard attention layers. q/k/v/o_proj only reach the 10 standard layers
    # (~0.02% of params) and training NaNs — we MUST also adapt the DeltaNet
    # linear_attn projections (in_proj_qkv / in_proj_z / out_proj).
    "qwen": ["q_proj", "k_proj", "v_proj", "o_proj",
             "in_proj_qkv", "in_proj_z", "out_proj"],
}
# Expert FFN projections, added when --include-mlp is set.
_MLP_TARGETS = {
    "laguna": ["gate_proj", "up_proj", "down_proj"],
    "cohere": ["gate_proj", "up_proj", "down_proj"],
    "qwen": ["gate_proj", "up_proj", "down_proj"],  # MoE switch_mlp experts
}


def lora_targets(fmt: str, include_mlp: bool) -> list[str]:
    targets = list(_ATTN_TARGETS[fmt])
    if include_mlp:
        targets += _MLP_TARGETS[fmt]
    return targets


def prepare_messages(rec: dict, fmt_cfg: dict) -> list[dict]:
    """Normalize one record's messages for the target chat template."""
    messages = []
    for m in rec["messages"]:
        m = dict(m)
        if fmt_cfg["rename_reasoning"] and m.get("reasoning_content"):
            # the Cohere template renders assistant thinking via msg.thinking
            m["thinking"] = m.pop("reasoning_content")
        if fmt_cfg["normalize_tool_args"] and m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                tc = copy.deepcopy(tc)
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        pass
                tcs.append(tc)
            m["tool_calls"] = tcs
        messages.append(m)
    return messages


def render_text(rec: dict, tokenizer, fmt_cfg: dict) -> str:
    """Render a record to text (for the human-readable preflight check)."""
    return tokenizer.apply_chat_template(
        prepare_messages(rec, fmt_cfg),
        tools=rec.get("tools") or None,
        tokenize=False,
        add_generation_prompt=False,
        **fmt_cfg["template_kwargs"],
    )


def _find(haystack: list[int], needle: list[int], start: int) -> int:
    if not needle:
        return -1
    for i in range(start, len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1


def marker_mask(ids: list[int], resp_ids: list[int],
                end_ids: list[int]) -> list[int]:
    """Fallback assistant mask: 1 between a response marker and the following
    end-of-turn token (inclusive of the end token so the model learns to stop).
    Used only when the template has no `{% generation %}` blocks."""
    mask = [0] * len(ids)
    pos = 0
    while True:
        start = _find(ids, resp_ids, pos)
        if start == -1:
            break
        span_start = start + len(resp_ids)
        end = _find(ids, end_ids, span_start) if end_ids else -1
        span_end = (end + len(end_ids)) if end != -1 else len(ids)
        for i in range(span_start, span_end):
            mask[i] = 1
        pos = span_end
    return mask


def tokenize_record(rec: dict, tokenizer, fmt_cfg: dict,
                    resp_ids: list[int], end_ids: list[int],
                    ) -> tuple[list[int], list[int]]:
    """Return (input_ids, assistant_mask) for one record. Prefers the chat
    template's `{% generation %}` mask; falls back to marker masking."""
    messages = prepare_messages(rec, fmt_cfg)
    tools = rec.get("tools") or None
    enc = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
        **fmt_cfg["template_kwargs"],
    )
    ids = list(enc["input_ids"])
    mask = enc.get("assistant_masks")
    if not mask or sum(mask) == 0:
        mask = marker_mask(ids, resp_ids, end_ids)
    return ids, list(mask)


def preflight_template(tokenizer, fmt_cfg: dict,
                       resp_ids: list[int], end_ids: list[int]) -> None:
    """Fail loudly if the chat template can't render/mask reasoning + tool
    calls. A bare try/except in the data loader would otherwise silently drop
    every agentic sample and train on nothing useful."""
    rec = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant",
             "content": "",
             "reasoning_content": "PREFLIGHT_THINK the user wants a sum.",
             "tool_calls": [{
                 "id": "call_0", "type": "function",
                 "function": {"name": "calc",
                              "arguments": "{\"expr\": \"2+2\"}"}}]},
            {"role": "tool", "tool_call_id": "call_0", "content": "4"},
            {"role": "assistant", "content": "It is 4."},
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "calc",
                "description": "Evaluate an arithmetic expression.",
                "parameters": {
                    "type": "object",
                    "properties": {"expr": {"type": "string"}},
                    "required": ["expr"],
                }}}],
    }
    try:
        text = render_text(rec, tokenizer, fmt_cfg)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"FATAL: chat template failed to render a tool-call + reasoning "
            f"sample: {exc!r}\nThe collected traces are agentic (tools + "
            f"reasoning_content); a template that can't render them would "
            f"drop every sample. Check the tokenizer for {tokenizer.name_or_path}."
        )
    missing = [name for name, needle in fmt_cfg["preflight_needles"]
               if needle not in text]
    if missing:
        raise SystemExit(
            f"FATAL: chat template rendered but is missing {missing}. "
            f"Reasoning and/or tool calls are being dropped on the floor — "
            f"training would not teach the behavior we collected.\n"
            f"--- rendered sample (truncated) ---\n{text[:2000]}"
        )
    _, mask = tokenize_record(rec, tokenizer, fmt_cfg, resp_ids, end_ids)
    if sum(mask) == 0:
        raise SystemExit(
            "FATAL: could not derive an assistant-token mask (template has no "
            "generation blocks and the turn markers did not match). Loss would "
            "be computed over every token. Check --chat-format / the markers.")
    print(f"preflight: template renders reasoning + tool calls and masks "
          f"{sum(mask)} assistant tokens OK")


def parse_allowed_models(spec: str) -> tuple[str, ...] | None:
    """Parse --allow-models into lowercase substrings; None = no filtering."""
    tokens = [t.strip().lower() for t in (spec or "").split(",") if t.strip()]
    if not tokens or "all" in tokens:
        return None
    return tuple(tokens)


def model_allowed(rec: dict, allowed: tuple[str, ...] | None) -> bool:
    """True when the record's producing model is an allowed teacher.

    Prefers the row's explicit "model" id (substring match); rows without one
    (build_dataset.py strips it) fall back to the exporter's "source" label
    via _MODEL_SOURCE_FALLBACK."""
    if allowed is None:
        return True
    model = str(rec.get("model") or "").lower()
    if model:
        return any(token in model for token in allowed)
    source = str(rec.get("source") or "").lower()
    allowed_sources = {_MODEL_SOURCE_FALLBACK[token] for token in allowed
                       if token in _MODEL_SOURCE_FALLBACK}
    return source in allowed_sources


def load_samples(path: Path, tokenizer, fmt_cfg: dict, max_seq_len: int,
                 resp_ids: list[int], end_ids: list[int],
                 allowed_models: tuple[str, ...] | None = None) -> "Dataset":
    """Tokenize each conversation and attach an assistant-only loss mask."""
    from datasets import Dataset

    rows = []
    skipped_render = skipped_long = skipped_nomask = skipped_model = 0
    first_error: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if not model_allowed(rec, allowed_models):
            skipped_model += 1
            continue
        try:
            ids, mask = tokenize_record(rec, tokenizer, fmt_cfg,
                                        resp_ids, end_ids)
        except Exception as exc:  # noqa: BLE001 — bad sample, skip
            skipped_render += 1
            if first_error is None:
                first_error = repr(exc)
            continue
        if len(ids) >= max_seq_len:
            skipped_long += 1
            continue
        if sum(mask) == 0:
            skipped_nomask += 1
            continue
        # precompute labels (standard column survives TRL/Trainer column
        # pruning); loss is masked to assistant tokens, -100 elsewhere
        labels = [tok if m else -100 for tok, m in zip(ids, mask)]
        rows.append({"input_ids": ids, "labels": labels})
    msg = (f"dataset: {len(rows)} samples from {path} "
           f"(skipped {skipped_render} unrenderable, {skipped_long} over "
           f"{max_seq_len} tok, {skipped_nomask} no-assistant")
    if allowed_models is not None:
        msg += f", {skipped_model} non-teacher-model"
    msg += ")"
    if first_error:
        msg += f"  [first render error: {first_error}]"
    print(msg)
    if not rows:
        raise SystemExit(f"FATAL: no usable samples from {path}")
    return Dataset.from_list(rows)


class MaskedCollator:
    """Pad a batch of pre-tokenized rows (input_ids + assistant-masked labels).
    Labels are already -100 outside assistant turns, so loss stays on assistant
    tokens only — the model's reasoning, tool calls, and final answer — never on
    system/user/tool tokens. Padding extends labels with -100."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = (tokenizer.pad_token_id
                       if tokenizer.pad_token_id is not None
                       else tokenizer.eos_token_id)

    def __call__(self, examples: list[dict]) -> dict:
        import torch

        seqs = [list(ex["input_ids"]) for ex in examples]
        labs = [list(ex["labels"]) for ex in examples]
        width = max(len(s) for s in seqs)
        right = self.tokenizer.padding_side != "left"
        input_ids, attn, labels = [], [], []
        for s, lb in zip(seqs, labs):
            pad = width - len(s)
            a = [1] * len(s)
            if right:
                input_ids.append(s + [self.pad_id] * pad)
                attn.append(a + [0] * pad)
                labels.append(lb + [-100] * pad)
            else:
                input_ids.append([self.pad_id] * pad + s)
                attn.append([0] * pad + a)
                labels.append([-100] * pad + lb)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def try_unsloth(args, fmt: str, load_in_4bit: bool):
    """Return (model, tokenizer, is_unsloth) or raise. `load_in_4bit=False`
    loads bf16/16-bit (the recommended path for Qwen MoE)."""
    try:
        # FastModel handles multimodal (text-only) bases; fall back to
        # FastLanguageModel for plain text architectures.
        from unsloth import FastModel as _Fast
    except Exception:  # noqa: BLE001
        from unsloth import FastLanguageModel as _Fast

    model, tokenizer = _Fast.from_pretrained(
        model_name=args.base,
        max_seq_length=args.max_seq_len,
        load_in_4bit=load_in_4bit,
        load_in_16bit=not load_in_4bit,
        full_finetuning=False,
        dtype=None,
    )
    model = _Fast.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=lora_targets(fmt, args.include_mlp),
        use_gradient_checkpointing="unsloth",
        random_state=0,
    )
    model.config.use_cache = False
    try:
        text_cfg = model.config.get_text_config()
        if text_cfg is not None:
            text_cfg.use_cache = False
    except Exception:  # noqa: BLE001 — nested text config may be absent
        pass
    return model, tokenizer, True


def lora_config(args, fmt: str):
    from peft import LoraConfig
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets(fmt, args.include_mlp),
    )


def load_4bit(args, fmt: str):
    """bitsandbytes 4-bit QLoRA path (for architectures whose linears bnb can
    quantize — NOT Laguna, whose fused experts stay bf16)."""
    import torch
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, lora_config(args, fmt))
    model.print_trainable_parameters()
    return model, tokenizer, False


def apply_liger_laguna(model) -> bool:
    """Patch LagunaForCausalLM to compute the LM loss with Liger's fused linear
    cross-entropy. The standard forward materializes a `[seq, vocab]` logits
    tensor (upcast to fp32) — ~20GB at 32k context — which is the single-H100
    bottleneck. Liger fuses the lm_head matmul + cross-entropy and chunks over
    tokens, so the full logits tensor is never built. Laguna isn't in Liger's
    auto-patch list, so we wire it manually. Returns True if applied."""
    if getattr(model.config, "model_type", "") != "laguna":
        return False
    try:
        from liger_kernel.transformers.model.loss_utils import (
            LigerForCausalLMLoss, unpack_cross_entropy_result)
        from liger_kernel.transformers.model.mixtral import (
            LigerMoeCausalLMOutputWithPast)
        from transformers.models.laguna import modeling_laguna as ml
    except Exception as exc:  # noqa: BLE001
        print(f"liger: unavailable ({exc!r}); using standard loss")
        return False

    def lce_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                    past_key_values=None, inputs_embeds=None, labels=None,
                    use_cache=None, output_router_logits=None,
                    logits_to_keep=0, skip_logits=None, **kwargs):
        output_router_logits = (output_router_logits
                                if output_router_logits is not None
                                else self.config.output_router_logits)
        # TRL passes these to control metrics; keep them out of the base model
        return_token_accuracy = kwargs.pop("return_token_accuracy", False)
        kwargs.pop("use_token_scaling", None)
        shift_labels = kwargs.pop("shift_labels", None)

        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_router_logits=output_router_logits, **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = (slice(-logits_to_keep, None)
                         if isinstance(logits_to_keep, int) else logits_to_keep)
        kept = hidden_states[:, slice_indices, :]

        logits = loss = token_accuracy = predicted_tokens = None
        if skip_logits is None:
            skip_logits = self.training and (
                labels is not None or shift_labels is not None)
        if skip_logits:
            # fused: no [seq, vocab] logits tensor is materialized
            result = LigerForCausalLMLoss(
                hidden_states=kept, lm_head_weight=self.lm_head.weight,
                labels=labels, shift_labels=shift_labels,
                hidden_size=self.config.hidden_size,
                return_token_accuracy=return_token_accuracy, **kwargs)
            loss, _, token_accuracy, predicted_tokens = \
                unpack_cross_entropy_result(result)
        else:
            logits = self.lm_head(kept)
            if labels is not None or shift_labels is not None:
                loss = self.loss_function(
                    logits=logits, labels=labels, shift_labels=shift_labels,
                    vocab_size=self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = ml.load_balancing_loss_func(
                outputs.router_logits, self.num_experts,
                self.num_experts_per_tok, attention_mask)
            if loss is not None and aux_loss is not None:
                loss = loss + self.router_aux_loss_coef * aux_loss.to(loss.device)

        return LigerMoeCausalLMOutputWithPast(
            loss=loss, aux_loss=aux_loss, logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            token_accuracy=token_accuracy, predicted_tokens=predicted_tokens)

    ml.LagunaForCausalLM.forward = lce_forward
    return True


def apply_liger_qwen3_5_moe(model) -> bool:
    """Patch Qwen3_5MoeForCausalLM to compute the LM loss with Liger's fused
    linear cross-entropy. Qwen3.6's vocab is ~248k, so the standard forward's
    `[seq, 248k]` logits tensor (upcast to fp32 in cross-entropy) is the single
    biggest VRAM consumer at long context (e.g. ~80GB at 81920 tokens) and the
    reason bf16 LoRA had to drop to short context. Liger fuses the lm_head matmul
    + cross-entropy and chunks over tokens, so the full logits tensor is never
    built — restoring long-context training. The MoE router aux-loss is left on
    the standard path. Qwen3.6 isn't in Liger's auto-patch list, so we wire it
    manually (mirrors apply_liger_laguna). Returns True if applied."""
    if "qwen3_5_moe" not in getattr(model.config, "model_type", ""):
        return False
    try:
        from liger_kernel.transformers.model.loss_utils import (
            LigerForCausalLMLoss, unpack_cross_entropy_result)
        from liger_kernel.transformers.model.mixtral import (
            LigerMoeCausalLMOutputWithPast)
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as mq
    except Exception as exc:  # noqa: BLE001
        print(f"liger: unavailable ({exc!r}); using standard loss")
        return False

    def lce_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                    past_key_values=None, inputs_embeds=None, labels=None,
                    use_cache=None, output_router_logits=None,
                    cache_position=None, logits_to_keep=0, skip_logits=None,
                    **kwargs):
        output_router_logits = (output_router_logits
                                if output_router_logits is not None
                                else self.config.output_router_logits)
        # TRL passes these to control metrics; keep them out of the base model
        return_token_accuracy = kwargs.pop("return_token_accuracy", False)
        kwargs.pop("use_token_scaling", None)
        shift_labels = kwargs.pop("shift_labels", None)

        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position, **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = (slice(-logits_to_keep, None)
                         if isinstance(logits_to_keep, int) else logits_to_keep)
        kept = hidden_states[:, slice_indices, :]

        logits = loss = token_accuracy = predicted_tokens = None
        if skip_logits is None:
            # Fuse whenever labels are present (BOTH train and eval-loss passes)
            # so the [seq, 248k] logits tensor is never built at long context;
            # only generation (labels is None) takes the lm_head path below.
            skip_logits = labels is not None or shift_labels is not None
        if skip_logits:
            # fused: no [seq, vocab] logits tensor is materialized
            result = LigerForCausalLMLoss(
                hidden_states=kept, lm_head_weight=self.lm_head.weight,
                labels=labels, shift_labels=shift_labels,
                hidden_size=self.config.hidden_size,
                return_token_accuracy=return_token_accuracy, **kwargs)
            loss, _, token_accuracy, predicted_tokens = \
                unpack_cross_entropy_result(result)
        else:
            logits = self.lm_head(kept)
            if labels is not None or shift_labels is not None:
                loss = self.loss_function(
                    logits=logits, labels=labels, shift_labels=shift_labels,
                    vocab_size=self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = mq.load_balancing_loss_func(
                outputs.router_logits, self.num_experts,
                self.num_experts_per_tok, attention_mask)
            if loss is not None and aux_loss is not None:
                loss = loss + self.router_aux_loss_coef * aux_loss.to(loss.device)

        return LigerMoeCausalLMOutputWithPast(
            loss=loss, aux_loss=aux_loss, logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            token_accuracy=token_accuracy, predicted_tokens=predicted_tokens)

    mq.Qwen3_5MoeForCausalLM.forward = lce_forward
    return True


def apply_liger_qwen3_5(model) -> bool:
    """Patch Qwen3_5ForCausalLM to compute the LM loss with Liger's fused linear
    cross-entropy. Qwen3.6-27B's vocab is ~248k, so the forward's [seq, 248k]
    logits tensor is the single biggest VRAM consumer at long context — ~80GB at
    82k tokens. Liger fuses the lm_head matmul + cross-entropy so the full logits
    tensor is never built. Returns True if applied."""
    if "qwen3_5" not in getattr(model.config, "model_type", ""):
        return False
    if "moe" in getattr(model.config, "model_type", ""):
        return False  # MoE variant handled by apply_liger_qwen3_5_moe
    try:
        from liger_kernel.transformers.model.loss_utils import (
            LigerForCausalLMLoss, unpack_cross_entropy_result)
        from transformers.modeling_outputs import CausalLMOutputWithPast
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    except Exception as exc:  # noqa: BLE001
        print(f"liger: unavailable ({exc!r}); using standard loss")
        return False

    def lce_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                    past_key_values=None, inputs_embeds=None, labels=None,
                    use_cache=None, logits_to_keep=0, **kwargs):
        # TRL passes these to control metrics; keep them out of the base model
        return_token_accuracy = kwargs.pop("return_token_accuracy", False)
        kwargs.pop("use_token_scaling", None)
        shift_labels = kwargs.pop("shift_labels", None)
        skip_logits = kwargs.pop("skip_logits", None)

        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = (slice(-logits_to_keep, None)
                         if isinstance(logits_to_keep, int) else logits_to_keep)
        kept = hidden_states[:, slice_indices, :]

        logits = loss = token_accuracy = predicted_tokens = None
        if skip_logits is None:
            skip_logits = labels is not None or shift_labels is not None
        if skip_logits:
            result = LigerForCausalLMLoss(
                hidden_states=kept, lm_head_weight=self.lm_head.weight,
                labels=labels, shift_labels=shift_labels,
                hidden_size=self.config.hidden_size,
                return_token_accuracy=return_token_accuracy, **kwargs)
            loss, _, token_accuracy, predicted_tokens = \
                unpack_cross_entropy_result(result)
        else:
            logits = self.lm_head(kept)
            if labels is not None or shift_labels is not None:
                loss = self.loss_function(
                    logits=logits, labels=labels, shift_labels=shift_labels,
                    vocab_size=self.vocab_size, **kwargs)

        output = CausalLMOutputWithPast(
            loss=loss, logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        output.token_accuracy = token_accuracy
        output.predicted_tokens = predicted_tokens
        return output

    mq.Qwen3_5ForCausalLM.forward = lce_forward
    return True


def load_lora(args, fmt: str):
    """LoRA on a model loaded in its native precision (no bitsandbytes).

    Used for FP8 compressed-tensors builds and for plain bf16 (--quant none).
    Gradient checkpointing keeps activation memory in check at long context.

    Note: Qwen3.6 is a multimodal (image-text-to-text) checkpoint. We load the
    text path for text-only SFT; if `AutoModelForCausalLM` can't materialize the
    text model directly, prefer the Unsloth path (handled by the caller)."""
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    # `experts_implementation="grouped_mm"` is essential for the Qwen MoE: the
    # default `batched_mm` path gathers a per-token copy of every selected
    # expert's weights, which blows up VRAM. grouped_mm sorts tokens by expert
    # and uses torch._grouped_mm (Hopper SM90+/torch>=2.9), so no per-token
    # weight copy. For dense models it's silently ignored by the loader.
    load_kwargs = dict(
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base, experts_implementation="grouped_mm", **load_kwargs)
        print("experts: grouped_mm (memory-efficient MoE)")
    except (ValueError, ImportError, TypeError) as exc:
        print(f"experts: grouped_mm unavailable ({exc!r}); using default")
        model = AutoModelForCausalLM.from_pretrained(args.base, **load_kwargs)
    model.config.use_cache = False
    # MoE aux-loss tensors blow VRAM at long context during LoRA SFT.
    if hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = False
    # needed so gradients reach LoRA params through a frozen/quantized base
    # when gradient checkpointing is on
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config(args, fmt))
    model.print_trainable_parameters()
    return model, tokenizer, False


def pull_hf_dataset(args) -> tuple[Path, Path]:
    """Pull a HF dataset to out/sft.jsonl (+ sft-eval.jsonl), running
    pull_dataset.py's cleaning pass (benchmark decontamination + teacher
    identity sanitize to --base) so a stale HF snapshot can't poison a run.
    Datasets that ship only a train split are split via --eval-frac."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pull_dataset import pull_split_clean

    print(f"Pulling dataset from HF: {args.hf_repo}...")
    return pull_split_clean(args.hf_repo, args.out, args.eval_frac,
                            model_name=args.base)


def _write_opencode_config(out_dir: Path, base_url: str, model_id: str) -> Path:
    """Write an opencode.json that points opencode at the in-process server, so
    the opencode-tasks section of run_evals.py routes to the in-training model.
    Returns the directory to run opencode from (its $PWD picks up the config)."""
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "local": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Local (in-training)",
                "options": {"baseURL": base_url},
                "models": {model_id: {"name": model_id}},
            }
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "opencode.json").write_text(json.dumps(cfg, indent=2))
    return out_dir


class IntermittentEvalCallback:
    """TrainerCallback that runs eval/run_evals.py against an in-process server
    backed by the live model, at training start, every N steps, and at the end,
    logging section pass rates to W&B. Imported lazily as a TrainerCallback
    subclass inside main() (so transformers stays an optional import here)."""

    def __init__(self, args, eval_dir: Path):
        self.cb_args = args
        self.eval_dir = eval_dir
        self.server = None
        self._last_step = -1
        self._epoch = 0

    # --- lifecycle hooks (names match transformers.TrainerCallback) ---
    def on_train_begin(self, args, state, control, **kw):
        if self.cb_args.eval_gate_at_start:
            self._run(state, kw, when="start")

    def on_step_end(self, args, state, control, **kw):
        n = self.cb_args.eval_gate_steps
        if n > 0 and state.global_step > 0 and state.global_step % n == 0:
            self._run(state, kw, when="step")

    def on_epoch_end(self, args, state, control, **kw):
        n = self.cb_args.eval_gate_epochs
        if n <= 0:
            return
        self._epoch += 1
        if self._epoch % n == 0:
            self._run(state, kw, when=f"epoch{self._epoch}")

    def on_train_end(self, args, state, control, **kw):
        self._run(state, kw, when="end")
        if self.server is not None:
            self.server.stop()

    # --- helpers ---
    def _ensure_server(self, model, tokenizer):
        if self.server is None:
            from eval_server import InProcessModelServer
            self.server = InProcessModelServer(
                model, tokenizer, port=self.cb_args.eval_gate_port,
                max_new_tokens=self.cb_args.eval_gate_max_new_tokens,
                chat_format=getattr(self.cb_args, "_fmt", "laguna"))
            url = self.server.start()
            print(f"[eval-gate] in-process server up at {url}")
        return self.server

    def _run(self, state, kw, when: str):
        step = state.global_step
        if step == self._last_step:
            return
        self._last_step = step
        model = kw.get("model")
        tokenizer = kw.get("processing_class") or kw.get("tokenizer")
        if model is None or tokenizer is None:
            print("[eval-gate] no model/tokenizer in callback; skipping")
            return

        was_training = model.training
        prev_cache = getattr(model.config, "use_cache", False)
        model.eval()
        model.config.use_cache = True
        # MoE aux-loss: when training enables output_router_logits, the model's
        # forward calls load_balancing_loss_func, which crashes during *cached*
        # generation (per-decode-step router logits mismatch attention_mask:
        # "tensor a (num_layers) must match tensor b (N)"). Force it off across
        # the model + text configs while the gate generates; restore after.
        cfgs = []
        seen = set()
        for c in (getattr(model, "config", None),
                  getattr(getattr(model, "config", None),
                          "get_text_config", lambda: None)()):
            if c is not None and id(c) not in seen and hasattr(
                    c, "output_router_logits"):
                cfgs.append((c, c.output_router_logits))
                seen.add(id(c))
                c.output_router_logits = False
        try:
            server = self._ensure_server(model, tokenizer)
            if not self.cb_args.eval_gate_skip_core:
                self._run_core(server, step, when)
            if self.cb_args.eval_gate_problem_pack:
                self._run_pack(server, step, when)
        except Exception as exc:  # noqa: BLE001 — never let eval kill training
            print(f"[eval-gate] FAILED at step {step}: {exc!r}")
        finally:
            model.config.use_cache = prev_cache
            for c, v in cfgs:
                c.output_router_logits = v
            if was_training:
                model.train()

    def _run_core(self, server, step: int, when: str):
        """run_evals.py: tool-validity + chat-sanity (+ optional opencode)."""
        import os
        import subprocess
        import sys
        run_evals = Path(__file__).resolve().parent.parent / "eval" / "run_evals.py"
        cmd = [sys.executable, str(run_evals), "--base-url", server.base_url]
        env = dict(os.environ)
        if self.cb_args.eval_gate_opencode:
            workdir = _write_opencode_config(
                self.cb_args.out / "opencode-eval", server.base_url,
                "local-code-model")
            cmd += ["--model", "local/local-code-model"]
            env["PATH"] = os.path.expanduser("~/.opencode/bin") + os.pathsep + env.get("PATH", "")
            cwd = str(workdir)
        else:
            cmd.append("--skip-opencode")
            cwd = None

        before = {p.name for p in self.eval_dir.glob("results-*.json")}
        print(f"[eval-gate] step {step} ({when}): {' '.join(cmd)}")
        subprocess.run(cmd, env=env, cwd=cwd, check=False)
        new = sorted(p for p in self.eval_dir.glob("results-*.json")
                     if p.name not in before)
        if not new:
            print("[eval-gate] no results file produced")
            return
        self._log(json.loads(new[-1].read_text()), step, when)

    def _run_pack(self, server, step: int, when: str):
        """agent-problem-pack against the in-training model (isolated opencode
        home so the gate never touches your real opencode.db)."""
        import os
        import subprocess
        import sys
        runner = Path(__file__).resolve().parent.parent / "eval" / "run_problem_pack.py"
        if not runner.exists():
            print("[eval-gate] run_problem_pack.py missing; skipping pack")
            return
        home = self.cb_args.out / "pack-opencode-home"
        cmd = [sys.executable, str(runner),
               "--base-url", server.base_url,
               "--served-model", "local-code-model",
               "--model", "local/local-code-model",
               "--subset", self.cb_args.eval_gate_pack_subset,
               "--out", str(self.eval_dir),
               "--opencode-home", str(home),
               "--timeout", str(self.cb_args.eval_gate_pack_timeout)]
        env = dict(os.environ)
        env["PATH"] = os.path.expanduser("~/.opencode/bin") + os.pathsep + env.get("PATH", "")
        before = {p.name for p in self.eval_dir.glob("pack-results-*.json")}
        print(f"[eval-gate] step {step} ({when}) pack: {' '.join(cmd)}")
        subprocess.run(cmd, env=env, check=False)
        new = sorted(p for p in self.eval_dir.glob("pack-results-*.json")
                     if p.name not in before)
        if not new:
            print("[eval-gate] no pack results produced")
            return
        self._log_pack(json.loads(new[-1].read_text()), step, when)

    def _log_pack(self, results: dict, step: int, when: str):
        try:
            import wandb
        except Exception:  # noqa: BLE001
            return
        if wandb.run is None:
            return
        pack = results.get("problem_pack") or {}
        total = pack.get("total", 0)
        metrics = {"eval_gate/pack_passed": pack.get("passed", 0),
                   "eval_gate/pack_total": total}
        if total:
            metrics["eval_gate/pack_pass_rate"] = pack["passed"] / total
        for diff, d in (pack.get("by_difficulty") or {}).items():
            if d.get("total"):
                metrics[f"eval_gate/pack_{diff}_rate"] = d["passed"] / d["total"]
        if pack.get("mean_steps") is not None:
            metrics["eval_gate/pack_mean_steps"] = pack["mean_steps"]
        if pack.get("mean_tokens") is not None:
            metrics["eval_gate/pack_mean_tokens"] = pack["mean_tokens"]
        wandb.log(metrics, step=step)
        rate = metrics.get("eval_gate/pack_pass_rate")
        print(f"[eval-gate] step {step} ({when}) pack -> "
              f"pass_rate={rate:.3f}" if rate is not None
              else f"[eval-gate] step {step} ({when}) pack -> logged")

    def _log(self, results: dict, step: int, when: str):
        try:
            import wandb
        except Exception:  # noqa: BLE001
            return
        if wandb.run is None:
            return
        metrics = {}
        for section in ("tool_validity", "chat_sanity", "opencode_tasks"):
            sec = results.get(section) or {}
            passed, total = sec.get("passed", 0), sec.get("total", 0)
            metrics[f"eval_gate/{section}_passed"] = passed
            metrics[f"eval_gate/{section}_total"] = total
            if total:
                metrics[f"eval_gate/{section}_rate"] = passed / total
        wandb.log(metrics, step=step)
        summary = "  ".join(f"{k.split('/')[-1]}={v}" for k, v in metrics.items()
                            if k.endswith("_rate"))
        print(f"[eval-gate] step {step} ({when}) -> {summary or 'logged'}")


def push_folder_to_hub(local_dir: Path, repo_id: str, private: bool) -> str | None:
    """Upload a local folder to an HF model repo (creating it if needed).
    Returns the repo URL, or None on failure (never fatal — the local
    checkpoint is already on disk)."""
    import os
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(repo_id, repo_type="model", private=private,
                        exist_ok=True)
        api.upload_folder(folder_path=str(local_dir), repo_id=repo_id,
                          repo_type="model")
        url = f"https://huggingface.co/{repo_id}"
        print(f"pushed {local_dir} -> {url}")
        return url
    except Exception as exc:  # noqa: BLE001
        print(f"push-to-hub FAILED for {repo_id}: {exc!r} "
              f"(local copy is at {local_dir})")
        return None


def main() -> None:
    args = parse_args()
    load_dotenv(Path(__file__).resolve().parent)
    args.out.mkdir(parents=True, exist_ok=True)

    # cuDNN's fused SDPA backend intermittently raises
    # "mha_graph.execute(...).is_good() == false" on Qwen3.6's full-attention
    # layers once the eval-gate has run long-context generation (memory
    # fragmentation) — it crashed training mid-step right after the step-20 gate.
    # Disable only the cuDNN SDPA backend so PyTorch falls back to the robust
    # flash / mem-efficient / math kernels. One process, so this covers both
    # training and the in-process eval server.
    import torch
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
        # Math SDPA materializes the full [seq, seq] attention matrix and
        # OOMs Qwen3.6's 16 full-attention layers at 64k (~17GB/layer in fp32).
        torch.backends.cuda.enable_math_sdp(False)
        print("sdpa: cuDNN + math backends disabled (flash/mem-efficient only)")
    except Exception as exc:  # noqa: BLE001
        print(f"sdpa: could not configure SDPA backends ({exc!r})")
    fmt = resolve_chat_format(args)
    quant = resolve_quant(args)
    fmt_cfg = CHAT_FORMATS[fmt]
    args._fmt = fmt  # consumed by the eval-gate server for output parsing
    print(f"base={args.base}  quant={quant}  chat-format={fmt}  "
          f"max-seq-len={args.max_seq_len}")
    if quant == "fp8" and args.include_mlp:
        print("note: --include-mlp ignored under fp8 (expert/MLP linears are "
              "compressed; only the bf16 attention projections are adapted)")
        args.include_mlp = False

    if args.hf_repo:
        args.data, args.eval_data = pull_hf_dataset(args)
    if args.data is None:
        raise SystemExit("provide --data or --hf-repo")

    # Qwen3.6 (a hybrid MoE / multimodal checkpoint) loads most reliably through
    # Unsloth — it handles the text-only path, MoE kernels, and the DeltaNet
    # LoRA targets. We try Unsloth first for both bf16 (`none`) and 4-bit Qwen,
    # falling back to transformers. Other bases keep the original routing.
    prefer_unsloth = ((quant == "4bit" or (fmt == "qwen" and quant == "none"))
                      and not args.no_unsloth)
    if prefer_unsloth:
        load_in_4bit = quant == "4bit"
        try:
            model, tokenizer, is_unsloth = try_unsloth(args, fmt, load_in_4bit)
            print(f"== using Unsloth ({'4-bit' if load_in_4bit else 'bf16'}) ==")
        except Exception as exc:  # noqa: BLE001
            print(f"== Unsloth unavailable ({exc!r}); "
                  f"transformers fallback ==")
            if load_in_4bit:
                model, tokenizer, is_unsloth = load_4bit(args, fmt)
            else:
                model, tokenizer, is_unsloth = load_lora(args, fmt)
    else:
        if quant == "4bit":
            model, tokenizer, is_unsloth = load_4bit(args, fmt)
            print("== 4bit LoRA ==")
        else:
            model, tokenizer, is_unsloth = load_lora(args, fmt)
            print(f"== {quant} LoRA (no bitsandbytes) ==")
    tokenizer = normalize_text_tokenizer(tokenizer)
    # Unsloth sets up its own gradient checkpointing; the native-precision LoRA
    # and transformers QLoRA paths need the trainer to enable it.
    grad_ckpt = not is_unsloth

    liger_on = False
    if not args.no_liger:
        liger_on = (apply_liger_laguna(model)
                    or apply_liger_qwen3_5_moe(model)
                    or apply_liger_qwen3_5(model))
        print("liger: fused linear cross-entropy enabled (frees logits VRAM)"
              if liger_on else "liger: not applied (unknown arch); standard loss")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    resp_ids = tokenizer(fmt_cfg["response_marker"],
                         add_special_tokens=False).input_ids
    end_ids = tokenizer(fmt_cfg["end_marker"],
                        add_special_tokens=False).input_ids

    preflight_template(tokenizer, fmt_cfg, resp_ids, end_ids)
    allowed_models = parse_allowed_models(args.allow_models)
    print("teacher filter: "
          + (f"only models matching {list(allowed_models)}"
             if allowed_models else "off (--allow-models all)"))
    dataset = load_samples(args.data, tokenizer, fmt_cfg, args.max_seq_len,
                           resp_ids, end_ids, allowed_models)
    if args.max_samples and len(dataset) > args.max_samples:
        dataset = dataset.select(range(args.max_samples))
        print(f"smoke: capped train set to {len(dataset)} samples "
              f"(--max-samples {args.max_samples})")
    eval_dataset = None
    if args.eval_data and Path(args.eval_data).is_file():
        eval_dataset = load_samples(args.eval_data, tokenizer, fmt_cfg,
                                    args.max_seq_len, resp_ids, end_ids,
                                    allowed_models)

    run_name = setup_wandb(args, fmt, len(dataset),
                           len(eval_dataset) if eval_dataset else 0)
    print(f"wandb: {'-> ' + run_name if run_name else 'off'}")

    from trl import SFTConfig, SFTTrainer

    config = SFTConfig(
        output_dir=str(args.out / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        # step-based checkpointing (aligned to the gate cadence) so a crash
        # mid-run is resumable instead of losing the whole epoch
        save_strategy="steps",
        save_steps=(args.eval_gate_steps if args.eval_gate
                    and args.eval_gate_steps > 0 else 20),
        save_total_limit=2,
        bf16=True,
        max_length=args.max_seq_len,
        # rows are already tokenized (input_ids + masked labels); don't let TRL
        # re-tokenize or prune our columns before the collator sees them
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if grad_ckpt else None,
        # our manual Laguna patch provides the fused loss; this flag makes TRL
        # take the liger code paths (skip_logits, token-accuracy from output)
        use_liger_kernel=liger_on,
        report_to="wandb" if run_name else "none",
        run_name=run_name,
        seed=0,
        **({"eval_strategy": "steps",
            "eval_steps": args.eval_steps,
            "per_device_eval_batch_size": args.batch_size}
           if eval_dataset is not None and not args.no_eval else {}),
    )
    callbacks = []
    if args.eval_gate:
        from transformers import TrainerCallback

        class _EvalGateCallback(IntermittentEvalCallback, TrainerCallback):
            pass

        eval_dir = Path(__file__).resolve().parent.parent / "eval"
        callbacks.append(_EvalGateCallback(args, eval_dir))
        cadence = []
        if args.eval_gate_steps > 0:
            cadence.append(f"every {args.eval_gate_steps} steps")
        if args.eval_gate_epochs > 0:
            cadence.append(f"every {args.eval_gate_epochs} epoch(s)")
        if args.eval_gate_at_start:
            cadence.append("at start")
        cadence.append("at end")
        extras = [f"opencode={'on' if args.eval_gate_opencode else 'off'}"]
        if args.eval_gate_problem_pack:
            extras.append(f"problem-pack={args.eval_gate_pack_subset}")
        if args.eval_gate_skip_core:
            extras.append("core=off")
        print(f"eval-gate: on ({', '.join(cadence)}, {', '.join(extras)})")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=config,
        data_collator=MaskedCollator(tokenizer),
        callbacks=callbacks or None,
    )
    print("loss masked to assistant turns (assistant-token mask)")

    trainer.train()

    adapter_dir = args.out / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"adapter saved -> {adapter_dir}")

    if args.push_to_hub:
        push_folder_to_hub(adapter_dir, args.push_to_hub, args.hub_private)

    if args.no_merge:
        return

    if quant == "fp8":
        print("skip merge: a LoRA adapter trained on the FP8 "
              "(compressed-tensors) base can't be merge_and_unload'd here "
              "(quantized weights + Hadamard transforms). Serve the adapter on "
              "top of the FP8 base with vLLM (`--enable-lora`), or merge "
              "offline with llmcompressor. Adapter is at "
              f"{args.out / 'adapter'}.")
        return

    merged_dir = args.out / "merged"
    if is_unsloth:
        model.save_pretrained_merged(
            str(merged_dir), tokenizer, save_method="merged_16bit",
        )
    else:
        # reload base in bf16 and merge the adapter into it
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        print("merging adapter into bf16 base (needs ~70GB RAM/offload)...")
        base = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.bfloat16, device_map="cpu",
        )
        merged = PeftModel.from_pretrained(base, str(adapter_dir))
        merged = merged.merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
    print(f"merged checkpoint -> {merged_dir}")

    if args.push_to_hub and args.push_merged:
        push_folder_to_hub(merged_dir, args.push_to_hub + "-merged",
                           args.hub_private)

    if run_name:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
