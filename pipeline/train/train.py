#!/usr/bin/env python3
"""QLoRA fine-tune a local code model on collected traces. Runs on a CUDA GPU.

Defaults to `poolside/Laguna-XS.2` (33B-total / 3B-active MoE). Tries Unsloth
first (fastest, lowest VRAM); falls back to plain transformers + PEFT + TRL if
Unsloth can't load the architecture (the likely path for Laguna today).

Loss is masked to assistant turns only. Laguna's chat template marks assistant
spans with Jinja `{% generation %}` blocks, so we get an exact assistant-token
mask straight from `apply_chat_template(..., return_assistant_tokens_mask=True)`
— no brittle string matching. For chat formats whose template lacks generation
blocks (e.g. Cohere), we fall back to masking between configured turn markers.

Input:
    --data sft.jsonl (one {"messages": [...], "tools": [...]|null, "source": ...}
        per line) or --hf-repo to pull from HF. A single-split HF dataset (e.g.
        nkasmanoff/opencode-sft) is split into train/eval via --eval-frac.
Output: <out>/adapter/  (LoRA weights)
        <out>/merged/   (merged bf16 checkpoint, ready for GGUF conversion)

Usage (on the pod):
    # Pull + split a single-split HF dataset and train Laguna:
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

LAGUNA_BASE = "poolside/Laguna-XS.2"
# FP8 (compressed-tensors) build: the MoE experts are FP8-compressed while the
# attention projections stay bf16, so the model fits on one 80GB H100 and LoRA
# wraps the un-quantized attention linears. This is the default because plain
# bitsandbytes 4-bit can't quantize Laguna's fused `LagunaExperts` module (94%
# of params), so QLoRA loads the model near full bf16 size and OOMs.
LAGUNA_FP8_BASE = "poolside/Laguna-XS.2-FP8"

# Per-chat-format settings. `apply_chat_template(return_assistant_tokens_mask=
# True)` is preferred for masking; the marker fields are only used as a fallback
# when a template has no `{% generation %}` blocks (so no assistant mask).
CHAT_FORMATS = {
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
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--base", default=LAGUNA_FP8_BASE)
    p.add_argument("--quant", choices=["auto", "fp8", "4bit", "none"],
                   default="auto",
                   help="base-model quantization. auto: 'fp8' (native "
                        "compressed-tensors) for *-FP8 repos, else '4bit' "
                        "(bitsandbytes QLoRA). 'none' loads bf16 (needs lots of "
                        "VRAM / multiple GPUs).")
    p.add_argument("--chat-format", choices=["auto", "laguna", "cohere"],
                   default="auto",
                   help="chat template family (auto-detected from --base)")
    # Laguna sequences are long (opencode-sft: mean ~30k, p95 ~79k tokens).
    # 65536 keeps roughly the shorter ~90% of samples; lower it (e.g. 32768)
    # if a single 80GB H100 OOMs at this context length.
    p.add_argument("--max-seq-len", type=int, default=65536)
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
    p.add_argument("--no-merge", action="store_true",
                   help="skip writing the merged bf16 checkpoint")
    p.add_argument("--wandb-project", default="laguna-sft",
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


def resolve_quant(args) -> str:
    if args.quant != "auto":
        return args.quant
    base = (args.base or "").lower()
    if "fp8" in base or "compressed" in base:
        return "fp8"
    return "4bit"


def resolve_chat_format(args) -> str:
    if args.chat_format != "auto":
        return args.chat_format
    base = (args.base or "").lower()
    if "laguna" in base:
        return "laguna"
    if any(k in base for k in ("cohere", "command", "north-mini", "north_mini")):
        return "cohere"
    # default base is Laguna; assume laguna for unknown ids
    return "laguna"


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


def lora_targets(include_mlp: bool) -> list[str]:
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if include_mlp:
        # expert FFN projections; router/gate module names are excluded on
        # purpose — adapting the router destabilizes MoE training
        targets += ["gate_proj", "up_proj", "down_proj"]
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


def load_samples(path: Path, tokenizer, fmt_cfg: dict, max_seq_len: int,
                 resp_ids: list[int], end_ids: list[int]) -> "Dataset":
    """Tokenize each conversation and attach an assistant-only loss mask."""
    from datasets import Dataset

    rows = []
    skipped_render = skipped_long = skipped_nomask = 0
    first_error: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        try:
            ids, mask = tokenize_record(rec, tokenizer, fmt_cfg,
                                        resp_ids, end_ids)
        except Exception as exc:  # noqa: BLE001 — bad sample, skip
            skipped_render += 1
            if first_error is None:
                first_error = repr(exc)
            continue
        if len(ids) > max_seq_len:
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
           f"{max_seq_len} tok, {skipped_nomask} no-assistant)")
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


def try_unsloth(args):
    """Return (model, tokenizer, is_unsloth) or raise."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=lora_targets(args.include_mlp),
        use_gradient_checkpointing="unsloth",
        random_state=0,
    )
    return model, tokenizer, True


def lora_config(args):
    from peft import LoraConfig
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets(args.include_mlp),
    )


def load_4bit(args):
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
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config(args))
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


def load_lora(args):
    """LoRA on a model loaded in its native precision (no bitsandbytes).

    Used for the FP8 compressed-tensors build (experts stay FP8-compressed, the
    attention projections we adapt stay bf16) and for plain bf16 (--quant none).
    We adapt the attention linears only; gradient checkpointing keeps activation
    memory in check at long context."""
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    # transformers reads the FP8 build's own quantization_config
    # (compressed-tensors); a plain bf16 repo just loads in bf16.
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    # needed so gradients reach LoRA params through a frozen/quantized base
    # when gradient checkpointing is on
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config(args))
    model.print_trainable_parameters()
    return model, tokenizer, False


def pull_hf_dataset(args) -> tuple[Path, Path]:
    """Pull a HF dataset to out/sft.jsonl (+ sft-eval.jsonl). Datasets that
    ship only a train split are deterministically split via --eval-frac."""
    from datasets import load_dataset

    print(f"Pulling dataset from HF: {args.hf_repo}...")
    ds = load_dataset(args.hf_repo)
    eval_key = next((k for k in ("eval", "test", "validation") if k in ds), None)
    if eval_key:
        train_split, eval_split = ds["train"], ds[eval_key]
        print(f"  using existing splits: train + {eval_key}")
    else:
        parts = ds["train"].train_test_split(test_size=args.eval_frac, seed=0)
        train_split, eval_split = parts["train"], parts["test"]
        print(f"  split train into {len(train_split)}/{len(eval_split)} "
              f"(eval_frac={args.eval_frac})")

    data_path = args.out / "sft.jsonl"
    eval_path = args.out / "sft-eval.jsonl"
    for split, out_path in ((train_split, data_path), (eval_split, eval_path)):
        with out_path.open("w") as f:
            for row in split:
                row = dict(row)
                # some repos store tools as a JSON string; normalize to a list
                if isinstance(row.get("tools"), str):
                    try:
                        row["tools"] = json.loads(row["tools"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                f.write(json.dumps(row) + "\n")
        print(f"  wrote {len(split)} samples -> {out_path}")
    return data_path, eval_path


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
                max_new_tokens=self.cb_args.eval_gate_max_new_tokens)
            url = self.server.start()
            print(f"[eval-gate] in-process server up at {url}")
        return self.server

    def _run(self, state, kw, when: str):
        import os
        import subprocess
        import sys
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
        try:
            server = self._ensure_server(model, tokenizer)
            run_evals = Path(__file__).resolve().parent.parent / "eval" / "run_evals.py"
            cmd = [sys.executable, str(run_evals),
                   "--base-url", server.base_url]
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
            results = json.loads(new[-1].read_text())
            self._log(results, step, when)
        except Exception as exc:  # noqa: BLE001 — never let eval kill training
            print(f"[eval-gate] FAILED at step {step}: {exc!r}")
        finally:
            model.config.use_cache = prev_cache
            if was_training:
                model.train()

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
    fmt = resolve_chat_format(args)
    quant = resolve_quant(args)
    fmt_cfg = CHAT_FORMATS[fmt]
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

    if quant == "4bit":
        try:
            model, tokenizer, is_unsloth = try_unsloth(args)
            print("== using Unsloth (4-bit) ==")
        except Exception as exc:  # noqa: BLE001
            print(f"== Unsloth unavailable ({exc!r}); "
                  f"transformers + bitsandbytes 4-bit ==")
            model, tokenizer, is_unsloth = load_4bit(args)
    else:
        model, tokenizer, is_unsloth = load_lora(args)
        print(f"== {quant} LoRA (no bitsandbytes) ==")
    # gradient checkpointing is set up inside the 4-bit/Unsloth paths; the
    # native-precision LoRA path needs the trainer to enable it
    grad_ckpt = quant in ("fp8", "none")

    liger_on = False
    if not args.no_liger and not is_unsloth:
        liger_on = apply_liger_laguna(model)
        print("liger: fused linear cross-entropy enabled (frees logits VRAM)"
              if liger_on else "liger: not applied (non-Laguna arch); standard loss")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    resp_ids = tokenizer(fmt_cfg["response_marker"],
                         add_special_tokens=False).input_ids
    end_ids = tokenizer(fmt_cfg["end_marker"],
                        add_special_tokens=False).input_ids

    preflight_template(tokenizer, fmt_cfg, resp_ids, end_ids)
    dataset = load_samples(args.data, tokenizer, fmt_cfg, args.max_seq_len,
                           resp_ids, end_ids)
    if args.max_samples and len(dataset) > args.max_samples:
        dataset = dataset.select(range(args.max_samples))
        print(f"smoke: capped train set to {len(dataset)} samples "
              f"(--max-samples {args.max_samples})")
    eval_dataset = None
    if args.eval_data and Path(args.eval_data).is_file():
        eval_dataset = load_samples(args.eval_data, tokenizer, fmt_cfg,
                                    args.max_seq_len, resp_ids, end_ids)

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
        save_strategy="epoch",
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
           if eval_dataset is not None else {}),
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
        print(f"eval-gate: on ({', '.join(cadence)}, "
              f"opencode={'on' if args.eval_gate_opencode else 'off'})")

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
