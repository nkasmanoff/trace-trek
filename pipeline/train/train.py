#!/usr/bin/env python3
"""QLoRA fine-tune a local code model on collected traces. Runs on a CUDA GPU.

Tries Unsloth first (fastest, lowest VRAM); falls back to plain
transformers + PEFT + TRL if Unsloth can't load the brand-new
`cohere2_moe` architecture.

Input:
    --data sft.jsonl from build_dataset.py or --hf-repo to pull from HF
    ({"messages": [...], "tools": [...]|null, "source": ...} per line)
Output: <out>/adapter/  (LoRA weights)
        <out>/merged/   (merged bf16 checkpoint, ready for GGUF conversion)

Usage (on the pod):
    python train.py --data sft.jsonl --out outputs \
        [--base CohereLabs/local-code-model] [--max-seq-len 65536] \
        [--epochs 2] [--lr 1e-4] [--lora-r 16] [--include-mlp]

    # Or pull from HuggingFace:
    python train.py --hf-repo nkasmanoff/local-code-sft-json --out outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESPONSE_MARKER = "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
INSTRUCTION_MARKER = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
END_OF_TURN_TOKEN = "<|END_OF_TURN_TOKEN|>"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=None,
                   help="local sft jsonl file (mutually exclusive with --hf-repo)")
    p.add_argument("--hf-repo", default=None,
                   help="HuggingFace dataset repo to pull from, e.g. "
                        "nkasmanoff/local-code-sft-json")
    p.add_argument("--eval-data", type=Path, default=None,
                   help="held-out sft jsonl; if set, eval loss is computed "
                        "every --eval-steps")
    p.add_argument("--eval-steps", type=int, default=20)
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--base", default="CohereLabs/local-code-model")
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
    p.add_argument("--no-merge", action="store_true",
                   help="skip writing the merged bf16 checkpoint")
    p.add_argument("--wandb-project", default="local-code-sft",
                   help="W&B project; logging activates when WANDB_API_KEY "
                        "is set (use --no-wandb to force off)")
    p.add_argument("--run-name", default=None,
                   help="W&B run name (default: auto from data + params)")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def setup_wandb(args, n_train: int, n_eval: int) -> str | None:
    """Init W&B if a key is available. Returns run name or None."""
    import os
    if args.no_wandb or not os.environ.get("WANDB_API_KEY"):
        return None
    import time
    import wandb
    run_name = args.run_name or (
        f"sft-{n_train}samp-r{args.lora_r}-lr{args.lr:g}"
        f"-{time.strftime('%m%d-%H%M')}")
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "base_model": args.base,
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


def render_messages(rec: dict, tokenizer) -> str:
    """Render one record's conversation with the model's chat template."""
    messages = []
    for m in rec["messages"]:
        m = dict(m)
        # the official template renders assistant thinking via msg.thinking
        if m.get("reasoning_content"):
            m["thinking"] = m.pop("reasoning_content")
        messages.append(m)
    return tokenizer.apply_chat_template(
        messages,
        tools=rec.get("tools") or None,
        tokenize=False,
        add_generation_prompt=False,
    )


def preflight_template(tokenizer) -> None:
    """Fail loudly (before loading the model fully matters) if the chat
    template can't render reasoning + tool calls. A bare try/except in the
    data loader would otherwise silently drop every agentic sample and train
    on nothing useful."""
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
        text = render_messages(rec, tokenizer)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"FATAL: chat template failed to render a tool-call + reasoning "
            f"sample: {exc!r}\nThe collected traces are agentic (tools + "
            f"reasoning_content); a template that can't render them would "
            f"drop every sample. Check the tokenizer for {tokenizer.name_or_path}."
        )
    missing = [name for name, needle in (
        ("response marker", RESPONSE_MARKER),
        ("reasoning/thinking", "PREFLIGHT_THINK"),
        ("tool name", "calc"),
    ) if needle not in text]
    if missing:
        raise SystemExit(
            f"FATAL: chat template rendered but is missing {missing}. "
            f"Reasoning and/or tool calls are being dropped on the floor — "
            f"training would not teach the behavior we collected.\n"
            f"--- rendered sample (truncated) ---\n{text[:2000]}"
        )
    print("preflight: chat template renders reasoning + tool calls OK")


def load_samples(path: Path, tokenizer, max_seq_len: int) -> "Dataset":
    """Render each conversation with the model's chat template into `text`."""
    from datasets import Dataset

    rows = []
    skipped = 0
    first_error: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        try:
            text = render_messages(rec, tokenizer)
        except Exception as exc:  # noqa: BLE001 — bad sample, skip
            skipped += 1
            if first_error is None:
                first_error = repr(exc)
            continue
        n_tokens = len(tokenizer(text, add_special_tokens=False).input_ids)
        if n_tokens > max_seq_len:
            skipped += 1
            continue
        rows.append({"text": text, "source": rec.get("source", "?")})
    msg = f"dataset: {len(rows)} samples ({skipped} skipped) from {path}"
    if first_error:
        msg += f"  [first render error: {first_error}]"
    print(msg)
    if not rows:
        raise SystemExit(f"FATAL: no usable samples from {path}")
    return Dataset.from_list(rows)


class AssistantOnlyCollator:
    """Pad a batch and mask loss to assistant turns only.

    The Unsloth path uses ``train_on_responses_only``; the plain
    transformers/PEFT fallback (the likely path for a brand-new arch) has no
    equivalent, so without this it would compute loss over system/user/tool
    tokens too — i.e. teach the model to *generate* tool outputs it should
    only ever read. We tokenize each rendered ``text`` and unmask only the
    spans between a response marker and the following end-of-turn token.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.resp_ids = tokenizer(
            RESPONSE_MARKER, add_special_tokens=False).input_ids
        self.end_ids = tokenizer(
            END_OF_TURN_TOKEN, add_special_tokens=False).input_ids
        if not self.resp_ids:
            raise SystemExit(
                "FATAL: response marker tokenizes to nothing; cannot mask "
                "assistant turns in the fallback path.")

    @staticmethod
    def _find(haystack: list[int], needle: list[int], start: int) -> int:
        if not needle:
            return -1
        for i in range(start, len(haystack) - len(needle) + 1):
            if haystack[i:i + len(needle)] == needle:
                return i
        return -1

    def __call__(self, examples: list[dict]) -> dict:
        import torch

        # SFTTrainer may hand us either raw {"text": ...} rows or rows it has
        # already tokenized into {"input_ids": [...]}, depending on the TRL
        # version. Handle both so masking happens regardless.
        if examples and "input_ids" in examples[0]:
            seqs = [list(ex["input_ids"]) for ex in examples]
            pad_id = (self.tokenizer.pad_token_id
                      if self.tokenizer.pad_token_id is not None
                      else self.tokenizer.eos_token_id)
            width = max(len(s) for s in seqs)
            right = self.tokenizer.padding_side != "left"
            input_ids, attn = [], []
            for s in seqs:
                pad = [pad_id] * (width - len(s))
                input_ids.append(s + pad if right else pad + s)
                a = [1] * len(s)
                amask = a + [0] * len(pad) if right else [0] * len(pad) + a
                attn.append(amask)
            batch = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
            }
        else:
            texts = [ex["text"] for ex in examples]
            batch = self.tokenizer(
                texts, add_special_tokens=False, padding=True,
                return_tensors="pt",
            )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for row, ids in enumerate(batch["input_ids"].tolist()):
            mask = [False] * len(ids)
            pos = 0
            while True:
                start = self._find(ids, self.resp_ids, pos)
                if start == -1:
                    break
                span_start = start + len(self.resp_ids)
                end = self._find(ids, self.end_ids, span_start) \
                    if self.end_ids else -1
                # include the end-of-turn token so the model learns to stop;
                # if no end token (truncated final turn), train to sequence end
                span_end = (end + len(self.end_ids)) if end != -1 else len(ids)
                for i in range(span_start, span_end):
                    mask[i] = True
                pos = span_end
            for i, keep in enumerate(mask):
                if not keep:
                    labels[row, i] = -100
        batch["labels"] = labels
        return batch


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


def fallback_hf(args):
    """Plain transformers + bitsandbytes + PEFT path."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets(args.include_mlp),
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tokenizer, False


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.hf_repo:
        from pathlib import Path
        from datasets import load_dataset
        import json

        print(f"Pulling dataset from HF: {args.hf_repo}...")
        ds = load_dataset(args.hf_repo,
                          data_files={"train": "train.jsonl", "eval": "eval.jsonl"})
        data_path = args.out / "sft.jsonl"
        eval_path = args.out / "sft-eval.jsonl"
        for split, out_path in (("train", data_path), ("eval", eval_path)):
            with out_path.open("w") as f:
                for row in ds[split]:
                    if row.get("tools"):
                        row["tools"] = json.loads(row["tools"])
                    f.write(json.dumps(row) + "\n")
            print(f"  wrote {len(ds[split])} samples -> {out_path}")
        args.data = data_path
        args.eval_data = eval_path

    try:
        model, tokenizer, is_unsloth = try_unsloth(args)
        print("== using Unsloth ==")
    except Exception as exc:  # noqa: BLE001
        print(f"== Unsloth unavailable for this arch ({exc!r}); "
              f"falling back to transformers+PEFT ==")
        model, tokenizer, is_unsloth = fallback_hf(args)

    preflight_template(tokenizer)
    dataset = load_samples(args.data, tokenizer, args.max_seq_len)
    eval_dataset = None
    if args.eval_data:
        eval_dataset = load_samples(args.eval_data, tokenizer,
                                    args.max_seq_len)

    run_name = setup_wandb(args, len(dataset),
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
        dataset_text_field="text",
        report_to="wandb" if run_name else "none",
        run_name=run_name,
        seed=0,
        **({"eval_strategy": "steps",
            "eval_steps": args.eval_steps,
            "per_device_eval_batch_size": args.batch_size}
           if eval_dataset is not None else {}),
    )
    trainer_kwargs = dict(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=config,
    )
    # In the fallback path, mask loss to assistant turns ourselves (Unsloth
    # gets this from train_on_responses_only below). Without a collator SFT
    # would train on tool outputs / user / system tokens too.
    if not is_unsloth:
        trainer_kwargs["data_collator"] = AssistantOnlyCollator(tokenizer)
        print("loss masked to assistant turns (fallback collator)")
    trainer = SFTTrainer(**trainer_kwargs)

    if is_unsloth:
        # only compute loss on assistant turns
        try:
            from unsloth.chat_templates import train_on_responses_only
            trainer = train_on_responses_only(
                trainer,
                instruction_part=INSTRUCTION_MARKER,
                response_part=RESPONSE_MARKER,
            )
            print("loss masked to assistant turns")
        except Exception as exc:  # noqa: BLE001
            print(f"train_on_responses_only unavailable ({exc!r}); "
                  f"training on full sequences")

    trainer.train()

    adapter_dir = args.out / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"adapter saved -> {adapter_dir}")

    if args.no_merge:
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

    if run_name:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
