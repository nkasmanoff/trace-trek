"""Merge the trained LoRA adapter into a bf16 copy of the base model and push
the merged checkpoint to the Hub.

We merge *manually* (W += (alpha/r) * B @ A) instead of using
PeftModel.merge_and_unload, because peft 0.19.1 + transformers 5.12.1 hit a
`WeightConverter.__init__() got an unexpected keyword 'distributed_operation'`
bug in peft's transformers-native adapter conversion. The math is plain LoRA
(no DoRA/rsLoRA/dropout-scaling), so a direct delta-add is exact."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train import load_dotenv, push_folder_to_hub  # noqa: E402

ADAPTER_DIR = HERE.parent / "outputs" / "adapter"
MERGED_DIR = HERE.parent / "outputs" / "merged"
REPO = "nkasmanoff/qwen3.6-opencode-lora-merged"


def main() -> None:
    load_dotenv(HERE)
    cfg = json.loads((ADAPTER_DIR / "adapter_config.json").read_text())
    base_name = cfg["base_model_name_or_path"]
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scaling = alpha / r
    assert not cfg.get("use_dora") and not cfg.get("use_rslora"), "plain LoRA only"
    print(f"base={base_name}  r={r} alpha={alpha} scaling={scaling}")

    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("loading bf16 base on CPU (needs ~70GB RAM)...")
    base = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True,
    )
    sd = load_file(str(ADAPTER_DIR / "adapter_model.safetensors"))

    # pair up lora_A / lora_B by module path
    modules = {}
    for k in sd:
        if ".lora_A.weight" in k:
            modules.setdefault(k[: k.index(".lora_A.weight")], {})["A"] = k
        elif ".lora_B.weight" in k:
            modules.setdefault(k[: k.index(".lora_B.weight")], {})["B"] = k
    print(f"merging {len(modules)} LoRA modules...")

    merged_n = 0
    for prefix, ab in sorted(modules.items()):
        assert "A" in ab and "B" in ab, f"unpaired adapter at {prefix}"
        # base_model.model.<path>  ->  <path> submodule on the HF model
        mod_path = prefix.replace("base_model.model.", "", 1)
        sub = base.get_submodule(mod_path)
        A = sd[ab["A"]].to(torch.float32)          # [r, in]
        B = sd[ab["B"]].to(torch.float32)          # [out, r]
        delta = (B @ A) * scaling                    # [out, in]
        W = sub.weight
        assert W.shape == delta.shape, f"shape {W.shape} vs {delta.shape} at {mod_path}"
        W.data = (W.data.to(torch.float32) + delta).to(W.dtype)
        merged_n += 1
    print(f"applied {merged_n} deltas")

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    base.save_pretrained(str(MERGED_DIR), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(str(ADAPTER_DIR), trust_remote_code=True)
    tok.save_pretrained(str(MERGED_DIR))
    print(f"merged checkpoint -> {MERGED_DIR}")

    url = push_folder_to_hub(MERGED_DIR, REPO, private=False)
    print(f"DONE merged_push_url={url}")


if __name__ == "__main__":
    main()
