"""Produce a *multimodal* merged Qwen3.6 checkpoint and push it to the Hub.

Qwen3.6 checkpoints (dense 27B and MoE 35B-A3B) are native multimodal
(image-text-to-text) models: Qwen3_5ForConditionalGeneration (qwen3_5) or
Qwen3_5MoeForConditionalGeneration (qwen3_5_moe), with a vision tower plus the
text tower nested under model.language_model.*.

The train.py merge path loads the base via AutoModelForCausalLM, which
materializes only the text backbone (Qwen3_5ForCausalLM / qwen3_5_text). That
drops the vision tower and writes a text-only config, so Modal's base-model
validator rejects the published repo.

Three modes:

  graft   (default) -- load the full multimodal base, then overwrite its text
          tower with the already-merged text weights from an existing merged
          repo/dir. Needs no LoRA adapter; the vision tower is inherited from
          base untouched. This is the cheapest fix when you already have a
          text-only merged checkpoint on the Hub.

  remerge -- re-apply the LoRA adapter (W += (alpha/r) * B @ A) onto the full
          multimodal base. Needs outputs/adapter/. We merge manually instead of
          PeftModel.merge_and_unload because peft 0.19.1 + transformers 5.12.1
          hit a `WeightConverter.__init__() ... 'distributed_operation'` bug.
          The math is plain LoRA (no DoRA/rsLoRA), so a direct delta-add is exact.

  --dry-run -- only verify an existing merged dir (architecture + vision tower)
          without loading the base or pushing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train import load_dotenv, push_folder_to_hub  # noqa: E402

DEFAULT_OUT = HERE / "outputs"
ADAPTER_DIR = DEFAULT_OUT / "adapter"
MERGED_DIR = DEFAULT_OUT / "merged"
REPO = "nkasmanoff/qwen3.6-opencode-lora-merged"
DEFAULT_BASE = "Qwen/Qwen3.6-35B-A3B"
# the existing text-only merged checkpoint to graft from
DEFAULT_SOURCE = REPO


def load_full_base(base_name: str):
    """Load the full multimodal base on CPU in bf16."""
    import torch
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(base_name, trust_remote_code=True)
    mtype = getattr(cfg, "model_type", "")
    print(f"loading bf16 base on CPU (full multimodal, ~80GB RAM): "
          f"{base_name} ({mtype})")
    if "moe" in mtype:
        from transformers import Qwen3_5MoeForConditionalGeneration
        cls = Qwen3_5MoeForConditionalGeneration
    else:
        from transformers import Qwen3_5ForConditionalGeneration
        cls = Qwen3_5ForConditionalGeneration
    try:
        return cls.from_pretrained(
            base_name, torch_dtype=torch.bfloat16, device_map="cpu",
            trust_remote_code=True,
        )
    except ImportError:
        # older transformers without the explicit symbol: AutoModel resolves the
        # full multimodal class for an image-text-to-text config.
        from transformers import AutoModel
        return AutoModel.from_pretrained(
            base_name, torch_dtype=torch.bfloat16, device_map="cpu",
            trust_remote_code=True,
        )


def save_and_verify(base, base_name: str, tok_src: Path | str,
                    merged_dir: Path) -> None:
    """save_pretrained + carry the multimodal processor, then verify."""
    from transformers import AutoTokenizer

    merged_dir.mkdir(parents=True, exist_ok=True)
    base.save_pretrained(str(merged_dir), safe_serialization=True)
    # carry the full multimodal preprocessing config + chat template, not just
    # the bare tokenizer, so the saved repo matches the base for the validator.
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(
            base_name, trust_remote_code=True).save_pretrained(str(merged_dir))
    except Exception as exc:
        print(f"AutoProcessor unavailable ({exc!r}); tokenizer only")
    AutoTokenizer.from_pretrained(
        str(tok_src), trust_remote_code=True).save_pretrained(str(merged_dir))
    print(f"merged checkpoint -> {merged_dir}")
    verify_dir(merged_dir)


def verify_dir(d: Path) -> None:
    """Assert a merged dir advertises the multimodal arch and kept vision."""
    cfg_out = json.loads((d / "config.json").read_text())
    arch = cfg_out.get("architectures")
    mtype = cfg_out.get("model_type")
    print(f"saved architectures={arch} model_type={mtype}")
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        keys = json.loads(idx.read_text()).get("weight_map", {})
        has_vision = any(".visual." in k or "vision" in k for k in keys)
        print(f"vision tower present in index: {has_vision} "
              f"({len(keys)} weights total)")
        assert has_vision, "vision weights missing -- wrong base class loaded"
    assert arch and "ForConditionalGeneration" in arch[0], (
        f"architecture {arch} is not the multimodal class")
    print("verify OK: checkpoint matches the multimodal base")


def graft(base_name: str, source: str, merged_dir: Path) -> None:
    """Overwrite the multimodal base's text tower with the already-merged
    text weights from `source` (a Hub repo id or local dir)."""
    import torch
    from safetensors.torch import safe_open
    from huggingface_hub import snapshot_download

    src_dir = source if Path(source).is_dir() else snapshot_download(
        repo_id=source, allow_patterns=["*.safetensors", "*.json", "*.jinja",
                                        "tokenizer*"])
    src_dir = Path(src_dir)
    print(f"grafting merged text weights from: {src_dir}")

    base = load_full_base(base_name)
    base_keys = dict(base.state_dict())

    shards = sorted(src_dir.glob("*.safetensors"))
    assert shards, f"no safetensors in {src_dir}"
    applied = skipped = 0
    missing: list[str] = []
    with torch.no_grad():
        for shard in shards:
            with safe_open(str(shard), framework="pt") as f:
                for k in f.keys():
                    if k not in base_keys:
                        # text-only-view key (model.layers.*) -> nested layout
                        alt = k.replace("model.", "model.language_model.", 1)
                        tgt = alt if alt in base_keys else None
                    else:
                        tgt = k
                    if tgt is None:
                        missing.append(k)
                        skipped += 1
                        continue
                    dst = base_keys[tgt]
                    src = f.get_tensor(k)
                    assert dst.shape == src.shape, (
                        f"shape {tuple(dst.shape)} vs {tuple(src.shape)} at {k}")
                    dst.data.copy_(src.to(dst.dtype))
                    applied += 1
    print(f"grafted {applied} tensors; skipped {skipped}")
    if missing:
        print(f"WARNING: {len(missing)} source keys had no base target, e.g. "
              f"{missing[:5]}")
    save_and_verify(base, base_name, src_dir, merged_dir)


def remerge(base_name: str, adapter_dir: Path, merged_dir: Path) -> None:
    """Re-apply the LoRA adapter onto the full multimodal base."""
    import torch
    from safetensors.torch import load_file

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scaling = alpha / r
    assert not cfg.get("use_dora") and not cfg.get("use_rslora"), "plain LoRA only"
    print(f"r={r} alpha={alpha} scaling={scaling}")

    base = load_full_base(base_name)
    sd = load_file(str(adapter_dir / "adapter_model.safetensors"))

    modules: dict[str, dict] = {}
    for k in sd:
        if ".lora_A.weight" in k:
            modules.setdefault(k[: k.index(".lora_A.weight")], {})["A"] = k
        elif ".lora_B.weight" in k:
            modules.setdefault(k[: k.index(".lora_B.weight")], {})["B"] = k
    print(f"merging {len(modules)} LoRA modules...")

    def resolve(model, mod_path):
        # the text tower may be nested under model.language_model.* in the
        # multimodal model; try the recorded path first, then remap.
        try:
            return model.get_submodule(mod_path)
        except AttributeError:
            alt = mod_path.replace("model.", "model.language_model.", 1)
            return model.get_submodule(alt)

    merged_n = 0
    for prefix, ab in sorted(modules.items()):
        assert "A" in ab and "B" in ab, f"unpaired adapter at {prefix}"
        mod_path = prefix.replace("base_model.model.", "", 1)
        sub = resolve(base, mod_path)
        A = sd[ab["A"]].to(torch.float32)          # [r, in]
        B = sd[ab["B"]].to(torch.float32)          # [out, r]
        delta = (B @ A) * scaling                    # [out, in]
        W = sub.weight
        assert W.shape == delta.shape, f"shape {W.shape} vs {delta.shape} at {mod_path}"
        W.data = (W.data.to(torch.float32) + delta).to(W.dtype)
        merged_n += 1
    print(f"applied {merged_n} deltas")
    save_and_verify(base, base_name, adapter_dir, merged_dir)


def merge_qwen_multimodal(base_name: str, adapter_dir: Path,
                          merged_dir: Path) -> None:
    """Merge a LoRA adapter into the full multimodal Qwen3.6 base (for train.py)."""
    remerge(base_name, adapter_dir, merged_dir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["graft", "remerge"], default="graft",
                   help="graft from an existing merged checkpoint (default) or "
                        "re-merge from the LoRA adapter")
    p.add_argument("--base", default=DEFAULT_BASE,
                   help="full multimodal base model id")
    p.add_argument("--source", default=DEFAULT_SOURCE,
                   help="graft mode: merged text checkpoint (Hub id or dir)")
    p.add_argument("--adapter-dir", type=Path, default=ADAPTER_DIR)
    p.add_argument("--merged-dir", type=Path, default=MERGED_DIR)
    p.add_argument("--repo", default=REPO, help="HF model repo to push")
    p.add_argument("--dry-run", action="store_true",
                   help="only verify an existing merged dir; no load/push")
    p.add_argument("--no-push", action="store_true", help="skip Hub push")
    args = p.parse_args()

    load_dotenv(HERE)

    if args.dry_run:
        verify_dir(args.merged_dir)
        return

    if args.mode == "graft":
        graft(args.base, args.source, args.merged_dir)
    else:
        remerge(args.base, args.adapter_dir, args.merged_dir)

    if args.no_push:
        print("--no-push set; skipping Hub push")
        return
    url = push_folder_to_hub(args.merged_dir, args.repo, private=False)
    print(f"DONE merged_push_url={url}")


if __name__ == "__main__":
    main()
