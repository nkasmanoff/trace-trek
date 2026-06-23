# Training on a rented GPU

QLoRA on `CohereLabs/North-Mini-Code-1.0` (30B-A3B MoE). One 80GB H100 is
plenty; an A100-80GB also works. Expect 1-3 hours for a few thousand samples
(~$5-10 on RunPod at ~$2-3/hr).

## Important: brand-new architecture

`cohere2_moe` requires `transformers >= 5.8.0` (per the model's config.json) —
install transformers from source if the release isn't out yet. Unsloth may or
may not load it on day one; `train.py` automatically falls back to plain
transformers + PEFT + TRL if Unsloth fails. Both paths mask the loss to
assistant turns only (Unsloth via `train_on_responses_only`, the fallback via
a custom `AssistantOnlyCollator`) so the model is never trained to generate
tool outputs / user / system tokens. On startup `train.py` runs a chat-template
preflight and aborts if reasoning or tool calls don't render — a silent drop
there would mean training on nothing useful.

## Pod setup (RunPod, PyTorch 2.x CUDA 12 image)

```bash
pip install -U "transformers>=5.8.0" trl peft datasets accelerate bitsandbytes wandb
pip install -U unsloth           # optional but preferred
# if transformers 5.8.0 isn't released:
# pip install git+https://github.com/huggingface/transformers.git
```

## Run a cycle

From the Mac, upload the dataset (train + eval splits):

```bash
rsync -avz dataset/sft.jsonl dataset/sft-eval.jsonl train/train.py <pod>:/workspace/
```

On the pod:

```bash
cd /workspace
export WANDB_API_KEY=...        # enables W&B logging (omit to disable)
python train.py --data sft.jsonl --eval-data sft-eval.jsonl --out outputs \
    --epochs 3 --lr 5e-5
# more capacity (more VRAM): add --include-mlp
# larger datasets (>500 samples): --epochs 2 --lr 1e-4
```

W&B logs to project `north-mini-sft` (override with `--wandb-project`,
`--run-name`; force off with `--no-wandb`). Train/eval loss, LR schedule,
and the full hyperparameter config are tracked per run; eval loss is
computed every 20 steps (`--eval-steps`).

What to watch in W&B:

- `train/loss` should drop fast in epoch 1, then flatten
- `eval/loss` is the overfit signal — stop when it turns upward
  (with ~95 samples expect the turn during epoch 2-3)

Download the merged checkpoint back to the Mac (~60GB, bf16):

```bash
rsync -avz <pod>:/workspace/outputs/merged/ ./checkpoints/v2-merged/
```

Then on the Mac:

```bash
../deploy.sh ./checkpoints/v2-merged v2
```

## Tips

- Loss should drop quickly in epoch 1 and flatten; if it goes to ~0 you are
  memorizing — reduce epochs or LR.
- Keep LoRA r=16 until a cycle shows gains; bigger adapters overfit small
  trace datasets.
- The router is deliberately excluded from LoRA targets; adapting it
  destabilizes MoE training.
- If you only want the adapter (to merge locally later), pass `--no-merge`
  and rsync `outputs/adapter/` (~100MB) instead of the 60GB merge.
