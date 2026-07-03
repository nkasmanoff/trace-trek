# Training on a rented GPU

LoRA on `Qwen/Qwen3.6-35B-A3B` (35B-total / 3B-active hybrid MoE). bf16 LoRA
fits one 80GB H100 (~74GB at modest context). Expect 1-3 hours for a few hundred
samples (~$5-10 on RunPod at ~$2-3/hr).

## Why bf16 LoRA, not bitsandbytes QLoRA

4-bit QLoRA is **discouraged** for the Qwen MoE family: bitsandbytes introduces
large quantization error on these experts (instability / quality loss), and the
fused experts don't quantize cleanly anyway. Since bf16 LoRA already fits one
80GB H100 (~74GB), the default is bf16. `--quant` selects the strategy:

- `none` (default for Qwen) — plain bf16 LoRA. Fits one H100; shards across
  multiple GPUs via `device_map="auto"` if present.
- `fp8` (default for `*-FP8` repos) — native compressed-tensors load, no bnb.
- `4bit` — bitsandbytes QLoRA. Discouraged for Qwen MoE; kept for legacy paths.

### Critical: LoRA targets on the hybrid architecture

Qwen3.6 is a hybrid: ~30 **Gated DeltaNet** (linear-attention) layers + ~10
standard attention layers + a 256-expert MoE. The usual `q/k/v/o_proj` targets
only reach the 10 standard attention layers (~0.02% of params) and **training
NaNs**. `train.py` therefore also adapts the DeltaNet projections
(`in_proj_qkv` / `in_proj_z` / `out_proj`) for the `qwen` chat format. Pass
`--include-mlp` to additionally adapt the MoE expert FFNs (`switch_mlp`).

### Loading

Qwen3.6 is a multimodal (image-text-to-text) checkpoint; we train the text path
only. `train.py` prefers **Unsloth** as the loader (it handles the text-only
path, the MoE kernels, gradient checkpointing, and the DeltaNet targets) and
falls back to transformers + PEFT.

## Chat template + loss masking

Qwen's chat template has **no** `{% generation %}` blocks, so `train.py` derives
the assistant-token mask from turn markers instead: it masks the span between
`<|im_start|>assistant\n` and the following `<|im_end|>` (reasoning + tool calls
+ answer). The `qwen` format sets `preserve_thinking=True` so reasoning from
*historical* assistant turns is rendered (and trained on), not just the last
turn. Templates that *do* expose generation blocks (e.g. Laguna,
`--chat-format laguna`) get an exact mask from
`apply_chat_template(..., return_assistant_tokens_mask=True)`. On startup
`train.py` runs a chat-template preflight and aborts if reasoning, tool calls,
or the assistant mask don't render — a silent drop there would mean training on
nothing useful.

The `opencode-sft` dataset stores tool-call arguments as JSON strings; the Qwen
template iterates `arguments.items()` (needs dicts), so `train.py` parses them
back automatically.

## Sequence length

opencode-sft sessions are long (mean ~30k, p95 ~79k tokens). `--max-seq-len`
defaults to **65536**, which keeps roughly the shorter ~90% of samples. Lower it
(e.g. `--max-seq-len 32768`) if a single 80GB H100 OOMs at this context length;
samples above the cap are skipped (reported at load time).

## Pod setup (RunPod, PyTorch 2.x CUDA 12 image)

```bash
pip install -U -r requirements.txt
# or minimally for the default Qwen bf16 path:
pip install -U "transformers>=5.7.0" trl peft datasets accelerate unsloth wandb
```

`compressed-tensors>=0.14` is only required for an FP8 base. (If `import
transformers` pulls in a broken `torchvision`, reinstall a matching torchvision —
the Qwen processor imports it.)

Delete `unsloth_compiled_cache/` in your training working directory whenever
`unsloth`, `unsloth_zoo`, `transformers`, or `flash-linear-attention` versions
change — Unsloth writes compiled kernels there at import time and the codegen
bakes in the environment at generation time.

## Run a cycle

Pull + split the dataset, then train:

```bash
cd pipeline
make split                       # dataset/sft.jsonl + dataset/sft-eval.jsonl
export WANDB_API_KEY=...          # enables W&B logging (omit to disable)
make train                        # bf16 LoRA on Qwen3.6
```

Or directly (e.g. on a fresh pod, pulling straight from HF):

```bash
python train/train.py --hf-repo nkasmanoff/opencode-sft --out outputs \
    --epochs 3 --lr 5e-5
# more capacity (more VRAM): add --include-mlp
# lower the context if you OOM: --max-seq-len 32768
```

W&B logs to project `qwen-sft` (override with `--wandb-project`,
`--run-name`; force off with `--no-wandb`). Train/eval loss, LR schedule,
and the full hyperparameter config are tracked per run; eval loss is
computed every 20 steps (`--eval-steps`).

What to watch in W&B:

- `train/loss` should drop fast in epoch 1, then flatten
- `eval/loss` is the overfit signal — stop when it turns upward
  (with a few hundred samples expect the turn during epoch 2-3)

## Intermittent behavioral eval during training (`--eval-gate`)

`eval/loss` only measures next-token likelihood. To also track *behavioral*
quality (does the model still emit valid tool calls and clean chat answers?)
while training, pass `--eval-gate`: at training start, every N steps, and at the
end, `train.py` runs `eval/run_evals.py` against the **live in-training model**
and logs the section pass rates to W&B under `eval_gate/*`.

How it works (see `train/eval_server.py`): there's no second GPU and no vLLM, so
`train.py` exposes the in-memory weights via a tiny OpenAI-compatible server
(`/v1/chat/completions`) inside the training process. It reproduces what the
matching vLLM tool-call parser does at deploy time — converting the model's
native `<think>…</think>` + tool-call text into OpenAI `reasoning_content` +
`tool_calls` — so the unmodified `run_evals.py` can score it. The parser is
chosen by chat format (Qwen's `qwen3_coder` `<function=…><parameter=…>` XML, or
Laguna's `<arg_key>/<arg_value>`). Training pauses for each gate
(`model.eval()`), then resumes.

```bash
python train/train.py \
    --data dataset/sft.jsonl --eval-data dataset/sft-eval.jsonl --out outputs \
    --eval-gate --eval-gate-steps 20
```

Flags: `--eval-gate-steps N` (cadence), `--no-eval-gate-at-start` (skip the
step-0 baseline), `--eval-gate-max-new-tokens` (per-prompt generation budget),
`--eval-gate-port`, and `--eval-gate-opencode` (also run the slow opencode-tasks
section — off by default; needs `opencode` on PATH, see below). `--max-samples N`
caps the train set for quick smoke tests.

In W&B you get `eval_gate/tool_validity_rate`, `eval_gate/chat_sanity_rate`
(and `eval_gate/opencode_tasks_rate` if enabled) alongside `train/loss` and
`eval/loss`. Note: gates run generation on the full model, so each one adds a
few minutes — keep N modest on long runs.

### opencode (for the opencode-tasks section / `make eval*`)

`run_evals.py`'s opencode section and the `eval/run_baseline.py` harness shell
out to `opencode`. Install it once on the pod:

```bash
curl -fsSL https://opencode.ai/install | bash   # installs to ~/.opencode/bin
# (npm i -g opencode-ai also works on some setups)
```

With `--eval-gate-opencode`, `train.py` writes an `opencode.json` under
`<out>/opencode-eval/` that registers a `local` provider pointing at the
in-process server, and runs `opencode run -m local/local-code-model`.

### Deploy

For the default `none` (bf16) path, `train.py` writes a merged bf16 checkpoint
to `outputs/merged/` (via Unsloth's `save_pretrained_merged`) for GGUF
conversion via `deploy/deploy.sh`, plus the adapter to `outputs/adapter/`. Serve
the adapter on the Qwen base with vLLM (`--enable-lora`), or the merged
checkpoint directly.

There is no bf16 merge under `--quant fp8` (the base is compressed). Training
then writes just the LoRA adapter to `outputs/adapter/` (tens of MB); serve it
on top of the FP8 base with vLLM `--enable-lora`:

```bash
rsync -avz <pod>:/workspace/outputs/adapter/ ./adapters/v2/
# serve Qwen/Qwen3.6-35B-A3B with --enable-lora and register ./adapters/v2
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

## GPU smoke test (dense Qwen 27B at 64k)

On an H200 pod with a pulled dataset:

```bash
cd pipeline/train
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train.py --quant 4bit --no-unsloth --no-eval --mem-debug \
  --max-samples 2 --smoke-longest --data outputs/sft.jsonl --out outputs
```

Expect `== 4bit LoRA ==`, `liger: fused CE path ACTIVE (qwen3_5)` on step 0,
the two longest samples selected, and either a training step or
`outputs/oom_snapshot.pickle` on OOM.
