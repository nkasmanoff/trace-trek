# Training on a rented GPU

LoRA on `poolside/Laguna-XS.2-FP8` (33B-total / 3B-active MoE). One 80GB H100
fits it. Expect 1-3 hours for a few hundred samples (~$5-10 on RunPod at
~$2-3/hr).

## Why FP8, not bitsandbytes QLoRA

94% of Laguna's params live in a fused `LagunaExperts` module (256 experts), not
in `nn.Linear` layers. bitsandbytes 4-bit only swaps `nn.Linear`, so it leaves
the experts in bf16 and the model loads at ~77GB — it OOMs on an 80GB H100.

So the default base is the **FP8 (compressed-tensors) build**: the MoE experts
ship FP8-compressed (~36GB total) while the attention projections stay bf16.
LoRA adapts those bf16 attention linears (`q/k/v/o_proj`); the frozen FP8
experts just run the forward pass. `--quant` selects the strategy:

- `fp8` (default for `*-FP8` repos) — native compressed-tensors load, no bnb.
- `4bit` — bitsandbytes QLoRA (only for architectures bnb can actually quantize).
- `none` — plain bf16 (needs ~70GB+ just for weights; use multiple GPUs / H200).

`--include-mlp` is ignored under fp8 (those linears are compressed).

> **Known incompatibility (transformers 5.12.x + torch 2.12.x):** the FP8 path
> currently fails in the MoE forward. `transformers/integrations/moe.py` casts
> the expert inputs to the experts' FP8 dtype and calls `torch._grouped_mm`,
> which only accepts fp32/bf16/fp16 — so both training *and* generation raise
> `RuntimeError: Expected mat_a to be ... got Float8_e4m3fn`. Until this is fixed
> upstream, run the **bf16 base across two GPUs** instead:
>
> ```bash
> python train/train.py --base poolside/Laguna-XS.2 --quant none ...
> ```
>
> bf16 weights (~67GB) shard across 2×80GB H100 via `device_map="auto"` (naive
> model parallelism), and `_grouped_mm(bf16, bf16)` works. This is also the
> natural "use both GPUs" layout. Re-enable `--quant fp8` once the upstream
> grouped-mm path handles FP8 experts.

## Chat template + loss masking

Laguna's chat template marks assistant spans with Jinja `{% generation %}`
blocks, so `train.py` gets an exact assistant-token mask straight from
`apply_chat_template(..., return_assistant_tokens_mask=True)` and masks the loss
to assistant turns only (reasoning + tool calls + answer). No string-marker
matching. For chat formats whose template lacks generation blocks (e.g. Cohere,
`--chat-format cohere`), it falls back to masking between configured turn
markers. On startup `train.py` runs a chat-template preflight and aborts if
reasoning, tool calls, or the assistant mask don't render — a silent drop there
would mean training on nothing useful.

Laguna XS.2 is supported in `transformers >= 5.7.0`.

The `opencode-sft` dataset stores tool-call arguments as JSON strings; the
Laguna template needs dicts, so `train.py` parses them back automatically.

## Sequence length

opencode-sft sessions are long (mean ~30k, p95 ~79k tokens). `--max-seq-len`
defaults to **65536**, which keeps roughly the shorter ~90% of samples. Lower it
(e.g. `--max-seq-len 32768`) if a single 80GB H100 OOMs at this context length;
samples above the cap are skipped (reported at load time).

## Pod setup (RunPod, PyTorch 2.x CUDA 12 image)

```bash
pip install -U -r requirements.txt
# or minimally for the fp8 path:
pip install -U "transformers>=5.7.0" trl peft datasets accelerate \
    "compressed-tensors>=0.14" wandb
```

`compressed-tensors` is required to load the FP8 base. (If `import transformers`
pulls in a broken `torchvision`, `pip uninstall torchvision` — it's unused here.)

## Run a cycle

Pull + split the dataset, then train:

```bash
cd pipeline
make split                       # dataset/sft.jsonl + dataset/sft-eval.jsonl
export WANDB_API_KEY=...          # enables W&B logging (omit to disable)
make train                        # QLoRA on Laguna
```

Or directly (e.g. on a fresh pod, pulling straight from HF):

```bash
python train/train.py --hf-repo nkasmanoff/opencode-sft --out outputs \
    --epochs 3 --lr 5e-5
# more capacity (more VRAM): add --include-mlp
# lower the context if you OOM: --max-seq-len 32768
```

W&B logs to project `laguna-sft` (override with `--wandb-project`,
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
(`/v1/chat/completions`) inside the training process. It reproduces what vLLM's
`poolside_v1` parser does at deploy time — converting Laguna's native
`<think>…</think>` + `<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value>
</tool_call>` text into OpenAI `reasoning_content` + `tool_calls` — so the
unmodified `run_evals.py` can score it. Training pauses for each gate
(`model.eval()`), then resumes.

```bash
python train/train.py --base poolside/Laguna-XS.2 --quant none \
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

### Deploy (FP8 path)

There is no bf16 merge under `--quant fp8` (the base is compressed and
Hadamard-transformed). Training writes just the LoRA adapter to
`outputs/adapter/` (tens of MB). Serve it on top of the FP8 base with vLLM:

```bash
rsync -avz <pod>:/workspace/outputs/adapter/ ./adapters/v2/
# serve poolside/Laguna-XS.2-FP8 with --enable-lora and register ./adapters/v2
# (see inference/laguna_modal.py — add --enable-lora / --lora-modules)
```

For the `4bit` / `none` paths, `train.py` instead writes a merged bf16
checkpoint to `outputs/merged/` for GGUF conversion via `deploy/deploy.sh`.

## Tips

- Loss should drop quickly in epoch 1 and flatten; if it goes to ~0 you are
  memorizing — reduce epochs or LR.
- Keep LoRA r=16 until a cycle shows gains; bigger adapters overfit small
  trace datasets.
- The router is deliberately excluded from LoRA targets; adapting it
  destabilizes MoE training.
- If you only want the adapter (to merge locally later), pass `--no-merge`
  and rsync `outputs/adapter/` (~100MB) instead of the 60GB merge.
