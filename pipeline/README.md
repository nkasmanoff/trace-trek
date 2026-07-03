# Pipeline

Training and evaluation pipeline for the trace-trek self-improvement loop.
The default target model is **`poolside/Laguna-XS.2`** (33B-total / 3B-active
MoE), fine-tuned with QLoRA on the **`nkasmanoff/opencode-sft`** dataset.

## Data flow

```
nkasmanoff/opencode-sft (HF, single train split)
                    │
                    ▼
   train/pull_dataset.py   ──>  dataset/sft.jsonl  +  dataset/sft-eval.jsonl
                    │                                        │
                    │                          eval/harvest_eval_tasks.py
                    │                                        │
                    │                                        ▼
                    │                               eval/eval-tasks.jsonl
                    │                                        │
                    │                          ┌─────────────┘
                    │                          ▼
                    │             eval/run_baseline.py  ← BASELINE on base model
                    │             (opencode harness, before training)
                    ▼
        train/train.py    (cloud GPU, bf16 LoRA, assistant-only loss)
                    │
                    ▼
        deploy/deploy.sh                 (GGUF convert + serve)
                    │
                    ▼
   eval/run_baseline.py EVAL_MODEL=<deployed>   ← re-score, measure the lift
   eval/run_evals.py                            (tool/chat/opencode gate)
```

The dataset (`nkasmanoff/opencode-sft`) is real opencode sessions: a user
request plus the assistant's ground-truth trajectory (reasoning + tool calls +
answer). `pull_dataset.py` holds out `--eval-frac` (default 10%) for evaluation,
and `harvest_eval_tasks.py` turns exactly those held-out rows into eval tasks —
so nothing is both trained on and evaluated.

The legacy proxy-based data flow (`collect/proxy.py`) is deprecated.

## Quick start

Recommended order — **evaluate the base model first**, then train, then
re-evaluate to measure the lift.

### 1. Split the dataset

```bash
cd pipeline
make split        # pull nkasmanoff/opencode-sft -> dataset/sft.jsonl + sft-eval.jsonl
```

### 2. Harvest eval tasks + baseline the base model

```bash
make harvest                          # held-out eval split -> eval/eval-tasks.jsonl

# Serve base Laguna so opencode can reach it, e.g.:
#   python inference/laguna_mlx.py            (local, Apple Silicon)
#   modal deploy inference/laguna_modal.py    (cloud, H100 + vLLM)
make eval-baseline                    # opencode-harness score on base Laguna
```

### 3. Train

```bash
make train        # QLoRA fine-tune Laguna on dataset/sft.jsonl (cloud GPU)
```

On a GPU pod you can pull + train in one shot instead:

```bash
python3 train/train.py --hf-repo nkasmanoff/opencode-sft --out outputs
```

See `train/train.py --help` and `train/README.md` for options (LoRA rank,
learning rate, `--max-seq-len`, `--include-mlp`, etc.).

### 4. Deploy

```bash
./deploy/deploy.sh /path/to/merged-checkpoint v2
./deploy/deploy.sh --serve-only v1   # rollback
```

### 5. Re-evaluate (measure the lift)

```bash
# Re-score the SAME harvested tasks against the deployed checkpoint
make eval-baseline EVAL_MODEL=llamacpp/laguna-xs.2

make eval              # tool-call validity + chat sanity + opencode gate
make eval-set          # base vs. fine-tuned agent comparison (eval/agents.json)
```

## Git-backed held-out tasks (optional)

`run_baseline.py` grades open-ended requests by reference-entity coverage. For
verifiable, repo-backed tasks (file-overlap / knowledge grading in an isolated
git worktree), generate them separately:

```bash
python3 collect/make_tasks.py        # -> dataset/tasks-test.jsonl
make eval-tasks                      # replay them on the served model
```

## Dataset quality gate

`dataset/build_dataset.py` applies filters shared by both the viewer export
path (shells out via `--sft-input`) and the HF pull path (`pull_dataset.py`):

- deduplicates identical/prefix trajectories
- drops sessions ending in a user correction/complaint
- drops malformed tool calls and tool-call loops
- drops incomplete exports and vacuous endings (empty assistant, empty subagent)
- drops agentic sessions with zero tool calls (tools present but never used)
- drops samples over the token cap
