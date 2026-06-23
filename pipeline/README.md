# Pipeline

Training and evaluation pipeline for the trace-trek self-improvement loop.

## Data flow

```
opencode sessions
       │  (stored in opencode SQLite DB)
       ▼
viewer/                   # dev server reads DB, converts to SFT
  └── SFT export ──> HuggingFace dataset     ←── primary path
                          │
                    ┌─────┘
                    ▼
              train/train.py --hf-repo <name>    (cloud GPU, QLoRA)
                    │
                    ▼
              deploy/deploy.sh                    (GGUF convert + serve)
                    │
                    ▼
              eval/run_evals.py                   (gate: keep or roll back)
```

The legacy proxy-based data flow (`collect/proxy.py`) is deprecated. Traces are
now collected from opencode's built-in SQLite session DB via the viewer app
(`../viewer/` — run `npm run dev`), which can export sessions in SFT format
and upload them directly to a HuggingFace dataset repo.

## Quick start

### 1. Collect traces

```bash
cd ../viewer
npm run dev
```

Open the viewer, connect to opencode's DB, and use the SFT export or
**Upload to HF** feature. This is the primary data collection path.

### 2. Pull dataset and train

```bash
cd ../pipeline

# Pull dataset from HF and train
python3 train/train.py --hf-repo nkasmanoff/local-code-sft-json --out outputs

# Or pull and train via Makefile
make train
```

See `train/train.py --help` for training options (base model, LoRA rank,
learning rate, etc.).

### 3. Deploy

```bash
./deploy/deploy.sh /path/to/merged-checkpoint v2
./deploy/deploy.sh --serve-only v1   # rollback
```

### 4. Evaluate

```bash
make eval              # tool-call validity + chat sanity gate
make eval-tasks        # held-out task pass rate
make eval-set          # agent comparison (eval tab in viewer)
```

## Tasks

Generate held-out evaluation tasks:

```bash
python3 collect/make_tasks.py
make split             # -> dataset/tasks-train.jsonl + dataset/tasks-test.jsonl
```

The test split is excluded from training data and used only for evaluation.

## Dataset quality gate

`dataset/build_dataset.py` applies filters shared by both the viewer export
path (shells out via `--sft-input`) and the legacy proxy-log build:

- drops sessions ending in a user correction/complaint
- drops malformed tool calls
- drops tool-call loops (same call 3+ times)
- drops samples over the token cap
- deduplicates identical/prefix trajectories
