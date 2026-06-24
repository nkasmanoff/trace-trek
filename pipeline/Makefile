PYTHON ?= python3
UPSTREAM ?= http://127.0.0.1:63450

# HF dataset (single train split; pull_dataset.py / train.py hold out --eval-frac)
HF_REPO ?= nkasmanoff/opencode-sft
EVAL_FRAC ?= 0.1

# opencode model id used by the eval harness (see eval/agents.json). Point this
# at a served fine-tuned checkpoint after deploy to measure the lift.
EVAL_MODEL ?= frontier/poolside/laguna-xs.2
LOCAL_MODEL ?= llamacpp/laguna-xs.2

.PHONY: dataset split harvest train eval eval-baseline eval-tasks \
	eval-set-pick eval-set clean help upload

help:
	@echo ""
	@echo "=== trace-trek pipeline (Laguna-XS.2) ==="
	@echo ""
	@echo "DATA:"
	@echo "  make split        - pull $(HF_REPO), split into dataset/sft.jsonl + sft-eval.jsonl"
	@echo "  make harvest       - turn the held-out eval split into eval/eval-tasks.jsonl"
	@echo ""
	@echo "EVAL FIRST (baseline, before training):"
	@echo "  make eval-baseline - run harvested tasks through opencode on base Laguna"
	@echo ""
	@echo "TRAINING:"
	@echo "  make train         - QLoRA fine-tune Laguna on dataset/sft.jsonl"
	@echo ""
	@echo "EVAL AFTER (post-deploy):"
	@echo "  make eval          - tool-validity + chat-sanity + opencode gate vs live server"
	@echo "  make eval-baseline EVAL_MODEL=<deployed>  - re-score the harvested tasks"
	@echo "  make eval-tasks    - replay held-out git-backed tasks (dataset/tasks-test.jsonl)"
	@echo "  make eval-set      - agent comparison (eval/agents.json), embed into viewer"
	@echo ""
	@echo "OTHER:"
	@echo "  make clean         - remove built dataset + harvested tasks"
	@echo ""

# ── data ───────────────────────────────────────────────────────────────────
# Pull the HF dataset and split it into train/eval. The eval split is held out
# from training and reused by `make harvest` so the same rows are never both
# trained on and evaluated.

split:
	$(PYTHON) train/pull_dataset.py --repo $(HF_REPO) --out dataset \
		--eval-frac $(EVAL_FRAC)

harvest:
	$(PYTHON) eval/harvest_eval_tasks.py --eval dataset/sft-eval.jsonl \
		--out eval/eval-tasks.jsonl

# Legacy local build from raw proxy/cursor traces (needs dataset/build_dataset.py).
dataset:
	$(PYTHON) dataset/build_dataset.py --opencode raw/opencode --cursor raw/cursor \
		--out dataset/sft.jsonl --exclude-tasks dataset/tasks-test.jsonl --eval-frac 0.1

upload:
	$(PYTHON) dataset/build_dataset.py --opencode raw/opencode --cursor raw/cursor \
		--exclude-tasks dataset/tasks-test.jsonl --eval-frac 0.1 \
		--stdout 2>/dev/null | \
	$(PYTHON) ../scripts/upload_to_hf.py --repo $(HF_REPO) \
		--path-in-repo train.jsonl

# ── training ────────────────────────────────────────────────────────────────
# bf16 LoRA fine-tune Qwen3.6 on the local split (run `make split` first).

train:
	$(PYTHON) train/train.py --data dataset/sft.jsonl \
		--eval-data dataset/sft-eval.jsonl --out outputs

# ── evaluation ──────────────────────────────────────────────────────────────

# Baseline (and post-deploy) opencode-harness score on the harvested tasks.
eval-baseline:
	$(PYTHON) eval/run_baseline.py --tasks eval/eval-tasks.jsonl \
		--model $(EVAL_MODEL)

eval:
	$(PYTHON) eval/run_evals.py --base-url $(UPSTREAM)/v1

EVAL_RESULTS = raw/replay/results-eval-$(shell date +%Y%m%d-%H%M%S).jsonl
eval-tasks:
	$(PYTHON) collect/replay.py --tasks dataset/tasks-test.jsonl --model $(LOCAL_MODEL) \
		--no-synth --results $(EVAL_RESULTS)
	@$(PYTHON) -c "import json; \
		rs=[json.loads(l) for l in open('$(EVAL_RESULTS)')]; \
		p=sum(bool(r.get('passed')) for r in rs); \
		print(f'held-out pass rate: {p}/{len(rs)} -> $(EVAL_RESULTS)')"

eval-set-pick:
	$(PYTHON) eval/pick_eval_set.py

eval-set:
	$(PYTHON) eval/run_eval_set.py --embed

# ── other ───────────────────────────────────────────────────────────────────

clean:
	rm -f dataset/sft.jsonl dataset/sft-eval.jsonl eval/eval-tasks.jsonl
