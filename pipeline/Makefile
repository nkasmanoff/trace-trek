PYTHON ?= python3
UPSTREAM ?= http://127.0.0.1:63450
LOCAL_MODEL ?= llamacpp/north-mini-code

.PHONY: dataset train eval eval-tasks eval-set-pick eval-set clean help upload

help:
	@echo ""
	@echo "=== trace-trek pipeline ==="
	@echo ""
	@echo "DATA COLLECTION (via viewer app):"
	@echo "  cd ../viewer && npm run dev"
	@echo "  Then use the UI to export sessions as SFT and upload to HF."
	@echo ""
	@echo "TRAINING:"
	@echo "  make dataset      - build dataset/sft.jsonl from local raw traces (legacy)"
	@echo "  make train        - pull dataset from HF and train on it"
	@echo "  make upload       - upload local dataset to HF"
	@echo ""
	@echo "EVALUATION:"
	@echo "  make eval         - run the eval gate against the live server"
	@echo "  make eval-tasks   - replay held-out dataset/tasks-test.jsonl on $(LOCAL_MODEL)"
	@echo "  make eval-set-pick- pick the agent-comparison eval set (eval/eval-tasks.jsonl)"
	@echo "  make eval-set     - run eval/agents.json over the eval set, embed into viewer"
	@echo ""
	@echo "OTHER:"
	@echo "  make clean        - remove built dataset (raw traces are kept)"
	@echo ""

# ── dataset ─────────────────────────────────────────────────────────────────
# Build dataset/sft.jsonl from locally collected traces (legacy — only needed
# if you have raw proxy/cursor logs; the modern path pushes straight to HF
# from the viewer app).

dataset:
	$(PYTHON) dataset/build_dataset.py --opencode raw/opencode --cursor raw/cursor \
		--out dataset/sft.jsonl --exclude-tasks dataset/tasks-test.jsonl --eval-frac 0.1

upload:
	$(PYTHON) dataset/build_dataset.py --opencode raw/opencode --cursor raw/cursor \
		--exclude-tasks dataset/tasks-test.jsonl --eval-frac 0.1 \
		--stdout 2>/dev/null | \
	$(PYTHON) ../scripts/upload_to_hf.py --repo nkasmanoff/local-code-sft-json \
		--path-in-repo train.jsonl

# ── training ────────────────────────────────────────────────────────────────
# Pull the latest dataset from HuggingFace and run QLoRA fine-tuning.
# Requires HF_TOKEN in the environment or in the root .env.

HF_REPO ?= nkasmanoff/local-code-sft-json

train:
	$(PYTHON) train/train.py --hf-repo $(HF_REPO) --out outputs

# ── evaluation ──────────────────────────────────────────────────────────────

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
	rm -f dataset/sft.jsonl dataset/sft-eval.jsonl
