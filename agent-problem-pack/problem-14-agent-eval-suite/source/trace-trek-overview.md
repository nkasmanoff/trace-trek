# trace-trek (repo overview)

trace-trek is a monorepo for the opencode **self-improvement loop**: collect agent traces,
visualize them, build datasets, train models, evaluate, and deploy.

## Structure

```
trace-trek/
├── viewer/        # Trace visualization (Vite + React) + dev-server middleware
│                  #   - anatomy view, dashboard, and an eval tab
│                  #   - connects to opencode's SQLite DB, exports SFT data, uploads to HF
├── pipeline/      # The self-improvement loop
│   ├── collect/   #   proxy/replay + task generators that gather traces
│   ├── dataset/   #   build_dataset.py quality gate, *.jsonl task files
│   ├── train/     #   QLoRA training (Unsloth / TRL+PEFT)
│   ├── eval/      #   evaluation runners + agent comparison
│   ├── inference/ #   MLX and Modal serving
│   └── deploy/    #   GGUF conversion + llama-server swap
├── agent-problem-pack/  # A reproducible benchmark suite for coding agents (see design notes)
├── menubar/       # macOS menu bar tracker for the viewer dev server
└── opencode.json  # opencode agent config
```

## End-to-end flow

```
opencode SQLite DB ──> viewer app ──> HuggingFace ──> train ──> eval ──> deploy
              (export SFT)     (upload)       (LoRA)    (benchmark)  (serve)
```

The loop only improves the agent if the **eval** stage is trustworthy: it must measure
real capability, resist gaming, and produce comparable numbers across model versions. The
`agent-problem-pack/` is that eval stage — a benchmark the trained agent is run against, and
whose pass/fail + token results feed the decision to deploy a new checkpoint. The viewer's
eval tab launches these runs and renders the results (pass rate, tokens, steps, traces).
