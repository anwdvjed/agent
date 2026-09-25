# Agent Astra

Agent Astra 0.4.0 is a standalone PyTorch research project for forecasting risk before a tool action. It combines a causal temporal Transformer, a versioned directed graph, repeated refinement with shared parameters, and a bounded risk-episode recorder.

The name identifies this local project. It does not reproduce an unpublished OpenAI architecture or use proprietary GPT weights. No OpenAI API key is required.

## Install and run

From this directory, using Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m agent_astra describe --model-config configs/model.json
.venv/bin/python -m pytest -q
```

The installed CLI is `agent-astra`. The `agent1` Python module and CLI remain available for compatibility with existing data tools and tests; the project includes all their source files.

Run the complete **synthetic engineering pipeline**:

```bash
.venv/bin/python scripts/run_pipeline.py --formation --groups 16 --output artifacts/my-run
```

This prepares synthetic trajectories, trains two independently seeded models, validates predictions, runs every frozen policy through the simulator, evaluates test data, and produces risk-window traces. Output directories must be empty. Synthetic calibration does not authorize production execution; a small sample normally yields `no_feasible_policy`.

## What changed

- Temporal tokens and graph nodes exchange information on each refinement iteration. The same block is reused; the default is two iterations, with a configurable limit of eight.
- Graph facts have immutable observation versions. Step/time attention and directed messages prevent later observations from entering earlier token states.
- Node aggregation combines candidate relevance, learned content gates and type-specific step/time decay. Persistent permission, policy, sensitive-data and untrusted-source facts receive no explicit age-decay penalty.
- The model keeps 32 recent committed actions plus the pending candidate, with persistent graph anchors from before that boundary. Required evidence exceeding configured capacity produces an error/HOLD rather than silent deletion.
- A risk window starts at the first detected signal. It records assessments separately from confirmed executions, retains a pre-trigger buffer, and requires verified resolution plus low-risk committed actions to close an episode.
- Graph-prefix and event-encoding caches are bounded, copy-isolated and in memory. Cached data never replaces live authorization checks.
- Exact/interval/censored first-event losses, direct reversible readiness heads, utility, real paired outcomes and masked explanation supervision remain connected to training.

Model aggregation weights appear in the `graph_attention` field of predictions. They are not causal responsibility percentages. `bin 1` means the current unexecuted candidate; a predicted future event bin is never permission to continue executing until that bin.

## Use your own data

Follow [the data contract](docs/DATA_CONTRACT.md), then run:

```bash
.venv/bin/python -m agent_astra prepare --episodes /path/to/episodes.jsonl --registry configs/tool_registry.json --output data/research
.venv/bin/python -m agent_astra train --data data/research --output artifacts/seed101 --seed 101 --model-config configs/model.json --config configs/train.json
.venv/bin/python -m agent_astra train --data data/research --output artifacts/seed202 --seed 202 --model-config configs/model.json --config configs/train.json
.venv/bin/python -m agent_astra test --data data/research --registry data/research/registry.json --checkpoint artifacts/seed101/best --checkpoint artifacts/seed202/best --output artifacts/test.json
```

`predict` and `explain` accept the same checkpoint and registry arguments plus `--request` and `--output`. They produce research forecasts, never permits.

## Compare architectures

```bash
.venv/bin/python scripts/benchmark_ablation.py --synthetic --output artifacts/ablation --epochs 1
# Or use a prepared real dataset:
.venv/bin/python scripts/benchmark_ablation.py --data data/research --output artifacts/real-ablation --epochs 10
```

The script compares GRU, one/two/four refinement iterations, no node-time weighting, temporal-only, and graph-only models. It reports parameter counts, bin-specific prediction metrics and warm-forward latency. One synthetic seed is an engineering check, not evidence of real-world accuracy gains. Changing loop depth changes compute even when parameters are shared.

## Execution and current status

Production admission requires a trusted calibration artifact, a live context/revocation verifier and an execution gateway. Missing services, invalid predictions, capacity exhaustion and failed authorization keep execution on HOLD. A permit is a one-use authorization object; only a separately verified execution receipt increments the risk window's committed-action count.

This repository contains no real production gateway or authentic real-world calibration dataset. Its included/generated simulator artifacts are explicitly synthetic. See [verification](IMPLEMENTATION_REPORT.md), [architecture](docs/ARCHITECTURE.md), [risk windows](docs/RISK_WINDOW.md), [deployment](docs/DEPLOYMENT.md), and [migration](docs/MIGRATION_ASTRA.md).

The original Agent2 source directory was missing at the start of this task. Its compatibility behaviour was reconstructed from the available Agent1 source, the recorded changes and Agent2's 77 regression tests before extending the model. [PROVENANCE.json](PROVENANCE.json) records the available input files; historical reports are under `docs/history/`.
