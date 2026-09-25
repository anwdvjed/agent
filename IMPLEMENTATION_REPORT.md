# Agent Astra 0.4.0 implementation and verification

Delivered in the new `agent-astra` folder on 2026-09-24. This is a local PyTorch temporal-graph research model. No OpenAI proprietary architecture or model weights were accessed or reproduced. Original `agent1` and `agent2` folders were not edited.

## Baseline recovery

Agent2's source package and previous temporary wheel were absent. The available Agent1 source, recorded Agent2 changes and surviving regression tests were used to reconstruct the compatibility behaviour. All 77 inherited tests passed before adding the new encoder. This was a tested reconstruction, not byte-identical recovery. Input-file hashes are recorded in `PROVENANCE.json`; old reports are preserved separately in `docs/history/`.

## Completed changes

| Plan area | Files | Delivered behaviour |
|---|---|---|
| Temporal–graph recurrence | `agent1/temporal_graph.py`, `model.py` | Causal temporal self-attention, directed graph attention and bidirectional token/node exchange, repeated with one shared block. Defaults to two passes; supports one through eight. A GRU comparison encoder remains available. |
| Time and node weight | `temporal_graph.py`, `graph.py`, `data.py`, `runtime.py` | Type-specific action/time age penalties, learned content gating and candidate-query relevance; immutable node versions and persistent memory anchors. Predictions expose node aggregation weights and observation/change steps without raw resource contents. |
| Step window | `risk_window.py`, `runtime.py`, `replay.py`, `configs/window.json` | Pre-trigger buffering, first-detected-signal provenance, bounded detail retention, pinned unresolved facts, hysteresis and verified resolution. Assessment count is separate from committed execution count. |
| Exact execution accounting | `runtime.py` | One-use permits, reassessment revocation, exact payload binding and live revision checks remain. Consuming a permit does not increment executed steps. Verified execution receipts do; another candidate waits for that outcome. Verified initial history sets the starting step count without inventing old scores. |
| Efficient processing | `cache.py`, `graph.py`, `data.py` | Bounded copy-isolated history-prefix and event-encoding caches, deterministic recent-token selection and retained old persistent facts. Cached and full reconstruction match exactly in tests. |
| Training and availability | `objective.py`, `formation.py`, `supervision.py`, `engine.py`, `dataset.py` | Existing stable exact/interval/censored event likelihood, full-graph NLL, masked utility/readiness/concept/edge supervision and actual best-checkpoint coverage are connected to the new encoder. No unknown label becomes a safe negative. |
| Evaluation and distribution | `scripts/benchmark_ablation.py`, `agent_astra/`, `pyproject.toml`, documentation | Standalone `agent-astra` distribution/CLI, compatibility `agent1` API, bin-specific metrics, seven architecture ablations, strict version migration and an install-tested wheel. |

The graph memory uses the latest persistent version **known before the left window boundary**, together with recent node versions. Tests caught the otherwise subtle possibility that a later reobservation could change which old facts were visible to an earlier token. The implementation preserves the causal visibility invariant through every recurrence. Versioned sources are also deduplicated when producing source-group explanations.

## Actual verification

All commands below were executed from this project using the existing local Python/PyTorch environment. The new repository does not depend on the sibling source package; the wheel was separately installed and tested outside the source directory.

| Check | Command or operation | Result |
|---|---|---|
| Unit/integration suite | `../agent1/.venv/bin/python -m pytest -q --tb=short` | **100 passed, 0 failed, 0 skipped** |
| Ordinary synthetic pipeline | `../agent1/.venv/bin/python scripts/run_pipeline.py --groups 16 --output artifacts/release/standard` | Two seeds, two epochs each; prepare/train/validate/calibrate/test/replay completed |
| Formation synthetic pipeline | Same pipeline with `--formation`, output `artifacts/release/formation` | Both current/future readiness heads received actual supervised optimizer updates and checkpoint coverage |
| Ablations | `../agent1/.venv/bin/python scripts/benchmark_ablation.py --synthetic --epochs 1 --output artifacts/release/ablation` | GRU, loop1, loop2, loop4, no-time, temporal-only and graph-only all completed |
| Prediction and explanation CLI | Final formation ensemble; saved synthetic request; explanation budget 3 | Both completed; node weights appeared in output; verified real replay remained null |
| Wheel | `pip wheel . --no-deps --no-build-isolation --no-cache-dir -w dist` | `agent_astra-0.4.0-py3-none-any.whl` built |
| Installed-package check | Installed wheel under `/tmp/agent-astra-install-20260924`, loaded final checkpoints and forecast from `/tmp` | Passed; imported package paths confirmed the isolated installation |

Test coverage includes later-token/node mutation, graph direction, persistent memory retention, finite attention, shared parameter counts, deterministic branch comparisons, capacity failures, window hysteresis, blocked/repeated candidates, confirmed step counting, initial-history import, cache invalidation/isolation, calibration lineage and preserved label/permit invariants.

Both final synthetic calibrations returned **`no_feasible_policy`**, with five independent synthetic groups per candidate policy. The resulting frozen test replays executed zero actions. Those runs are successful engineering validations, not successful production certification.

The one/two/four-pass models each had 29,759 parameters in the small ablation configuration. Greater depth increased warm-forward cost. The one-epoch synthetic bin-1 Brier scores were approximately the same to four decimal places; this experiment does **not** establish better predictive accuracy. GRU had a different parameter count, so it is not a parameter-matched superiority comparison. Full measured results and latency scope are in [verification/ablation.md](verification/ablation.md).

Machine-readable results, source implementation digest and wheel checksum are in [verification/summary.json](verification/summary.json). Generated synthetic inputs, checkpoints and reports live under `artifacts/release/` and are ignored by Git.

## Contracts and boundaries

The standalone project version is 0.4.0. Data and checkpoints use `agent-astra-dataset-v1` and `agent-astra-checkpoint-v1`; model task metadata is `agent-astra-risk-v1`. Old checkpoints are rejected. The forecast preserves the v2 first-event interface and adds encoder and graph-attention diagnostics. Window configuration is part of calibration lineage. Changing model, retention, loop, policy, window or execution semantics requires new training/calibration artifacts.

The risk window is observational in this release. Its own prediction-triggered state is not fed back into the same prediction. The encoder uses a fixed configured recent window plus deterministic persistent graph anchors. Neural/KV states are recomputed; only immutable preparation work is cached. This avoids unsupported reuse across evolving graphs.

The default model retains 32 history actions plus the pending candidate, with up to 2,048 active node versions and 512 persistent boundary anchors. Input history remains explicitly bounded at 256 events; dataset preparation rejects implicit truncation beyond its declared budget. Longer trusted streaming histories require an external verified state-summary adapter. Exhaustion never clears risk or grants permission.

The causal-token guarantee assumes fixed episode-level goal/policy conditioning. This schema does not encode a historical timeline of global policy revisions. A live policy service is still required for current-step admission.

Real-data retraining: **not performed**. Real complete closed-loop certification: **not performed**. Current production admission: **HOLD**. A real artifact-distribution verifier, live context/revocation service, execution-receipt verifier and atomic/idempotent external gateway remain deployment responsibilities. The in-memory runtime does not provide cross-service atomic execution or a distributed persistent ledger.

Node/edge weights describe computation, not causal blame. Explicit future action-path simulation and guarantees about a last safe action are not implemented. Every new candidate still requires evaluation. Positive necessity-loss weight remains rejected until genuine removable-evidence labels and a defined experiment exist.
