# Migration to Agent Astra 0.4

The original Agent2 source directory was unavailable. Agent2 behaviour was reconstructed against its surviving regression suite, then extended. Historical reports under `docs/history` describe previous runs, not the current release.

| Object | Current contract |
|---|---|
| Project/CLI | `agent-astra`, `python -m agent_astra`; complete `agent1` compatibility package included |
| Input | `agent-astra-input-v1`, exact action preserved separately from lossy features |
| Dataset | `agent-astra-dataset-v1`; regenerate using `prepare` |
| Graph/encoder | Immutable observation versions; `agent-astra-temporal-graph-v1` |
| Checkpoint | `agent-astra-checkpoint-v1`, strict state loading and whole-package implementation digest |
| Head metadata | `agent-astra-head-contract-v1`, actual optimizer batches and best-checkpoint coverage |
| Forecast | Retained `agent1-forecast-v2`, plus explicit encoder and graph-attention diagnostics |
| Selector | Retained `agent1-temporal-policy-v2`; shared by runtime and closed-loop replay |
| Calibration | `agent1-policy-calibration-v2` with required window-configuration lineage |
| Window | `agent-astra-risk-window-v1`, observation and verified execution counters separated |

Old checkpoints cannot be loaded with this architecture and are never accepted through `strict=False`. Prepare the data again, retrain, declare the whole finite policy family before collecting outcomes, then run all policies through the same independent held-out episode/group set. Recalibrate after changing model inputs, loop count, retention budgets, policy thresholds, window settings or executor semantics.

A positive `necessity_weight` remains unsupported until genuine removable-evidence labels and a well-defined experiment exist. Sufficiency, full-graph event likelihood, concept and stable-edge labels are supported with masks. Missing labels do not mark an output head trained.

This release does not simulate or certify future action paths, identify causal blame from attention, or supply an external execution gateway. The risk window records detection history and supported memory; it is not a count of safe actions remaining.
