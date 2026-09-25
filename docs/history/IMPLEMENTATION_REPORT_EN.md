# Agent2 Implementation Report

Date: 2026-09-23. Source: the sibling `agent1` project. This folder is a separate, installable engineering copy. This implementation followed the sequence in `CODEX_START_PROMPT.txt` and tasks T01–T10 in `CODEX_REFACTOR_SPEC.md`. The filename suffixes in the attachments were not used in Python imports. The Python package remains named `agent1`; the project version is 0.3.0.

## Tasks and Changes Made

| Task | Files | Result |
|---|---|---|
| T01 Time contract | `policy.py`, `tests/test_policy_v2.py`, `tests/test_v2_contracts.py`; the existing `labels.py` was retained | Preserved `event_step-step+1`. Bin 1 represents the current, unexecuted candidate action. Outputs use action bins and a conditional event time; they do not claim a number of safe steps remaining. |
| T02 Complete risk curve | `runtime.py`, `policy.py` | Computes each ensemble member’s first-event CDF before calculating the mean and population standard deviation at each bin. Exposes event mass, survival, CDF, cause-specific cumulative incidence when the head is trained, and the conditional event bin. Invalid values are rejected before scoring. |
| T03 v2 policy and alerts | `policy.py`, `runtime.py`, `replay.py`, `configs/policies.json` | Freezes near-term and full-horizon thresholds, the running-maximum decision score, utility requirements, and a separate alert. Closed-loop replay and runtime admission use the same pure selector. Readiness alerts default to off. |
| T04 Per-head supervision | `formation.py`, `objective.py`, `engine.py`, `runtime.py` | Records effective label counts, optimizer update steps, and bin and condition coverage from training up to the best checkpoint in `agent1-head-contract-v2`. Untrained auxiliary outputs are null; a policy that requires such a head returns HOLD. |
| T05 Calibration and lineage | `calibration.py`, `cli.py`, `runtime.py` | Adds a v2 artifact that binds the frozen policy family, disjoint training and calibration groups, the same complete episode set for every policy, recomputed bounds, and implementation, executor, and repair-provider lineage. v1 artifacts are rejected. |
| T06 Execution permits | `data.py`, `io.py`, `runtime.py` | Adds strict JSON and Unicode checks for the original action and a separate digest of its exact payload. Random permit IDs support one-time consumption within one process, revocation on reassessment, expiration, and context-revision checks. Admission remains HOLD without real verification services. |
| T07 Reversible joint readiness | `formation.py`, `objective.py`, `engine.py`, `runtime.py`, `policy.py` | Adds direct current and future joint-readiness heads with separately masked supervision. The historical first-formation distribution remains available but cannot by itself trigger a permanent alert. Readiness alerts require trained heads and a newly certified policy. |
| T08 Losses in the actual training path | `objective.py`, `engine.py`, `dataset.py` | `research_loss` retains exact, interval, and censored first-event likelihoods. The full-graph branch also receives a real event NLL. Sufficiency compares the H×K cumulative-incidence distribution against a detached full-graph reference. Verified concept and edge labels flow through dataset preparation, batches, and the actual training entry point. Unsupervised placeholder branches do not receive sparsity loss. |
| T09 Graph evidence and redaction | `graph.py`, `evidence.py`, `data.py` | Generates node identity digests from complete bounded values and stable edge IDs. Final sensitivity labels override earlier previews. Public source IDs are redacted. The explanation report separates observed facts, forecasts, structural edges, and model sensitivity. Unverified `argument_sources` no longer alter experimental actions. |
| T10 Regression tests and delivery | `tests/`, `README.md`, `docs/`, `pyproject.toml`, this report | Adds migration guidance, unit and integration tests, complete synthetic pipelines, and a wheel build. |

P2 explicit future-path simulation was not implemented. The model predicts the first unsafe event under a learned continuation distribution. A predicted event bin does not mean that actions before that bin are safe to execute. Each new real action must be assessed again.

## Contracts and Migration

The forecast interface is `agent1-forecast-v2`. Its key fields include `branch_id`, `candidate_event_id`, `horizon_bins`, `time_unit=action_bin`, `bin_one_semantics=current_candidate`, `event_mass_mean`, `survival_mean`, `event_cdf_mean/std`, `cause_names`, `head_availability`, `prepared_digest`, and `expected_event_bin_conditional`. The retained `risk_mean` and `ensemble_disagreement` fields equal the final-bin CDF mean and standard deviation, respectively. Cause-specific cumulative incidence is null when the cause head lacks trustworthy training. The probability-conservation tolerance is `1e-6`; the denominator floor for conditional expected event time is `1e-8`.

The policy format is `agent1-temporal-policy-v2`, with `score_method=cdf_member_mean_std_running_max_v1`. The raw score is `min(1, mean+z*std)`, and the decision score is its running maximum across bins. Both appear in `branch_checks`; neither is described as a probability upper confidence bound. A probability score equal to a stop threshold passes that threshold, while a score above it fails. An alert fires at or above its warning threshold. Only `decision=ALLOW` can receive a permit.

The dataset, checkpoint, and calibration formats are now `agent1-dataset-v3`, `agent1-checkpoint-v3`, and `agent1-policy-calibration-v2`, respectively. Old data must be prepared again, old models must be retrained with strict loading, and old policies must undergo a new complete closed-loop run on the same trusted groups. The runtime cannot silently add a near-term threshold or load old weights with `strict=False`. See `docs/MIGRATION_V2.md` for details. Copied files under `artifacts/` are historical v1 outputs and are not used for v2 certification.

A permit record binds the exact original action bytes, the prepared-request digest, the candidate event and actor, the context revision, the policy-artifact digest, and deployment lineage. `authorize_and_consume` returns an `AuthorizedAction` at most once. A public SHA-256 digest is only a consistency check; it is neither authentication nor a permit lookup key. Runtime admission defaults to HOLD when `artifact_verifier` or `current_context_verifier` is unavailable. The in-memory lock only covers one process; a real executor must provide atomic handling of external side effects.

## Verification Actually Performed

The following commands were run from the `agent2` directory:

```text
../agent1/.venv/bin/python -m pytest -q --tb=short
77 passed

../agent1/.venv/bin/python scripts/run_pipeline.py --groups 16 --output /tmp/agent2-final2-smoke-20260923
Completed synthetic data preparation, two training epochs for each of two seeds, validation, v2 calibration, testing, and closed-loop replay. Calibration returned no_feasible_policy; test replay executed 0 actions.

../agent1/.venv/bin/python scripts/run_pipeline.py --formation --groups 16 --output /tmp/agent2-final2-formation-smoke-20260923
Completed the synthetic formation-data pipeline. The best checkpoints for both seeds recorded training coverage for current and future joint readiness, formation hazard, and future conditions. Calibration returned no_feasible_policy.

../agent1/.venv/bin/python -m pip wheel . --no-deps --no-build-isolation --no-cache-dir -w /tmp/agent2-final-wheels-20260923
Successfully built agent1-0.3.0-py3-none-any.whl.
```

The fixed probabilities, fake ensemble, fake artifact verifier, and simulated environment used in testing exist only in `tests/` or in demo paths explicitly marked synthetic. They do not establish real-world predictive accuracy or deployment safety. The pipeline runs above retrained models on **synthetic data**, not on real data.

## Admission Status and Remaining Dependencies

- Retraining on real data: **No.** Only the ordinary and formation engineering demos explicitly marked synthetic were run.
- Certification using real complete closed-loop data: **No.** Both synthetic calibrations returned `no_feasible_policy`; no production policy is available.
- Current production admission: **HOLD.** Real data, a trusted closed-loop runner, controlled artifact-distribution verification, a live revocation service, a production executor, and a distributed atomic permit and event ledger are missing.
- Explanations can present stable structural edges, independently verified current conditions when available, model forecasts, and bounded sensitivity experiments. They do not prove causal data flow or identify the cause of an incident. Verified lineage for derived arguments has not been established; `verified_replay` is null by default. Removing edges still leaves nodes, temporal information, and whole-graph pooling in place.
- `necessity_weight` remains disabled. A positive value raises an error until genuine removable-evidence labels and a well-defined deletion experiment exist. No real-data ablation or outcome evaluation was performed for old, full, temporal-only, or graph-only models.
- A race remains between local permit consumption and external tool submission. State loss on process restart and concurrent multi-process use also require a production executor or shared transaction system. This repository does not claim atomic execution across services. Self-reported booleans cannot establish human confirmation, sandboxing, approval, or verified replay.
