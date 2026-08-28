# Agent New

Agent New is a clean P2THN-v2 implementation for assessing a proposed tool
action **before** it produces a real side effect. It combines hard, trusted
capability checks with an intervention-conditioned neural forecast.

This repository contains an untrained model implementation and engineering
smoke tests. It intentionally contains no pretrained safety checkpoint and
makes no benchmark-performance claim.

## Architecture

```text
trusted goal + policy + observed prefix + unexecuted candidate action
                                  |
                   trusted tool capability registry
                                  |
              persistent provenance / permission graph
                                  |
       +---------------- intervention branches ----------------+
       | none | approval | redact | constrain | sandbox | deny |
       +--------------------------------------------------------+
                                  |
          step/time encoder + deterministic relational GNN
                                  |
           differentiable evidence mask used by prediction
                                  |
            factorized first-event competing-risk head
                 P(event at t) x P(cause | event)
                                  |
             group-calibrated risk upper bounds / ensemble
                                  |
           ALLOW / HOLD / REPLAN / CONFIRM / BLOCK
```

## Improvements over the previous local prototype

- Any-event probability and cause are factorized, with a low base-rate
  initialization instead of a saturated `K+1` softmax.
- Right-censored intervention losses operate in stable logit/survival space;
  there is no `clamp` followed by zero-gradient BCE.
- Intervention branches share deterministic graph parameters and contain no
  independent dropout noise.
- Permission is not assumed to equal safety. Approval, redaction, destination
  constraint, sandboxing, and capability restriction are separate branches.
- Tool operations, required scopes and side effects come from a trusted
  registry; Agent-declared capability fields are ignored.
- Unknown tools are represented explicitly and fail closed.
- Unicode character/byte n-gram features preserve Chinese and other non-ASCII
  input.
- Graph nodes retain persistent source, data, destination, permission and
  policy identities, with readable node/edge metadata.
- Evidence edge weights participate in message passing and are trained with
  sufficiency, necessity and sparsity objectives.
- Calibration aggregates prefixes by trajectory before fitting or selecting a
  safe-trajectory false-positive operating point.
- Checkpoints separate JSON metadata from tensor state, verify SHA-256 lineage,
  and load tensors only with `weights_only=True`.

## Package layout

```text
agent_new/
  registry.py      trusted capability specifications
  features.py      bounded Unicode features
  graph.py         persistent heterogeneous graph
  data.py          request validation, interventions and collation
  model.py         temporal/graph/evidence/competing-risk model
  losses.py        censored, intervention and evidence objectives
  calibration.py   group-aware conformal bounds and trajectory FPR
  policy.py        minimum-cost intervention policy
  runtime.py       hard constraints + calibrated ensemble forecast
  checkpoint.py    integrity-checked safe checkpoint loading
  training.py      compact training and early-stopping loop
  metrics.py       trajectory-aware metrics/bootstrap
  cli.py           trained-checkpoint prediction CLI
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e .
```

Run the full engineering smoke and tests:

```bash
.venv/bin/python scripts/smoke.py
.venv/bin/python -m unittest discover -s tests -v
```

The smoke command uses random weights only. Its numerical risks are not safety
predictions; it verifies request parsing, six graph branches, forward/backward,
the censored loss and finite gradients.

## Input contract

See [examples/unsafe_request.json](examples/unsafe_request.json). A request has:

```json
{
  "trusted_goal": "...",
  "policy": {"id": "...", "clauses": ["..."]},
  "history": [
    {
      "event_id": "...",
      "timestamp": 1.0,
      "observation": {
        "source_id": "...",
        "source_type": "...",
        "trust_level": "trusted|untrusted|unknown",
        "content": "...",
        "data_ids": []
      },
      "action": {"tool": "...", "arguments": {}},
      "permission_receipt": {
        "granted_scopes": [],
        "approval_verified": false,
        "receipt_verified": true
      },
      "runtime": {
        "actor_verified": true,
        "sandboxed": false,
        "user_confirmed": false,
        "sensitive_data_ids": [],
        "known_destinations": [],
        "policy_evaluation_verified": true,
        "policy_clause_results": {"clause-id": true},
        "intervention_policy_evaluations": {
          "verified_approval": {
            "verified": true,
            "policy_clause_results": {"clause-id": true}
          }
        }
      }
    }
  ],
  "candidate": {"event_id": "...", "action": {"tool": "..."}}
}
```

Security booleans must be real JSON booleans; strings such as `"false"` are
rejected. Bare destination allowlist entries are exact matches. Domain-wide
authority must be explicit, for example `domain:internal.example`.

The caller must establish the authenticity of the trusted goal, policy,
permission receipt and runtime metadata. The model does not manufacture trust.
For admission, `SafetyRuntime` additionally requires a trusted-context verifier.
The package provides `sign_request` and `HMACRequestVerifier`; deployments may
replace HMAC with their own signature/attestation service.

```python
from agent_new.trust import HMACRequestVerifier, sign_request

signed = sign_request(
    request,
    secret,
    issuer="policy-engine",
    subject="agent-1",
    nonce="single-use-nonce",
)
verifier = HMACRequestVerifier(secret, ["policy-engine"])
```

## Python API

```python
import json
from pathlib import Path

from agent_new import AgentNewConfig, AgentNewModel, ToolRegistry
from agent_new import collate_prepared, prepare_request

registry = ToolRegistry.from_json("configs/tool_registry.json")
request = json.loads(Path("examples/unsafe_request.json").read_text())
batch = collate_prepared([prepare_request(request, registry)])

model = AgentNewModel(AgentNewConfig())
outputs = model(batch)
print(outputs["risk_matrix"].shape)  # [batch, intervention branches]
```

Training labels use:

- `event_time`: one-based first-event action bin;
- `event_type`: zero-based cause index;
- `censor_time`: number of fully observed no-event bins;
- per-intervention labels under `branch_targets[name]`, or a weaker
  `intervention_event_targets [B, I]` horizon label matrix.

Use `agent_new.training.fit` for the training loop, fit calibration on separate
trajectory groups, then save the state with `agent_new.checkpoint.save_checkpoint`.
For confidence level `1-alpha`, calibration requires at least
`ceil(1/alpha)-1` independent unsafe groups plus safe groups. The stored score
protocol binds the ensemble standard-deviation multiplier.

## Runtime prediction

Prediction requires trained checkpoints. Multiple independently trained
checkpoints are recommended so the runtime can estimate epistemic dispersion.
Each checkpoint must contain distinct `extra_metadata.training_run_id` and
`extra_metadata.seed` values; repeated or unidentified runs cannot authorize an
`ALLOW` decision.

```bash
agent-new-predict \
  --request examples/unsafe_request.json \
  --registry configs/tool_registry.json \
  --checkpoint artifacts/seed-1 \
  --checkpoint artifacts/seed-2 \
  --calibration artifacts/calibration.json
```

The runtime applies registry/policy prerequisites before neural risk. If every
branch violates a hard prerequisite, the result is `BLOCK` regardless of model
weights. Without both calibration and at least two checkpoints, the runtime is
conservative and cannot emit a model-based `ALLOW`.

CLI exit codes are `0=ALLOW`, `20=CONFIRM`, `21=REPLAN`, `22=HOLD`, and
`23=BLOCK`. Input, integrity, and configuration errors use the normal nonzero
Python error exit. An executor must not treat every successful JSON print as
authorization.

The current calibration artifact certifies only the actual `none` branch.
Repair-branch risks are reported for research, but the runtime will not certify
or automatically commit a repair until intervention-conditional simultaneous
calibration is implemented.

An `ALLOW` response contains only an opaque commit SHA-256, event ID and tool
name. The executor must rebuild the prepared request and verify the digest
immediately before committing the action. Sensitive payload and permission
receipts are intentionally not printed by the CLI.

## Existing-dataset pipeline

The dataset plan in `dataset_training.docx` is implemented with strict role
separation:

- **Training:** ASSEBench safety/security plus LinuxArena trajectories.
- **Internal split:** deterministic 80/10/10 train/validation/calibration over
  complete conversation/task groups.
- **Training batch target:** 70% ASSEBench and 30% LinuxArena through a
  source-balanced sampler.
- **Frozen tests only:** ATBench and OpenAgentSafety never enter the initial
  training corpus.
- ASSEBench event time is explicitly recorded as a
  `weak_terminal_proxy`; strict/loose alternative labels and vendored
  AgentHarm/RJudge files are not imported.
- LinuxArena `side_task`, `policy`, and `side_task_success` remain label or
  lineage fields. They never enter the model request. `has_side_task` trains
  the attack-intent head, `side_task_success` trains harmful outcome, and
  `main_task_success` trains utility.

The model therefore has three supervised outputs:

1. first-event hazard, event time, and event type;
2. hidden attack-intent probability;
3. main-task utility probability.

### 1. Review the download plan — no network writes

```bash
.venv/bin/python scripts/download_datasets.py --profile train
.venv/bin/python scripts/download_datasets.py --profile eval
```

The downloader is dry-run by default. Network and disk acquisition occurs only
when a human explicitly adds `--execute`. This repository did not download the
datasets during implementation.

### 2. Normalize and split authorized local data

```bash
.venv/bin/python scripts/prepare_datasets.py --profile train
.venv/bin/python scripts/prepare_datasets.py --profile eval --overwrite
```

Outputs include deterministic gzip JSONL files, separate derived tool
registries, SHA-256 file manifests, group-leakage lineage, and source counts.
Derived registries are marked as weak dataset semantics and must not be reused
as production authorization registries without review.

### 3. Inspect batches fed to the model

```bash
.venv/bin/python scripts/feed_model.py --split train --forward
```

### 4. Train independent seeds — visible epochs

```bash
.venv/bin/python scripts/train_model.py \
  --seed 101 --run-id seed-101 --output-dir artifacts/runs/seed-101

.venv/bin/python scripts/train_model.py \
  --seed 202 --run-id seed-202 --output-dir artifacts/runs/seed-202
```

The console displays `Train Epoch 001/040` and
`Validation Epoch 001/040` progress plus epoch-level hazard, intent, and utility
metrics.

### 5. Validate and calibrate the multi-seed ensemble — visible epochs

```bash
.venv/bin/python scripts/validate_model.py \
  --checkpoint artifacts/runs/seed-101/best \
  --checkpoint artifacts/runs/seed-202/best \
  --output-dir artifacts/validation
```

Validation writes actual-branch group conformal calibration, a
safe-trajectory-FPR operating threshold, and calibrated copies of every
checkpoint. The requested confidence level requires enough independent unsafe
trajectory groups; undersized calibration sets fail closed.

### 6. Run frozen external tests — visible epochs

```bash
.venv/bin/python scripts/test_model.py \
  --checkpoint artifacts/validation/checkpoints/seed-101 \
  --checkpoint artifacts/validation/checkpoints/seed-202 \
  --calibration artifacts/validation/calibration.json \
  --operating-threshold artifacts/validation/operating_threshold.json
```

The test script refuses to fall back to validation data. It evaluates only
manifest splits beginning with `test_` and prints `Test Epoch 001/001` for each
frozen dataset.

## Required evidence before research claims

This implementation fixes architectural problems; it does not close the
empirical gap. A research evaluation still requires:

- real action-level first-event and censoring labels;
- matched intervention outcomes, including authorized-but-harmful cases;
- independent frozen test/OOD splits;
- simple rules, flags-only, temporal-only, graph-only and standard survival
  baselines;
- multiple seeds and trajectory/scenario cluster confidence intervals;
- live pre-side-effect interception with attack success and benign utility;
- evidence-path deletion/insertion faithfulness tests.
