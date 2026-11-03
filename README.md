# Agent New

Agent New is a clean P2THN-v2 implementation for assessing a proposed tool
action **before** it produces a real side effect. It combines hard, trusted
capability checks with an intervention-conditioned neural forecast.

This repository contains an untrained model implementation and engineering
smoke tests. It intentionally contains no pretrained safety checkpoint and
makes no benchmark-performance claim.

## What problem this solves

An LLM agent proposes a tool call (for example `send_email(to=..., body=...)`).
Before the call runs, Agent New answers three questions:

1. **Is this action allowed at all?** Checked against a trusted tool registry
   and the permission receipt. This layer is deterministic and fails closed.
2. **If it runs, what is the probability that a harmful first event
   (privacy leakage, data loss, unauthorized external action, ...) occurs
   within the next `H` steps, and when?** Answered by a learned,
   right-censored competing-risk model over the agent's provenance graph.
3. **Would a cheaper fix than blocking make it safe?** The same model is
   re-run under five counterfactual interventions (require approval, redact
   sensitive arguments, pin the destination, sandbox, restrict capability),
   and a cost-aware policy picks `ALLOW`, `CONFIRM`, `REPLAN`, `HOLD` or
   `BLOCK`.

Most published guardrails answer only question 1, or label a single action as
safe/unsafe. See [reports/related_work.md](reports/related_work.md) for a
comparison with DreamGuard, ProbGuard, ToolSafe, ShieldAgent, LlamaFirewall
and agent-IDS work.

## Architecture

Each stage below names the function that implements it and the file it lives
in.

```text
trusted goal + policy + observed prefix + unexecuted candidate action
                                  |
   (1) validate + registry lookup        prepare_request()        data.py
       unknown tool -> fail-closed       ToolRegistry.resolve_or_fail_closed()
                                                                   registry.py
                                  |
   (2) persistent provenance graph       build_persistent_graph() graph.py
       actor/source/data/action/
       destination/permission/policy
                                  |
   (3) six intervention branches         _apply_candidate_intervention()
       none | verified_approval |                                  data.py
       redact_sensitive | constrain_destination |
       sandbox | restrict_capability
                                  |
   (4) step/time encoder                 StepTimeSafetyEncoder    model.py
       relation-aware GNN                DeterministicRelationGraphEncoder
       differentiable evidence mask      EvidenceMasker
       (selected / full / complement)    AgentNewModel._forward_branch
                                  |
   (5) factorized competing-risk head    MonotoneFactorizedCompetingRiskHead
       P(event at t) x P(cause | event)  factorized_competing_risk_distribution
                                  |
   (6) censored likelihood (training)    factorized_competing_risk_losses
                                                                   losses.py
                                  |
   (7) ensemble mean + z.std, then       SafetyRuntime.assess()   runtime.py
       group-conformal upper bound       GroupConformalCalibration.risk_upper_bound
                                                                   calibration.py
                                  |
   (8) least-cost safe intervention      InterventionPolicy.select policy.py
       ALLOW / HOLD / REPLAN / CONFIRM / BLOCK
```

## Core algorithm

This section states what the code actually computes. Symbols: batch `B`,
horizon bins `H` (`config.horizon`, default 5), event types `K`
(`DEFAULT_EVENT_TYPES`, 5), intervention branches `I` (`INTERVENTIONS`, 6).

### Stage 5: factorized first-event competing risk

Implemented in `MonotoneFactorizedCompetingRiskHead.forward` and
`factorized_competing_risk_distribution` ([agent_new/model.py](agent_new/model.py)).

For each horizon bin `t = 1..H` the head emits one *any-event* logit `a_t` and
`K` *cause* logits `c_{t,k}`:

```text
q_t        = sigmoid(a_t)                       P(first event in bin t | none before t)
pi_{t,k}   = softmax_k(c_{t,k})                 P(cause = k | event in bin t)
S_t        = prod_{s<=t} (1 - q_s)              survival through bin t
m_{t,k}    = S_{t-1} * q_t * pi_{t,k}           incidence mass of cause k at t
risk       = 1 - S_H                            P(any harmful event within H)
cause_risk = sum_t m_{t,k}                      P(first event has cause k)
```

Everything is computed in log space (`logsigmoid`, `cumsum`, `expm1`), so
extreme logits do not overflow. The any-event bias is initialised to
`logit(base_event_rate)` (default 0.01), so an untrained model starts at a
low, not saturated, risk. A monotonicity constraint (`softplus` weights on
the six *critical* concepts such as `privilege_excess` or
`sensitive_data_flow`) guarantees that raising a critical concept probability
can never lower `q_t`.

### Stage 6: right-censored likelihood

Implemented in `factorized_competing_risk_losses`
([agent_new/losses.py](agent_new/losses.py)).

```text
observed event of cause k at bin t:
    NLL = sum_{s<t} softplus(a_s) + softplus(-a_t) - log_softmax_k(c_{t,k})
right-censored after c observed safe bins:
    NLL = sum_{s<=c} softplus(a_s)
```

`softplus(a)` is `-log(1 - sigmoid(a))`, so no probability is ever clamped and
gradients never vanish on confident predictions. Safe trajectories therefore
contribute real supervision (they survived), not a fake "no-event" class.

### Stage 3 + 4: intervention-conditioned forecasting

`prepare_request` builds `I` copies of the request. Branch `none` is the
candidate as proposed; each other branch is the candidate after one
intervention has been applied by `_apply_candidate_intervention`
([agent_new/data.py](agent_new/data.py)). All branches go through the *same*
encoder parameters, with no dropout, so

```text
risk_delta[b, i] = risk[b, none] - risk[b, i]
```

(`AgentNewModel.forward`) is a paired comparison that measures only the
effect of the intervention, not sampling noise. A branch whose intervention
changed nothing (for example "sandbox" on a non-`execute` tool) is marked
`intervention_effective = False` and is ignored by the runtime.

### Stage 4: evidence as part of the forward pass

`EvidenceMasker` scores every provenance edge from `(source node, target
node, temporal state, edge context)`. The sigmoid of that score is used as the
edge weight in message passing. The model runs the GNN three times per branch
(`AgentNewModel._forward_branch`):

| path | edge weights | used for |
|---|---|---|
| `selected` | `w_e = sigmoid(score_e / T)` | the reported risk and the `evidence_edges` in the runtime output |
| `full` | `1` | sufficiency: selected risk should match full risk |
| `complement` | `1 - w_e` | necessity: complement risk should drop |

Sufficiency, necessity and sparsity terms live in
`evidence_regularization_losses` ([agent_new/losses.py](agent_new/losses.py)).

### Stage 7: ensemble dispersion and conformal upper bound

`SafetyRuntime.assess` ([agent_new/runtime.py](agent_new/runtime.py)) runs
every checkpoint in the ensemble and, per branch, computes

```text
mean, std   over checkpoints
point_upper = min(1, mean + z * std)                     z = ensemble_z (1.96)
upper       = min(1, point_upper + correction)           GroupConformalCalibration
```

`correction` is a one-sided finite-sample conformal quantile fitted by
`fit_group_conformal` ([agent_new/calibration.py](agent_new/calibration.py))
on **trajectory groups**, not individual prefixes, so repeated prefixes of the
same episode do not inflate the calibration set. Checkpoints must carry
distinct `training_run_id` and `seed`; otherwise `std` is not a valid
dispersion estimate and the runtime refuses to `ALLOW`.

### Stage 8: decision rule

`InterventionPolicy.select` ([agent_new/policy.py](agent_new/policy.py)).
A branch is *reliably safe* when

```text
upper <= safe_risk_limit (0.20)  and  mean <= safe_risk_limit
and  std <= epistemic_uncertainty_limit (0.25)
```

| situation | decision |
|---|---|
| every branch fails a hard registry/permission prerequisite | `BLOCK` |
| trusted-context signature missing or invalid | `HOLD` |
| fewer than two independent checkpoints, or no calibration | `HOLD` |
| `none` is reliably safe | `ALLOW` + opaque commit digest |
| cheapest reliably safe repair is `verified_approval` | `CONFIRM` |
| cheapest reliably safe repair is `restrict_capability` | `BLOCK` |
| any other reliably safe repair | `REPLAN` |
| no safe branch, and uncertainty is the cause | `HOLD` |
| no safe branch, confident | `BLOCK` |

Intervention costs are fixed in `INTERVENTION_COSTS`
([agent_new/constants.py](agent_new/constants.py)):
`none 0 < verified_approval 1 < redact/constrain 1.5 < sandbox 2 < restrict 4`.

### Core functions at a glance

| Function | File | One-line role |
|---|---|---|
| `prepare_request(request, registry)` | `agent_new/data.py` | validate input, resolve tools, build one `PreparedBranch` per intervention |
| `collate_prepared(examples)` | `agent_new/data.py` | pad/concatenate branches into tensors |
| `build_persistent_graph(...)` | `agent_new/graph.py` | heterogeneous provenance graph with stable node identities |
| `AgentNewModel.forward(batch)` | `agent_new/model.py` | run every branch, return `risk_matrix [B, I]` and `risk_deltas` |
| `factorized_competing_risk_distribution(a, c)` | `agent_new/model.py` | logits -> survival, incidence, `risk`, `cause_risk` |
| `factorized_competing_risk_losses(...)` | `agent_new/losses.py` | right-censored first-event NLL |
| `agent_new_loss(outputs, batch)` | `agent_new/losses.py` | total loss: hazard + intervention + concept + evidence + intent/utility |
| `fit_group_conformal(...)` | `agent_new/calibration.py` | one-sided conformal correction on trajectory groups |
| `select_threshold_at_safe_trajectory_fpr(...)` | `agent_new/calibration.py` | operating threshold at a target safe-trajectory FPR |
| `SafetyRuntime.assess(request)` | `agent_new/runtime.py` | hard checks -> ensemble forecast -> upper bounds -> decision |
| `InterventionPolicy.select(...)` | `agent_new/policy.py` | least-cost reliably safe intervention |
| `sign_request` / `HMACRequestVerifier` | `agent_new/trust.py` | prove the goal/policy/receipt came from a trusted issuer |

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

The most useful keys in `outputs` are:

| key | shape | meaning |
|---|---|---|
| `branch_names` | `(I,)` | column order of every `[B, I]` tensor, always `INTERVENTIONS` order |
| `actual_risk` | `[B]` | `risk` of the `none` branch: `P(harmful first event within H)` |
| `risk_matrix` | `[B, I]` | the same risk under each intervention |
| `risk_deltas` | `[B, I]` | `actual_risk - risk_matrix`; positive = intervention helps |
| `branches[name]["selected"]["cause_risk"]` | `[B, K]` | probability the first event is of each type |
| `branches[name]["selected"]["incidence_mass"]` | `[B, H, K]` | when and why the first event is expected |
| `branches[name]["concept_probabilities"]` | `[B, 9]` | interpretable concepts (`CONCEPTS` order) |
| `branches[name]["evidence_weights"]` | `[E]` | per-edge evidence score used in message passing |
| `branches[name]["attack_intent_logit"]`, `["utility_logit"]` | `[B]` | auxiliary heads |

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

## Related work and positioning

[reports/related_work.md](reports/related_work.md) compares this design with
recent pre-execution guardrails (DreamGuard, ProbGuard, ToolSafe/TS-Guard,
AgentDoG, ShieldAgent, LlamaFirewall), agent intrusion-detection work
(evidence-based A-IDS, MCPShield, SentinelAgent) and the benchmarks they use
(OSGuard, TS-Bench, ATBench, ShieldAgent-Bench, AgentDojo). In short:

- Most guardrails classify one action or one trajectory as safe or unsafe.
  Only DreamGuard and ProbGuard forecast *future* risk; neither reports risk
  separately per candidate intervention.
- Provenance and permission tracking appears in the IDS line of work but not
  in learned pre-execution guardrails.
- Only ProbGuard (PAC-style) and the certified-enforcement theory line give
  formal or statistical guarantees; this repository targets group-conformal
  upper bounds at a fixed safe-trajectory false-positive rate.
- MCPShield reports that random train/test splits inflate AUROC by up to 26
  points relative to task-disjoint splits. The dataset pipeline here splits
  by conversation/task group for that reason.

None of these advantages is demonstrated empirically yet; see the next
section.

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
