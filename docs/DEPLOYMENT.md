# Admission boundary

Agent Astra is usable for training, forecasting, controlled experiments and adapter integration. Production authorization remains HOLD until real prerequisites are provided.

`SafetyRuntime` requires:

1. A request verifier for each independently authenticated candidate and repair.
2. A live context verifier returning `valid`, a nonempty `revision`, a finite `expires_at`, and optional trusted unresolved/resolved fact identifiers.
3. An artifact verifier that authenticates a calibration artifact from controlled distribution. Recalculating its public digest is not authentication.
4. Deployment lineage matching the trained models, code, registry, frozen policy family, calibration episode/group manifest, window configuration, repair provider and executor.
5. For committed-step accounting, an execution verifier that binds an authentic execution receipt to the returned `AuthorizedAction`.

Only ALLOW produces a random permit ID. Consumption checks the exact original action, prepared request, current context revision, registry, artifact and lineage, and succeeds at most once in this process. All new candidates require assessment; a risk-window state or a predicted event time never grants execution authority.

The library cannot make a remote tool's side effect atomic with its in-memory permit consumption. A real gateway needs transactions/locking, a shared execution ledger and revocation handling. Missing or invalid state must stop before tool submission. The included simulator mutates only its own in-memory state and its outcomes are marked synthetic. Synthetic artifacts are refused by production admission.

Exceptions returned to clients use reason codes. Sensitive request payloads are retained only for evaluation/verification inside the owning process and are not emitted in public forecast or window records. A production service must still supply its own access control, secure process/storage handling and resource limits.
