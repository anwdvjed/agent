# Risk-window contract

`RiskWindow` is an observational state machine for one trusted session. Configuration and output use `agent-astra-risk-window-v1`. Its first trigger means **first detected signal**, not identified causal onset.

Three distinct lengths are preserved: the future event horizon H (default 5), the detailed recent model history (default 32), and a risk episode whose original trigger remains recorded while unresolved. The episode does not expire merely because its detailed ring buffer fills.

## Lifecycle

The inactive state retains eight pre-trigger actions. A score at or above `activate_score`, a hard-constraint reason, or a trusted unresolved fact opens an active episode and copies that prehistory into the detail buffer. Repeated assessments of the same pending event update its record and attempt count; they never advance the executed-action counter.

An active episode enters recovering only when the score is at or below `clear_score` and all pinned facts are resolved by trusted instrumentation. Three distinct, verified low-risk executions resolve it by default. Repeated low-risk assessments cannot satisfy this requirement. A higher score or unresolved fact returns it to active. Later triggers create a new episode without changing the archived original trigger.

Details are capped at 32 records; the original trigger, summary counters and unresolved fact digests survive eviction. Evidence identifiers and candidate identifiers are hashed in public records. More unresolved facts than `max_pinned_facts` causes a sticky capacity error. Resolution cannot be manufactured by dropping an old record or accepting a low model score.

## Runtime integration

A newly owned runtime window imports the independently verified initial history count and a small buffer of its event digests. Historical scores are null because they were not observed by this recorder. Subsequent history must match the confirmed ordered event ledger.

`SafetyRuntime.assess` always performs its ordinary checks. If those checks permit forecasting, it updates the window using the baseline branch's near-term decision score and the live context verifier's unresolved/resolved fact IDs. A safe repair can be selected while the baseline warning remains recorded.

A session with a consumed permit must report its verified execution outcome before it can assess another candidate.

`authorize_and_consume` reserves a candidate once and returns immutable exact action bytes. It does not claim that a tool has executed. `record_execution(permit_id, receipt)` requires an independently supplied `execution_verifier(receipt, authorized_action)` before incrementing the committed-action counter. The production verifier must bind the real receipt to the event, action digest, session and actual completion.

Window configuration is bound by `window_config_digest` in the calibration/deployment lineage. Custom calibration and replay configurations use `--window-config`. The default example is `configs/window.json`; its thresholds are research settings, not generally certified deployment values.

The synthetic replay runner updates windows after actual commits to its in-memory environment. It does not contact an external service or create real approvals.

## Ownership

All state is bounded process-local memory. The runtime serializes its own assessment/consumption operations, caps owned windows and permits, and revokes pending permits when a new assessment changes the session's candidate. A real gateway must enforce external atomicity, idempotency and persistent execution history. Restarting this library is not a substitute for that gateway.
