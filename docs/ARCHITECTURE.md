# Recurrent temporal–graph model

The implementation lives in `agent1/temporal_graph.py`; `agent_astra.AstraModel` is the public alias. This is a local shared-depth research architecture inspired by published recurrent Transformer and temporal graph ideas, not a statement about GPT-6 internals.

## Data and computation

Each validated prefix supplies event features, ordinal steps, wall-clock deltas, structured safety state, versioned graph nodes and directed typed edges. Graph nodes record first observation, current version availability, last state change, entity identity and persistence flags. Permission changes create new versions and `state_update` edges; old tensors are not rewritten. Final display taint redacts all previews for a resource that is later marked sensitive.

One shared refinement block performs causal temporal attention, directed graph attention, node-to-token and token-to-node attention, and residual feed-forward updates. Dropout is absent. Identical repair inputs therefore receive identical computations in train and evaluation modes. Loops refine the same observed prefix; they neither execute actions nor advance time.

At every iteration:

- A token at step t reads temporal tokens and node versions available at or before t.
- A node version at step s reads temporal tokens at or before s.
- A graph message moves from a node available no later than its destination.
- Evidence gates on historical edges do not depend on the final candidate query.

The causal-token invariant assumes the supplied episode-level goal and policy are fixed conditioning context. The current input schema does not contain a timeline of global policy revisions; an external adapter must supply that information or separate episodes if historical policy timing matters. Current-step admission still requires a live policy check.

This invariant is tested both with a full prefix and with a cropped recent window. Choosing only the latest node version overall would break it: a future observation could remove the earlier fact from memory. The implementation retains the latest persistent version **as of the left window boundary**, plus the detailed nodes inside the window.

Candidate-query attention combines content affinity, a learned content gate and:

`-softplus(step_rate[type])*log1p(action_age) - softplus(wall_rate[type])*log1p(elapsed_seconds)`.

Persistent facts bypass this explicit age penalty. These weights affect representation aggregation, not hard authorization. Learned model weights cannot override a registry or live permission failure.

The selected graph uses learned edge gates; the full graph uses all edges and receives its own event likelihood; the complement uses inverse gates only for research diagnostics. Gates are applied after structural attention normalization, so removing a sole edge does not get cancelled by renormalization. The existing stable competing-risk distribution and exact/interval/censored losses are shared by GRU and Transformer encoders.

## Budgets and caching

Default temporal history: 32 committed actions plus the candidate. Default retained node limit: 2,048; persistent boundary anchors: 512. Graph construction supports at most 8,192 node versions and 16,384 edges within the validated input prefix. `prepare_request` still rejects more than 256 history events. Dataset preparation rejects implicit history truncation once its declared history budget is exceeded; a verified state-summary adapter is needed for longer production histories.

The recorder's detail capacity and the model's history capacity are separately versioned. The recorder is observational in this release: its predicted warning state is not fed back into the same forecast. Model memory retention follows the deterministic configured window and persistent graph facts, avoiding a circular detector-dependent input-selection rule.

`cache.py` holds up to four historical graph-prefix snapshots with a 20,000 node+edge weight budget and up to 1,024 static event encodings with a 2 MiB ndarray budget. Keys cover normalized observations, actions, state, tool semantics, policy and timing; event encodings also bind the trusted goal. Candidate branches clone a history snapshot before appending their pending action. No candidate snapshot is committed to historical memory. Permission or policy changes produce different keys. All entries are private process memory and disappear on restart.

Refined neural token states are recomputed. Reusing their KV/state cache across graph updates would need additional dependency proofs and is intentionally not claimed by this implementation. Tests compare cached preparation and full reconstruction exactly, including branch isolation and caller mutation.

## Configuration

`encoder` selects `loop_transformer` or `gru`. `refinement_steps` selects 1–8 shared iterations. `attention_heads` must divide `hidden_dim`. `node_time_weighting` disables both the explicit time penalty and intrinsic node-history projection for ablation. `ablation` selects full, temporal-only or graph-only computation. Models/checkpoints bind all these fields; a changed architecture requires retraining and calibration.
