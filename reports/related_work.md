# Related Work: Intrusion Detection and Runtime Guardrails for LLM Agents

Compiled 2026-11-03. Papers are grouped by the question they answer. The last
row of each table is this repository (Agent New / P2THN-v2) for positioning.

> **Verification status.** arxiv.org was not reachable while this table was
> compiled, so every entry comes from abstracts, search-index snippets, venue
> pages, and secondary summaries. The **Source** column says where each number
> came from. Check each number against the PDF before citing it. `?` means the
> information could not be found, not that the feature is absent.

## Table 1: Overview

| # | Paper | Year / venue | Category | What is monitored | Method | Evaluation data | Headline reported result | Source of numbers |
|---|---|---|---|---|---|---|---|---|
| 1 | **Intrusion Detection for Agentic Processes: Evidence-Based Runtime Monitoring** (Brömme) [arXiv 2610.00151] | Sep 2026, preprint | Agent IDS (specification / expectation-based) | The agentic *process*: workflow state, authorization, communication, mandatory events | Compare evidence-backed observations to a versioned expectation baseline; treats the monitoring layer itself as attack surface; prompt injection seen as a cause of later process deviation | Planned baselines only (single-call filter, pure anomaly detection, runtime spec monitoring) | No empirical results found | Abstract |
| 2 | **Content-Aware Attack Detection in LLM Agent Tool-Call Traffic** / *MCPShield* [arXiv 2605.11053] | 2026, preprint | Agent IDS (learned, traffic-based) | Passive monitor over the full MCP tool-call stream (names, arguments, responses); session = graph of calls | Content embeddings + GraphSAGE/GNN, MLP, tree ensembles; compares metadata-only vs content features | RAS-Eval (primary) + combined dataset | Metadata-only AUROC ≈ 0.64; content features > 0.89; best 0.975 (tree ensemble). GNN 0.917 on RAS-Eval. **Random splits inflate AUROC by up to 26 pts vs task-disjoint splits** | Abstract / HTML snippets |
| 3 | **SentinelAgent: Graph-based Anomaly Detection in LLM-based Multi-Agent Systems** [arXiv 2505.24201] | May 2025, preprint | Agent anomaly detection (multi-agent) | Interaction graph of agents; node-, edge-, path-level anomalies; tool parameters (e.g. unauthorised email recipients) | LLM-powered oversight agent over a dynamic execution graph; can intervene; works with black-box agents | Case studies (email assistant, Magentic-One) | Qualitative / case-study detections | Abstract + memory |
| 4 | **DreamGuard: Efficient Runtime Guardrail for LLM Agents via Risk-Aware World Model** (Lin et al.) [arXiv 2608.05695] | Aug 2026, preprint | Proactive pre-execution guardrail | Trajectory → compact recurrent latent state | World model predicts future latent states; immediate-hazard + prefix-risk signals fused across horizons into intervention decisions | 4 benchmarks + online guardrail eval; baselines Llama-Guard-4, PolicyGuard, GuardAgent, AgentDoG-1.5, SafePred, TRACES | Online: 72.92 % safety rate at 90.38 % utility; ~25 ms per call. (Blog: 96.4 % F1 on SafetyDrift, up to +22 F1 vs per-action guardrails, **unverified**) | Abstract + secondary |
| 5 | **ProbGuard (earlier Pro2Guard): Proactive Runtime Monitoring for LLM Agent Safety via Probabilistic Prediction** (Wang, Poskitt, Wei, Sun) [arXiv 2508.00500] | ASE 2026 | Proactive pre-execution monitor (formal / probabilistic) | Symbolic abstraction of agent state | Learn a DTMC from traces; estimate probability of reaching unsafe states; intervene above a threshold; PAC-style guarantees on the learned model | Embodied agents; autonomous driving | Embodied: −65.37 % unsafe behaviour at 80.4 % task completion (re-prompt); −93.60 % with halting. Driving: warnings up to 38.66 s ahead. Overhead 5–30 ms (embodied), ~100 ms (driving) | Abstract / ASE page |
| 6 | **ToolSafe: Enhancing Tool Invocation Safety of LLM-based Agents via Proactive Step-level Guardrail and Feedback** (Mou et al.) [arXiv 2601.10156] | ACL Findings 2026 | Step-level pre-execution guardrail | Each tool call given interaction history | TS-Guard: guardrail model trained with multi-task RL (request harmfulness, action–attack correlation); TS-Flow feeds verdicts back to a ReAct agent | **TS-Bench** (AgentHarm-Traj, ASB-Traj, AgentDojo-Traj 1,220 samples) | TS-Flow reduces harmful tool invocations by up to 65 % on average while preserving or improving benign task performance | Abstract |
| 7 | **AgentDoG: A Diagnostic Guardrail Framework for AI Agent Safety and Security** (Shanghai AI Lab) [arXiv 2601.18491]; v1.5 [arXiv 2605.29801] | 2026, preprint | Trajectory-level diagnostic guardrail | Full agent trajectories | Fine-tuned guardrail models (4B/7B/8B, Qwen & Llama) that label and *diagnose* why an action is unsafe | **ATBench** (500 trajectories), R-Judge | R-Judge accuracy 92.7 % (4B), **unverified secondary** | Abstract + secondary |
| 8 | **ShieldAgent: Shielding Agents via Verifiable Safety Policy Reasoning** [arXiv 2503.22738] | ICML 2025 | Policy-verification guardrail | Action trajectories vs extracted safety policies | Builds probabilistic rule circuits (Markov logic) from policy docs; retrieves relevant rules; formal verification with tools/code | **ShieldAgent-Bench** (3K instruction–trajectory pairs, 6 web envs, 7 risk categories) + 3 existing | +11.3 % avg over prior methods, 90.1 % recall; −64.7 % API queries, −58.2 % inference time | ICML page |
| 9 | **LlamaFirewall: An Open Source Guardrail System for Building Secure AI Agents** (Meta) [arXiv 2505.03574] | 2025, preprint + OSS | Layered guardrail system | Inputs (PromptGuard 2), agent reasoning trace (AlignmentCheck), generated code (CodeShield) | Jailbreak classifier + LLM chain-of-thought alignment auditor + static analysis | AgentDojo | AgentDojo ASR ≈ 17.6 % → 1.75 % with PromptGuard 2 + AlignmentCheck (**from memory, verify**) | Memory |
| 10 | **What Can Be Enforced? A Theory of Certified Runtime Safety for Tool-Using Agents** [arXiv 2607.22868] | Jul 2026, preprint | Theory of runtime enforcement | What can be guaranteed before an irreversible tool action | Formal characterisation of enforceable properties (builds on Falcone et al., "What can you verify and enforce at runtime?") | Theory | ? | Secondary |
| — | **Agent New / P2THN-v2 (this repo)** | — | Proactive, intervention-conditioned pre-execution forecaster | Unexecuted candidate action + observed prefix + persistent provenance/permission graph | Trusted capability registry hard checks → 6 intervention branches → temporal encoder + relational GNN + differentiable evidence mask → factorised first-event competing-risk head → group-calibrated conformal upper bounds / ensemble → ALLOW/HOLD/REPLAN/CONFIRM/BLOCK | Loaders for ATBench, ASSEBench, LinuxArena, OpenAgentSafety | **None yet** (untrained; README makes no benchmark claim) | — |

## Table 2: Capability matrix

✓ = yes, ◐ = partial, ✗ = no, ? = could not determine from available text.

| Paper | Acts **before** side effect | Uses trajectory / history | **Forecasts** future risk (multi-step) | **Intervention-conditioned** (scores sandbox / redact / approve / deny counterfactually) | Data-flow **provenance** / permission tracking | Trusted capability registry / hard constraints | **Calibrated** or formal guarantee | Explanation / evidence | Empirical evaluation |
|---|---|---|---|---|---|---|---|---|---|
| 1 A-IDS (Brömme) | ◐ (detects deviations at runtime) | ✓ | ✗ | ✗ | ✓ (evidence chain) | ✓ (expectation baseline) | ✗ | ✓ | ✗ |
| 2 MCPShield | ✗ (passive) | ✓ (session graph) | ✗ | ✗ | ◐ (call graph, not data flow) | ✗ | ✗ | ✗ | ✓ |
| 3 SentinelAgent | ◐ | ✓ | ✗ | ✗ | ◐ (interaction graph) | ✗ | ✗ | ✓ (LLM) | ◐ (case studies) |
| 4 DreamGuard | ✓ | ✓ | ✓ | ? | ? | ? | ? | ? | ✓ |
| 5 ProbGuard | ✓ | ✓ | ✓ (DTMC reachability) | ◐ (compares re-prompt vs halt) | ✗ | ◐ (symbolic rules) | ✓ (PAC-style) | ◐ | ✓ |
| 6 ToolSafe / TS-Guard | ✓ | ✓ | ✗ | ◐ (feedback to agent) | ✗ | ✗ | ✗ | ◐ (reasoning) | ✓ |
| 7 AgentDoG | ◐ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ (diagnosis) | ✓ |
| 8 ShieldAgent | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ (policy circuits) | ◐ (probabilistic logic) | ✓ | ✓ |
| 9 LlamaFirewall | ✓ | ◐ (AlignmentCheck) | ✗ | ✗ | ✗ | ✗ | ✗ | ◐ | ✓ |
| 10 Certified runtime safety | ✓ | ? | ? | ? | ? | ✓ | ✓ (theory) | ? | ✗ |
| **Agent New (this repo)** | ✓ | ✓ | ✓ (competing-risk, first event over time) | ✓ (6 branches) | ✓ | ✓ | ✓ (group-aware conformal) | ✓ (evidence mask) | **✗ (to do)** |

## Table 3: Benchmarks and datasets

| Benchmark | From | Granularity | Size / composition | Relevance to this repo |
|---|---|---|---|---|
| **OSGuard** [arXiv 2606.15034] (Mohammadmirzaei & Flanigan, Jun 2026) | Standalone | Action-level labels (allowed / unrelated / unsafe) + risk-augmented end-to-end execution | ? | Directly tests pre-execution decisions; finds guardrails good on isolated actions but weak end-to-end |
| **TS-Bench** | ToolSafe | Step-level tool-call safety | AgentHarm-Traj 731, ASB-Traj 5,237, AgentDojo-Traj 1,220 (868 safe / 352 unsafe) | Step-level pre-execution labels; strong baseline (TS-Guard) |
| **ATBench** | AgentDoG | Trajectory | 500 trajectories | Loader already in `agent_new/datasets/atbench.py` |
| **ShieldAgent-Bench** | ShieldAgent | Instruction–trajectory pairs | 3K, 6 web envs, 7 risk categories | Policy-grounded web-agent risks |
| **RAS-Eval** | Used by MCPShield | Tool-call traffic | ? | Attack detection on MCP traffic |
| **AgentDojo** | Debenedetti et al. 2024 | Prompt-injection tasks | ? | Standard injection benchmark used by LlamaFirewall and TS-Bench |
| ASSEBench, LinuxArena, OpenAgentSafety | Various | — | — | Loaders already in `agent_new/datasets/` |
| GuardianAgentBench [arXiv 2607.20982] | — | Pre-execution interception | Reported 580 scenarios / 6 domains (**could not confirm**) | Verify before citing |

## Positioning summary

1. **Two lines of work that rarely meet.** IDS-style monitors (1–3) model the
   process or traffic and detect deviations, but mostly *after* or *at* the
   event and without forecasting. Guardrails (4–9) act *before* execution but
   mostly judge a single action or trajectory as safe or unsafe.
2. **Forecasting is recent.** Only DreamGuard and ProbGuard predict *future*
   risk. Neither (as far as the abstracts show) asks "what is the risk *if* we
   sandbox / redact / require approval?" Intervention-conditioned forecasting
   is this repo's clearest distinguishing claim.
3. **Provenance is under-used in learned guardrails.** Data-flow and permission
   provenance appears in the IDS framing (1) but not in the learned
   pre-execution guardrails (4–9).
4. **Calibration and guarantees.** Only ProbGuard (PAC-style) and the
   certified-enforcement theory (10) give formal or statistical guarantees.
   Group-aware conformal bounds at a fixed safe-trajectory FPR would be a
   practical contribution.
5. **Evaluation hygiene.** MCPShield shows random splits inflate AUROC by up to
   26 points. Report task-disjoint (and ideally tool- or environment-disjoint)
   splits, and evaluate per trajectory, not per prefix.
6. **Gap to close.** This repo has no empirical result yet. Minimum credible
   comparison: TS-Bench (vs TS-Guard), OSGuard, and the DreamGuard / ProbGuard
   settings, with latency reported (DreamGuard ≈ 25 ms per call is the bar).

## Links

- [2610.00151 Intrusion Detection for Agentic Processes](https://arxiv.org/abs/2610.00151)
- [2605.11053 Content-Aware Attack Detection / MCPShield](https://arxiv.org/abs/2605.11053)
- [2505.24201 SentinelAgent](https://arxiv.org/abs/2505.24201)
- [2608.05695 DreamGuard](https://arxiv.org/abs/2608.05695)
- [2508.00500 ProbGuard](https://arxiv.org/abs/2508.00500) · [ASE 2026 page](https://conf.researchr.org/details/ase-2026/ase-2026-research-track/117/ProbGuard-Proactive-Runtime-Monitoring-for-LLM-Agent-Safety-via-Probabilistic-Predic)
- [2601.10156 ToolSafe](https://arxiv.org/abs/2601.10156) · [ACL Findings 2026](https://aclanthology.org/2026.findings-acl.1850/)
- [2601.18491 AgentDoG](https://arxiv.org/abs/2601.18491) · [2605.29801 AgentDoG 1.5](https://arxiv.org/abs/2605.29801)
- [2503.22738 ShieldAgent](https://arxiv.org/abs/2503.22738)
- [2505.03574 LlamaFirewall](https://arxiv.org/abs/2505.03574)
- [2607.22868 What Can Be Enforced?](https://arxiv.org/abs/2607.22868)
- [2606.15034 OSGuard](https://arxiv.org/abs/2606.15034)
- [2607.20982 GuardianAgentBench](https://arxiv.org/abs/2607.20982)
- [2606.28450 LLM agents security duality survey](https://arxiv.org/abs/2606.28450)
