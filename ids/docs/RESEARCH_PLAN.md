# Research plan

This turns the `ids/` prototype into a publishable study. A paper needs a
claim that is new, testable and measurable; the scaffold alone is a textbook
hybrid IDS, so the first job is to commit to a contribution.

## Research question

**RQ:** Does attaching a per-feature explanation to anomaly alerts, and fusing
them with declarative signatures through a shared de-duplicating alert
manager, reduce analyst alert volume at matched recall compared with
signature-only and anomaly-only baselines?

Sub-questions:

- RQ1: Per attack class, what are precision / recall / FPR of the signature
  path, anomaly path and fused system?
- RQ2: How often does the anomaly path's arg-max feature agree with the
  attack's defining characteristic (e.g. `syn_ratio` for scans,
  `pkts_per_sec` for floods)?
- RQ3: How much does de-duplication reduce alerts per true incident, and
  does it ever suppress a true positive?

## Candidate contributions (pick one primary)

| # | Contribution | Novelty risk | Effort |
|---|---|---|---|
| A (recommended) | Feature-attributed z-score anomaly path, evaluated for attribution accuracy against labelled attack types | low–medium: cheap explainability baselines are rarely measured for correctness | low |
| B | Incident-level (not flow-level) evaluation protocol that counts alerts per true incident | medium: methodological, appeals to reviewers citing Sommer & Paxson | medium |
| C | Learned rule mining: distil anomaly alerts into new YAML signatures | high novelty, higher effort | high |

Recommendation: A as the primary contribution, with B as the evaluation
lens. C is future work.

## Hypotheses

- H1: The fused system achieves recall within 5 points of anomaly-only while
  keeping FPR within 2x of signature-only.
- H2: Arg-max feature attribution matches the expected feature for >80% of
  scan and DoS flows.
- H3: De-duplication cuts alerts per incident by >10x with zero suppressed
  incidents at window w = 60 s.

## Datasets

| Dataset | Role | Notes |
|---|---|---|
| CIC-IDS2017 | primary | Use the corrected labels from Engelen et al. 2021 |
| UNSW-NB15 | generalisation | Different capture setup, tests overfitting to CIC artefacts |
| NSL-KDD | smoke tests only | Too old to headline |

Train the anomaly baseline on benign-only Monday traffic (CIC); evaluate on
Tue–Fri. Never fit on attack days.

## Baselines

1. Signature-only (`rules/default.yaml`)
2. Anomaly-only z-score (ours, no fusion)
3. Isolation Forest on the same six features (`AnomalyDetector` interface)
4. Fused system (ours)

## Metrics

Flow-level: precision, recall, F1, FPR per attack class and overall.
Incident-level: alerts per true incident, incidents missed.
Attribution: fraction of true-positive anomaly alerts whose arg-max feature
matches the expected feature for that attack class.
Cost: flows per second processed.

All metrics are produced by `ids/evaluation/` and written to
`results/*.csv` and `paper/tables/*.tex` so the paper is regenerated from
code, never by hand.

## Experiment protocol

1. `scripts/download_datasets.sh` and place CSVs under `data/`.
2. `python -m ids.evaluation.run --dataset cicids2017 --config configs/eval.json`
3. Sweep theta in {2, 3, 4, 5, 6}; record PR curve.
4. Repeat with Isolation Forest baseline.
5. Repeat on UNSW-NB15 without retuning.
6. `make -C paper` to build the PDF.

## Target venues

Workshop first, then extend: IEEE S&P Workshops (DLS), ACSAC poster,
EuroS&P workshops, or a UK venue such as the Cyber Security PhD Winter
School. Undergraduate / MSc dissertation format also fits.

## Timeline (12 weeks)

| Weeks | Milestone |
|---|---|
| 1–2 | Dataset loaders, evaluation harness end-to-end on CIC-IDS2017 |
| 3–4 | Baselines 1–3 run; first results table |
| 5–6 | Attribution study (RQ2), incident-level metrics (RQ3) |
| 7–8 | UNSW-NB15 generalisation; ablations |
| 9–10 | Write-up: method, results, threats to validity |
| 11–12 | Internal review, polish, submit |

## Writing checklist

- Every number in the paper comes from a file in `results/`.
- Every claim has a baseline or a citation.
- Report seeds, hardware and wall-clock time.
- State dataset label issues up front; reviewers will raise them.
