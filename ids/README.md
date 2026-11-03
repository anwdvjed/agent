# IDS — Intrusion Detection System design project

A self-contained scaffold for designing and prototyping a network intrusion
detection system (NIDS). It lives in this repository as a top-level folder so
it can be split into its own repository later without changes.

The design combines two complementary detection paths:

- **Signature detection**: rule-based matching against known-bad patterns
  (ports, payload substrings, IP reputation). Fast, precise, low recall on
  novel attacks.
- **Anomaly detection**: statistical / ML models over per-flow features that
  flag behaviour deviating from a learned baseline. Higher recall on novel
  attacks, more false positives.

Alerts from both paths flow through a common alert manager that de-duplicates,
scores and dispatches them.

## Architecture

```text
   packet source (pcap file | live interface | synthetic generator)
                               |
                     capture.PacketSource
                               |
                     features.FlowAssembler
          (5-tuple flows, counters, timing, size statistics)
                               |
            +------------------+------------------+
            |                                     |
   detectors.SignatureDetector         detectors.AnomalyDetector
   (rules/*.yaml)                      (baseline + z-score / Isolation Forest)
            |                                     |
            +------------------+------------------+
                               |
                     alerting.AlertManager
                 (dedupe, severity, sinks: stdout / JSONL)
```

## Layout

```text
ids/
  ids/
    capture/     packet sources (pcap, live, synthetic)
    features/    flow assembly and feature extraction
    detectors/   signature and anomaly detectors
    alerting/    alert model, manager and sinks
    cli.py       `ids run` entry point
  configs/       runtime configuration (JSON)
  rules/         signature rules (YAML)
  docs/          design notes and decision records
  scripts/       helper scripts (dataset download, baseline training)
  tests/         pytest suite
```

## Quick start

```bash
cd ids
pip install -e ".[dev]"
pytest
ids run --source synthetic --config configs/default.json
```

## Datasets to evaluate against

- CIC-IDS2017 / CSE-CIC-IDS2018 (labelled flows, modern attacks)
- UNSW-NB15
- NSL-KDD (legacy, small, useful for quick experiments)

## Toward a paper

- `docs/RESEARCH_PLAN.md`: research question, hypotheses, candidate
  contributions, baselines, metrics, protocol and a 12-week timeline.
- `ids/evaluation/`: dataset loader (CIC-IDS2017 CSVs), flow-level and
  attribution metrics, and a runner that writes `results/*.csv` and
  `paper/tables/*.tex`.
- `paper/`: IEEE-format LaTeX skeleton with section-by-section TODOs and a
  starter bibliography. Build with `make -C paper`.

```bash
python -m ids.evaluation.run --dataset cicids2017 --config configs/eval.json
make -C paper
```

See `docs/DESIGN.md` for goals, threat model and open decisions.
