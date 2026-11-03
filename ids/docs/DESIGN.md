# Design notes

## Goals

1. Detect a representative set of network attacks (port scans, floods,
   brute-force logins, known malicious payloads) on a single monitored link.
2. Keep false positives low enough that every alert is worth a human look.
3. Run from pcap files for offline evaluation and from a live interface later.
4. Be measurable: every detector reports precision / recall on labelled data.

## Non-goals (for now)

- Inline prevention (IPS). The system is passive.
- Encrypted payload inspection.
- Distributed / multi-sensor deployment.

## Threat model

An external attacker on the untrusted side of the monitored link who may
scan, flood, or attempt credential attacks against hosts on the trusted side.
The sensor itself is assumed uncompromised.

## Pipeline decisions

| Decision | Choice | Why |
|---|---|---|
| Unit of analysis | bidirectional 5-tuple flow | matches CIC-IDS feature format, keeps per-packet cost low |
| Flow expiry | idle timeout (default 60 s) | simple, deterministic, good enough for offline evaluation |
| Signature engine | declarative YAML rules | easy to read and diff; Suricata-style rule import can come later |
| Anomaly baseline | log1p z-score per feature | zero extra dependencies, explainable alerts; Isolation Forest optional |
| Alert dedupe | (detector, rule, src, dst, proto) within window | collapses scan bursts into one alert |

## Open questions

- Which labelled dataset is the primary benchmark? CIC-IDS2017 is the default
  candidate; NSL-KDD for fast iteration.
- Should flows be split by active timeout as well as idle timeout?
- How do we bound memory for the active-flow table on long captures?
- Does the anomaly detector need per-service baselines (port 53 vs 443)?

## Roadmap

- [x] Scaffold: capture, flows, signature + anomaly detectors, alert manager
- [x] CIC-IDS2017 loader and evaluation harness (precision / recall per attack)
- [ ] Run harness on real CIC-IDS2017 data; first results table
- [ ] Isolation Forest detector behind the same interface
- [ ] Payload-pattern rules (byte signatures)
- [ ] Live capture via scapy / pyshark
- [ ] Alert sink for a SIEM (syslog / webhook)
