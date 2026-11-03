import os
import textwrap

from ids.detectors import AnomalyDetector, SignatureDetector
from ids.detectors.signature import Rule
from ids.evaluation import evaluate, load_cicids2017_csv
from ids.evaluation.tables import write_csv, write_latex

HEADER = "Source IP, Destination IP, Source Port, Destination Port, Protocol, Flow Duration, Total Fwd Packets, Total Backward Packets, Total Length of Fwd Packets, Total Length of Bwd Packets, SYN Flag Count, Label"


def _csv(tmp_path):
    rows = [HEADER]
    for i in range(50):
        rows.append(f"10.0.0.{i%9+2},10.0.0.1,40000,443,6,2000000,10,10,8000,9000,1,BENIGN")
    for p in range(1, 21):
        rows.append(f"192.168.1.66,10.0.0.1,40000,{p},6,1000,1,0,60,0,1,PortScan")
    path = tmp_path / "flows.csv"
    path.write_text("\n".join(rows) + "\n")
    return str(path)


def test_loader_maps_columns(tmp_path):
    flows = list(load_cicids2017_csv(_csv(tmp_path)))
    assert len(flows) == 70
    assert sum(f.is_attack for f in flows) == 20
    assert flows[-1].family == "PortScan"
    assert flows[0].flow.features()["duration"] == 2.0


def test_evaluate_reports_and_attribution(tmp_path):
    flows = list(load_cicids2017_csv(_csv(tmp_path)))
    benign = [f.flow for f in flows if not f.is_attack]
    anom = AnomalyDetector(threshold=3.0).fit(benign)
    rep = evaluate([anom], flows)
    assert rep.tp + rep.fn == 20 and rep.fp + rep.tn == 50
    assert rep.recall > 0.9
    assert "PortScan" in rep.per_family
    assert 0.0 <= rep.attribution_accuracy <= 1.0


def test_tables_written(tmp_path):
    flows = list(load_cicids2017_csv(_csv(tmp_path)))
    sig = SignatureDetector([Rule(id="S", description="scan", min_syn_ratio=0.9)])
    systems = {"signature-only": evaluate([sig], flows)}
    write_csv(str(tmp_path / "r.csv"), systems)
    write_latex(str(tmp_path / "r.tex"), systems, "cap", "tab:x")
    assert "signature-only" in (tmp_path / "r.csv").read_text()
    assert r"\begin{table}" in (tmp_path / "r.tex").read_text()
