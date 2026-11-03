from ids.alerting import AlertManager
from ids.capture import SyntheticSource
from ids.detectors import AnomalyDetector, SignatureDetector
from ids.detectors.signature import Rule
from ids.features import FlowAssembler


def _flows(inject_scan=True):
    return list(FlowAssembler().feed(SyntheticSource(n_packets=300, inject_scan=inject_scan)))


def test_flow_assembly_groups_by_5tuple():
    flows = _flows(inject_scan=False)
    assert flows
    assert all(f.packets >= 1 for f in flows)
    assert len({f.key for f in flows}) == len(flows)


def test_signature_detector_flags_syn_scan():
    det = SignatureDetector([Rule(id="SCAN", description="syn", severity="high", min_syn_ratio=0.9)])
    alerts = det.inspect_many(_flows())
    assert alerts
    assert all(a.flow_key[0] == "192.168.1.66" for a in alerts)


def test_signature_detector_quiet_on_benign():
    det = SignatureDetector([Rule(id="SCAN", description="syn", min_syn_ratio=0.9)])
    assert det.inspect_many(_flows(inject_scan=False)) == []


def test_anomaly_detector_requires_fit():
    import pytest

    with pytest.raises(RuntimeError):
        AnomalyDetector().inspect(_flows()[0])


def test_anomaly_detector_scores_flows():
    flows = _flows()
    det = AnomalyDetector(threshold=4.0).fit(flows)
    for f in flows:
        assert det.score(f).shape == (6,)


def test_alert_manager_dedupes_scan_burst():
    det = SignatureDetector([Rule(id="SCAN", description="syn", severity="high", min_syn_ratio=0.9)])
    alerts = det.inspect_many(_flows())
    manager = AlertManager(sinks=[], dedupe_window=60.0)
    manager.handle(alerts)
    assert len(manager.emitted) == 1
    assert manager.suppressed == len(alerts) - 1
