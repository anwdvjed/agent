import os

from ids.cli import main

HERE = os.path.dirname(os.path.abspath(__file__))


def test_cli_runs_synthetic(capsys):
    cfg = os.path.join(HERE, "..", "configs", "default.json")
    assert main(["run", "--source", "synthetic", "--config", cfg]) == 0
    out = capsys.readouterr().out
    assert "SCAN-001" in out
    assert "flows=" in out
