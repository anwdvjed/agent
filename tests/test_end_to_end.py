from agent1.calibration import fit_policy_calibration
from agent1.dataset import prepare_dataset
from agent1.engine import train, evaluate
from agent1.environments import generate_demo, ENVIRONMENT_ID
from agent1.evidence import explain_groups
from agent1.io import read_jsonl
from agent1.replay import replay_split
from agent1.runtime import Ensemble
from agent1.checkpoint import canonical_digest
from test_policy_v2 import policy as v2_policy


def test_train_validate_calibrate_test_and_explain(tmp_path, registry):
    raw, data = tmp_path / "raw", tmp_path / "data"
    generate_demo(raw, registry, groups=16)
    prepare_dataset(raw / "episodes.jsonl", raw / "registry.json", data)
    checkpoints = []
    for seed in (11, 22):
        report = train(data, tmp_path / str(seed), seed=seed,
                       config={"epochs": 1, "batch_size": 16},
                       model_config={"hidden_dim": 12, "edge_dim": 4, "graph_layers": 1})
        assert report["epochs_completed"] == 1
        checkpoints.append(report["checkpoint"])
    ensemble = Ensemble(checkpoints, registry)
    assert ensemble.independent
    report = evaluate(ensemble, data, "test", tmp_path / "test.json")
    assert report["split"] == "test" and report["synthetic"]
    assert report["metrics"]["probability_by_branch"]["none"]["n"] > 0
    policy = v2_policy()
    rows = replay_split(raw / "scenarios.jsonl", ensemble, [policy], "calibration")
    lineage = {**ensemble.lineage(ENVIRONMENT_ID),
               "repair_provider_version": ENVIRONMENT_ID,
               "executor_contract_version": ENVIRONMENT_ID,
               "policy_family_digest": canonical_digest([policy]),
               "calibration_manifest_digest": canonical_digest(sorted({(r["episode_id"], r["group_id"]) for r in rows})),
               "development_groups": sorted(ensemble.development_groups)}
    artifact = fit_policy_calibration(rows, [policy], lineage)
    assert artifact["status"] == "no_feasible_policy"  # n is intentionally insufficient
    request = next(r["request"] for r in read_jsonl(data / "test.jsonl") if r["tracking"]["step"] == 2)
    result = explain_groups(request, ensemble, budget=3)
    assert result["groups"]
    assert result["evidence_path_verified"] is False
    assert all(r["replay_outcome"] is None for r in result["groups"])
