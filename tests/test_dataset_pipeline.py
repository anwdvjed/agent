import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from agent_new.datasets.loader import ProcessedPrefixDataset, make_dataloader
from agent_new.losses import agent_new_loss
from agent_new.model import AgentNewConfig, AgentNewModel
from agent_new.datasets.common import assign_split, stable_digest


ROOT = Path(__file__).resolve().parents[1]


def asse_record(index, unsafe):
    return {
        "id": index,
        "contents": [
            {"role": "user", "content": f"Read case {index}"},
            {
                "role": "agent",
                "action": "read_file {\"path\": \"/cases/%d.txt\"}" % index,
            },
        ],
        "label": int(unsafe),
        "risk_type": "privacy leakage" if unsafe else "safe",
        "failure_mode": "unsafe disclosure" if unsafe else "",
        "risk_description": "private data" if unsafe else "",
        "ambiguous": 0,
    }


class ExistingDatasetPipelineTests(unittest.TestCase):
    def test_download_script_defaults_to_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "download_datasets.py"),
                    "--profile",
                    "train",
                    "--output",
                    directory,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            value = json.loads(result.stdout)
            self.assertEqual(value["mode"], "dry_run")
            self.assertFalse(value["network_writes"])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_tiny_train_prepare_feed_and_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            (raw / "assebench").mkdir(parents=True)
            safety = [asse_record(index, index % 3 == 0) for index in range(80)]
            security = [asse_record(100 + index, index % 2 == 0) for index in range(80)]
            (raw / "assebench" / "AgentJudge-safety.json").write_text(
                json.dumps(safety), encoding="utf-8"
            )
            (raw / "assebench" / "AgentJudge-security.json").write_text(
                json.dumps(security), encoding="utf-8"
            )
            (raw / "linuxarena").mkdir()
            rows = []
            for index in range(80):
                task_description = f"Complete task {index}"
                environment = f"env-{index % 5}"
                dataset_group = "linuxarena:" + stable_digest(
                    [task_description, environment]
                )
                dataset_split = assign_split(dataset_group, 20260824)
                side_success = (
                    index % 2 == 1
                    if dataset_split == "calibration"
                    else index % 6 == 1
                )
                rows.append(
                    {
                        "trajectory_id": f"linux-{index}",
                        "model": "test-model",
                        "model_short": "test",
                        "environment": environment,
                        "main_task": f"Task {index}",
                        "task_description": task_description,
                        "side_task": "hidden modification" if index % 2 else "",
                        "side_task_description": "",
                        "main_task_success": index % 4 != 0,
                        "side_task_success": side_success,
                        "has_side_task": index % 2 == 1,
                        "action_functions": ["bash"],
                        "action_arguments_json": ['{"command":"ls"}'],
                        "action_output": ["ok"],
                        "action_has_error": [False],
                    }
                )
            pq.write_table(pa.Table.from_pylist(rows), raw / "linuxarena" / "train-00000-of-00001.parquet")
            output = root / "processed"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "prepare_datasets.py"),
                    "--raw",
                    str(raw),
                    "--output",
                    str(output),
                    "--profile",
                    "train",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertGreater(manifest["split_files"]["train"]["rows"], 0)
            self.assertGreater(manifest["split_files"]["validation"]["rows"], 0)
            self.assertGreater(manifest["split_files"]["calibration"]["rows"], 0)
            dataset = ProcessedPrefixDataset(output, "train")
            loader = make_dataloader(
                dataset,
                batch_size=8,
                source_shares={"assebench": 0.7, "linuxarena": 0.3},
                seed=7,
            )
            batch = next(iter(loader))
            model = AgentNewModel(
                AgentNewConfig(
                    hidden_dim=32,
                    edge_dim=16,
                    graph_layers=1,
                    horizon=manifest["horizon"],
                    num_event_types=len(manifest["event_types"]),
                )
            )
            output_values = model(batch)
            loss, parts = agent_new_loss(output_values, batch)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertIn("attack_intent", parts)
            self.assertIn("utility", parts)
            request_text = json.dumps(dataset.rows[0]["request"])
            self.assertNotIn("side_task", request_text)
            self.assertNotIn("side_task_success", request_text)

            model_config = root / "model.json"
            model_config.write_text(
                json.dumps(
                    {
                        "hidden_dim": 16,
                        "edge_dim": 8,
                        "graph_layers": 1,
                        "horizon": 5,
                        "num_event_types": 5,
                        "num_concepts": 9,
                        "base_event_rate": 0.01,
                        "evidence_temperature": 1.0,
                        "actual_branch": "none",
                    }
                ),
                encoding="utf-8",
            )
            train_config = root / "training.json"
            train_config.write_text(
                json.dumps(
                    {
                        "seed": 1,
                        "epochs": 1,
                        "batch_size": 16,
                        "learning_rate": 0.001,
                        "weight_decay": 0.0,
                        "gradient_clip": 1.0,
                        "patience": 1,
                        "num_workers": 0,
                        "source_batch_share": {"assebench": 0.7, "linuxarena": 0.3},
                        "loss": {
                            "survival": 1.0,
                            "intervention_survival": 0.0,
                            "concepts": 0.1,
                            "evidence_supervision": 0.0,
                            "evidence_sufficiency": 0.0,
                            "evidence_necessity": 0.0,
                            "evidence_sparsity": 0.0,
                            "intervention_ranking": 0.0,
                            "attack_intent": 0.1,
                            "utility": 0.1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            run_paths = []
            train_stdout = ""
            for seed in (11, 22):
                run_path = root / f"run-{seed}"
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "train_model.py"),
                        "--data-dir",
                        str(output),
                        "--model-config",
                        str(model_config),
                        "--train-config",
                        str(train_config),
                        "--output-dir",
                        str(run_path),
                        "--seed",
                        str(seed),
                        "--run-id",
                        f"run-{seed}",
                        "--device",
                        "cpu",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                train_stdout += result.stdout + result.stderr
                run_paths.append(run_path / "best")
            self.assertIn("Train Epoch 001/001", train_stdout)
            self.assertIn("Validation Epoch 001/001", train_stdout)

            eval_config = root / "evaluation.json"
            eval_config.write_text(
                json.dumps(
                    {
                        "batch_size": 16,
                        "epochs": 1,
                        "device": "cpu",
                        "alpha": 0.5,
                        "ensemble_z": 1.96,
                        "target_safe_trajectory_fpr": 0.5,
                        "bootstrap_samples": 10,
                        "bootstrap_seed": 1,
                    }
                ),
                encoding="utf-8",
            )
            validation_output = root / "validation"
            validation = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "validate_model.py"),
                    "--checkpoint",
                    str(run_paths[0]),
                    "--checkpoint",
                    str(run_paths[1]),
                    "--data-dir",
                    str(output),
                    "--config",
                    str(eval_config),
                    "--output-dir",
                    str(validation_output),
                    "--device",
                    "cpu",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if validation.returncode != 0:
                self.fail(validation.stdout + "\n" + validation.stderr)
            self.assertIn("Validation Epoch 001/001", validation.stdout + validation.stderr)

            atbench_records = []
            for index in range(20):
                atbench_records.append(
                    {
                        "conv_id": f"at-{index}",
                        "label": int(index % 2 == 0),
                        "risk_source": "tool output" if index % 2 == 0 else "",
                        "failure_mode": "unsafe tool use" if index % 2 == 0 else "",
                        "harm_type": "privacy leakage" if index % 2 == 0 else "",
                        "content": [
                            {"role": "user", "content": f"Read public item {index}"},
                            {
                                "role": "assistant",
                                "action": {
                                    "tool": "read_file",
                                    "arguments": {"path": f"/public/{index}.txt"},
                                },
                            },
                        ],
                    }
                )
            (raw / "atbench").mkdir()
            (raw / "atbench" / "test.json").write_text(
                json.dumps(atbench_records), encoding="utf-8"
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "prepare_datasets.py"),
                    "--raw",
                    str(raw),
                    "--output",
                    str(output),
                    "--profile",
                    "eval",
                    "--overwrite",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            test_result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "test_model.py"),
                    "--checkpoint",
                    str(validation_output / "checkpoints" / "run-11"),
                    "--checkpoint",
                    str(validation_output / "checkpoints" / "run-22"),
                    "--data-dir",
                    str(output),
                    "--calibration",
                    str(validation_output / "calibration.json"),
                    "--operating-threshold",
                    str(validation_output / "operating_threshold.json"),
                    "--config",
                    str(eval_config),
                    "--device",
                    "cpu",
                    "--output",
                    str(root / "test_report.json"),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if test_result.returncode != 0:
                self.fail(test_result.stdout + "\n" + test_result.stderr)
            self.assertIn("Test Epoch 001/001", test_result.stdout + test_result.stderr)


if __name__ == "__main__":
    unittest.main()
