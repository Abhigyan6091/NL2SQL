"""Sequential subprocess orchestration with validation before reusing artifacts."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .common import digest, file_hash, fingerprint_tree, planned_conditions, read_json
from .freeze import verify_final
from .model import verify_run
from .provenance import verify_evaluation_archive, verify_predictions


def _call(config_path: str, *arguments: str) -> None:
    command = [sys.executable, "-m", "nl2sql", "--config", config_path, *arguments]
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise RuntimeError(f"Stage failed with exit code {completed.returncode}: {' '.join(arguments)}")


def run_experiments(cfg: dict, config_path: str) -> None:
    frozen = verify_final(cfg)
    if fingerprint_tree(cfg["model"]["path"])["sha256"] != frozen["pretrained_model"]["sha256"]:
        raise ValueError("Pretrained model changed since final freeze")
    for condition in planned_conditions(cfg):
        for seed in cfg["training"]["seeds"]:
            root = Path(cfg["paths"]["runs"]) / condition / f"seed_{seed}"
            if root.exists():
                verify_run(cfg, str(root / "model"), condition, seed)
                print(f"Keeping verified completed run: {root}", flush=True)
            else:
                _call(config_path, "train", "--condition", condition, "--seed", str(seed))


def evaluate_all(cfg: dict, config_path: str) -> None:
    frozen = verify_final(cfg)
    for index, dataset in enumerate(frozen["datasets"]):
        evaluations = []
        for condition in planned_conditions(cfg):
            for seed in cfg["training"]["seeds"]:
                root = Path(cfg["paths"]["runs"]) / condition / f"seed_{seed}"
                _, checkpoint = verify_run(cfg, str(root / "model"), condition, seed)
                predictions = root / "predictions" / f"dataset_{index}.jsonl"
                evaluation = root / "evaluation" / f"dataset_{index}"
                if predictions.exists():
                    _, _, manifest = verify_predictions(dataset, str(predictions), cfg)
                    if manifest["checkpoint"]["sha256"] != checkpoint["sha256"] or manifest["condition"] != condition or manifest["seed"] != seed:
                        raise ValueError(f"Existing predictions belong to a different run: {predictions}")
                else:
                    _call(config_path, "predict", "--checkpoint", str(root / "model"), "--dataset", dataset,
                          "--output", str(predictions), "--condition", condition, "--seed", str(seed), "--final")
                if evaluation.exists():
                    metrics = verify_evaluation_archive(evaluation)
                    if (metrics["predictions_sha256"] != file_hash(predictions)
                            or metrics["dataset_sha256"] != file_hash(dataset)
                            or metrics["config_sha256"] != digest(cfg)
                            or metrics["condition"] != condition or metrics["seed"] != seed):
                        raise ValueError(f"Existing evaluation belongs to a different run: {evaluation}")
                else:
                    _call(config_path, "evaluate", "--dataset", dataset, "--predictions", str(predictions), "--output", str(evaluation))
                evaluations.append(str(evaluation))
        summary = Path(cfg["paths"]["runs"]) / "summary" / f"dataset_{index}"
        if summary.exists():
            completion = read_json(summary / "completion.json")
            expected = {str(Path(p).resolve()): file_hash(Path(p) / "completion.json") for p in evaluations}
            if (completion["evaluations"] != expected or completion["expected_seeds"] != cfg["training"]["seeds"]
                    or completion["expected_conditions"] != planned_conditions(cfg)
                    or completion["bootstrap_samples"] != 2000 or completion["bootstrap_seed"] != 2026):
                raise ValueError(f"Existing summary belongs to a different analysis: {summary}")
            for name, expected_hash in completion["files"].items():
                if file_hash(summary / name) != expected_hash:
                    raise ValueError(f"Existing summary modified: {summary / name}")
        else:
            _call(config_path, "summarize", "--evaluations", *evaluations, "--output", str(summary))
