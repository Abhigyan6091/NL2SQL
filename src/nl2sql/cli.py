"""Explicit subcommands only: importing this package never starts a pipeline."""
from __future__ import annotations

import argparse
import sqlite3
import sys
import yaml

from .common import config, offline, planned_conditions


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Offline Text-to-SQL semantic robustness experiments")
    root.add_argument("--config", default="configs/experiment.yaml", help="Paths resolve relative to the working directory")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("run-experiments", help="Run planned conditions/seeds, verifying any completed outputs")
    commands.add_parser("evaluate-all", help="Evaluate every frozen dataset with verified artifact reuse")
    commands.add_parser("prepare", help="Read local Spider files and freeze database-disjoint splits")
    commands.add_parser("propose", help="Create pending training augmentation annotation candidates")
    commands.add_parser("validate-augmentation", help="Validate reviewed training candidates and log all rejections")
    commands.add_parser("build-conditions", help="Freeze equal-budget A0–A3 and A3_matched files after review gates")
    commands.add_parser("build-diagnostics", help="Create controlled fixtures and pending annotations")
    p = commands.add_parser("freeze-diagnostics", help="Validate manually reviewed controlled test families")
    p.add_argument("--reviewed", required=True)
    p = commands.add_parser("freeze", help="Freeze final protocol and every final evaluation dataset")
    p.add_argument("--datasets", nargs="+", required=True)
    p = commands.add_parser("train", help="Fine-tune one condition and seed from local model files")
    p.add_argument("--condition", choices=["A0", "A1", "A2", "A3", "A3_matched"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--pilot", action="store_true")
    p = commands.add_parser("predict", help="Decode one frozen dataset")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--condition", choices=["A0", "A1", "A2", "A3", "A3_matched"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--final", action="store_true")
    p = commands.add_parser("evaluate", help="Archive execution and paired results using strict SQLite policy")
    p.add_argument("--dataset", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("official-evaluate", help="Invoke a locally supplied official Spider/test-suite evaluator")
    for flag in ("dataset", "predictions", "script", "database-root", "tables", "output"):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--kind", choices=["spider", "test-suite"], required=True)
    p = commands.add_parser("import-benchmark", help="Normalize local NLQ perturbations or optional composition examples")
    for flag in ("source", "tables", "output", "name", "benchmark-version", "license"):
        p.add_argument("--" + flag, required=True)
    p = commands.add_parser("summarize", help="Require all main conditions/seeds and calculate uncertainty")
    p.add_argument("--evaluations", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=2026)
    return root


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    offline()
    try:
        cfg = config(args.config)
        if args.command == "run-experiments":
            from .orchestrate import run_experiments
            run_experiments(cfg, args.config)
        elif args.command == "evaluate-all":
            from .orchestrate import evaluate_all
            evaluate_all(cfg, args.config)
        elif args.command == "prepare":
            from .data import prepare
            prepare(cfg)
        elif args.command in {"propose", "validate-augmentation", "build-conditions"}:
            from .augment import build_conditions, propose, validate_augmentation
            {"propose": propose, "validate-augmentation": validate_augmentation, "build-conditions": build_conditions}[args.command](cfg)
        elif args.command == "build-diagnostics":
            from .diagnostics import build_diagnostics
            build_diagnostics(cfg)
        elif args.command == "freeze-diagnostics":
            from .diagnostics import freeze_diagnostics
            freeze_diagnostics(cfg, args.reviewed)
        elif args.command == "freeze":
            from .freeze import freeze_experiment
            freeze_experiment(cfg, args.datasets)
        elif args.command == "train":
            from .model import train
            train(cfg, args.condition, args.seed, args.pilot)
        elif args.command == "predict":
            from .model import predict
            predict(cfg, args.checkpoint, args.dataset, args.output, args.condition, args.seed, args.final, args.batch_size)
        elif args.command == "evaluate":
            from .evaluate import evaluate
            evaluate(cfg, args.dataset, args.predictions, args.output)
        elif args.command == "official-evaluate":
            from .evaluate import official_evaluate
            official_evaluate(args.dataset, args.predictions, args.script, args.database_root, args.tables, args.output, args.kind, cfg)
        elif args.command == "import-benchmark":
            from .external import import_benchmark
            import_benchmark(cfg, args.source, args.tables, args.output, args.name, args.benchmark_version, args.license)
        elif args.command == "summarize":
            from .analysis import summarize
            summarize(args.evaluations, args.output, cfg["training"]["seeds"], args.bootstrap_samples, args.bootstrap_seed, planned_conditions(cfg))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, sqlite3.Error, yaml.YAMLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(f"Completed: {args.command}")


if __name__ == "__main__":
    main()
