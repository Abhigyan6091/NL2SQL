"""Complete-condition summaries, seed variability and paired family bootstrap."""
from __future__ import annotations

import random
import statistics
from collections import defaultdict
from pathlib import Path

from .common import digest, file_hash, read_jsonl, write_json
from .provenance import verify_evaluation_archive


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(evaluation_dirs: list[str], output: str, expected_seeds: list[int],
              bootstrap_samples: int = 2000, seed: int = 2026,
              expected_conditions: list[str] | None = None) -> None:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not evaluation_dirs or not expected_seeds or len(expected_seeds) != len(set(expected_seeds)):
        raise ValueError("Supply evaluation directories and a nonempty list of unique planned seeds")
    groups, runs = defaultdict(list), {}
    for path in evaluation_dirs:
        root = Path(path)
        metrics = verify_evaluation_archive(root)
        key = (metrics["dataset_sha256"], metrics["condition"], metrics["seed"])
        if key in runs:
            raise ValueError(f"Duplicate evaluation run: {key}")
        runs[key] = (root, metrics)
        groups[(key[0], key[1])].append(metrics)
    datasets = sorted({key[0] for key in runs})
    if len({m["config_sha256"] for _, m in runs.values()}) != 1:
        raise ValueError("Cannot compare evaluations from different configurations")
    for dataset in datasets:
        policies = {digest(m["policy"]) for (data_hash, _, _), (_, m) in runs.items() if data_hash == dataset}
        if len(policies) != 1:
            raise ValueError("Cannot compare conditions with different execution policies on the same dataset")
    summary, deltas = [], []
    for dataset in datasets:
        planned = list(expected_conditions or ["A0", "A1", "A2", "A3"])
        if expected_conditions is None and any(k[:2] == (dataset, "A3_matched") for k in runs):
            planned.append("A3_matched")
        if {k[1] for k in runs if k[0] == dataset} != set(planned):
            raise ValueError("Evaluation conditions do not match the complete planned matrix")
        for condition in planned:
            actual = {k[2] for k in runs if k[:2] == (dataset, condition)}
            if actual != set(expected_seeds):
                raise ValueError(f"Report must contain all A0–A3 and planned seeds: {condition}: {sorted(actual)}, expected {expected_seeds}")
    for (dataset, condition), metrics in sorted(groups.items()):
        values = {}
        for metric in metrics[0]["metrics"]:
            observations = [m["metrics"][metric]["value"] for m in metrics]
            if any(v is None for v in observations):
                values[metric] = {"mean": None, "std": None, "n_seeds": len(observations)}
            else:
                denominators = {m["metrics"][metric]["denominator"] for m in metrics}
                if len(denominators) != 1:
                    raise ValueError("Metric denominator changed between seeds")
                values[metric] = {"mean": statistics.mean(observations),
                                  "std": statistics.stdev(observations) if len(observations) > 1 else None,
                                  "n_seeds": len(observations), "denominator_per_seed": denominators.pop()}
        summary.append({"dataset_sha256": dataset, "condition": condition, "metrics": values,
                        "per_seed_per_phenomenon": [{"seed": m["seed"], "metrics": m["per_phenomenon"]} for m in metrics]})
    rng = random.Random(seed)
    for dataset in datasets:
        conditions = sorted({key[1] for key in runs if key[0] == dataset and key[1] != "A0"})
        for condition in conditions:
            seeds = sorted({k[2] for k in runs if k[:2] == (dataset, condition)})
            for metric in ("strict_suite_accuracy", "paraphrase_family_correctness", "paraphrase_pair_consistency", "counterfactual_contrast_accuracy"):
                observations = []
                for run_seed in seeds:
                    if (dataset, "A0", run_seed) not in runs:
                        raise ValueError("Missing matched baseline seed")
                    def family_values(cond):
                        root, _ = runs[(dataset, cond, run_seed)]
                        if metric == "strict_suite_accuracy":
                            values = defaultdict(list)
                            for row in read_jsonl(root / "items.jsonl"):
                                values[row["family_id"]].append(float(row["suite_correct"]))
                            return {family: statistics.mean(v) for family, v in values.items()}
                        field = {"paraphrase_family_correctness": "paraphrase_family_correct",
                                 "paraphrase_pair_consistency": "paraphrase_consistency",
                                 "counterfactual_contrast_accuracy": "counterfactual_contrast"}[metric]
                        result = {}
                        for row in read_jsonl(root / "pairs.jsonl"):
                            val = row[field]
                            if isinstance(val, list):
                                if val:
                                    result[row["family_id"]] = statistics.mean(val)
                            elif val is not None:
                                result[row["family_id"]] = float(val)
                        return result
                    baseline, treatment = family_values("A0"), family_values(condition)
                    if set(baseline) != set(treatment):
                        raise ValueError("Paired family membership differs between conditions")
                    observations.append({k: treatment[k] - baseline[k] for k in baseline})
                if not observations or not observations[0]:
                    continue
                families = sorted(observations[0])
                if any(set(o) != set(families) for o in observations):
                    raise ValueError("Family membership differs across seeds")
                # Crossed resampling: same family sample for all sampled matched seeds.
                draws = []
                for _ in range(bootstrap_samples):
                    sampled_families = rng.choices(families, k=len(families))
                    sampled_seeds = rng.choices(range(len(observations)), k=len(observations))
                    draws.append(statistics.mean(observations[s][f] for s in sampled_seeds for f in sampled_families))
                deltas.append({"dataset_sha256": dataset, "condition": condition, "metric": metric,
                               "delta_from_A0": statistics.mean(v for obs in observations for v in obs.values()),
                               "ci95": [_percentile(draws, .025), _percentile(draws, .975)],
                               "families": len(families), "seeds": len(seeds),
                               "weighting": "equal family then equal matched seed"})
    dest = Path(output)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(dest)
    write_json(dest / "summary.json", {"conditions": summary, "paired_deltas": deltas,
        "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed,
        "limitations": "Intervals are exploratory and unadjusted for multiple comparisons. One-seed runs cannot estimate training variability. Controlled threshold families share templates."})
    lines = ["# Experiment results", "", "All values below are computed from the supplied evaluation archives.", "",
             "Strict SQLite metrics are not official Spider or official test-suite scores.", "",
             "| Dataset hash | Condition | Seeds | Strict suite accuracy mean | Seed SD |", "|---|---|---:|---:|---:|"]
    for row in summary:
        metric = row["metrics"]["strict_suite_accuracy"]
        sd = f"{metric['std']:.4f}" if metric["std"] is not None else "unavailable"
        lines.append(f"| {row['dataset_sha256'][:12]} | {row['condition']} | {metric['n_seeds']} | {metric['mean']:.4f} | {sd} |")
    lines.extend(["", "Per-phenomenon denominators, paired metrics, and bootstrap intervals are in summary.json.", "",
                  "Interpret gains only after inspecting paired measures, data quality, and transfer to held-out benchmarks. Report negative and mixed results."])
    (dest / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(dest / "completion.json", {"evaluations": {str(Path(p).resolve()): file_hash(Path(p) / "completion.json") for p in evaluation_dirs},
        "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed,
        "expected_seeds": expected_seeds, "expected_conditions": expected_conditions,
        "files": {name: file_hash(dest / name) for name in ("summary.json", "results.md")}})
