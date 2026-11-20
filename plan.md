# Execution Plan — NL2SQL Semantic Robustness

> This file is source of truth. Any deviation updates this file first, then executes.
> Status: 🔲 pending | 🔄 in progress | ✅ done | ⚠️ blocked (needs human) | ❌ failed

---

## Phase 0 — Remote Setup

**Goal:** Get project running on GPU SSH machine `cse@10.50.35.50`.

| # | Step | How | Status |
|---|------|-----|--------|
| 0.1 | Inspect remote env (Python, CUDA, disk, existing data/models) | SSH, `python3 --version`, `nvidia-smi`, `df -h`, `ls` | 🔲 |
| 0.2 | Zip project locally and scp to remote | `zip -r PROJECT_NLP.zip PROJECT_NLP/`, `scp` to `~/NLP_PROJECT_TEAM_VERMA/` | 🔲 |
| 0.3 | Extract zip on remote | `unzip PROJECT_NLP.zip` | 🔲 |
| 0.4 | Create/activate venv, install deps | `python3 -m venv .venv && pip install -r requirements.txt` (use existing torch if CUDA mismatch) | 🔲 |
| 0.5 | Verify Spider data layout and T5-small model files exist | `ls data/spider/` and `ls models/t5-small/` | 🔲 |
| 0.6 | Fill `configs/experiment.yaml` — dataset_version, license, model revision | Edit FILL_IN fields | 🔲 |
| 0.7 | Smoke test: `python -m nl2sql --help` | Should print CLI help | 🔲 |

---

## Phase 1 — Literature Gate (⚠️ Human required)

**Goal:** Approve `docs/novelty_review.json` before any final training.

| # | Step | How | Status |
|---|------|-----|--------|
| 1.1 | Fill `docs/novelty_review.json` with real literature evidence | Manual: read related work in `docs/references.md`, fill gap statement | ⚠️ |

> NOTE: Pilot train (Phase 2) can run before this gate. Final training (Phase 4+) needs it approved.

---

## Phase 2 — Data Preparation + Baseline Pilot

**Goal:** Freeze splits and confirm T5-small trains on Spider.

| # | Step | How | Status |
|---|------|-----|--------|
| 2.1 | Prepare frozen splits | `python -m nl2sql prepare` | 🔲 |
| 2.2 | Inspect `artifacts/prepared/` — counts, audit.json, quarantine | `cat artifacts/prepared/audit.json` | 🔲 |
| 2.3 | Pilot train A0 seed 13 | `python -m nl2sql train --condition A0 --seed 13 --pilot` | 🔲 |
| 2.4 | Pilot predict on validation | `python -m nl2sql predict --checkpoint runs/pilot/seed_13/model --dataset artifacts/prepared/validation.jsonl --output runs/pilot/seed_13/validation_predictions.jsonl --condition A0 --seed 13` | 🔲 |
| 2.5 | Pilot evaluate | `python -m nl2sql evaluate --dataset artifacts/prepared/validation.jsonl --predictions runs/pilot/seed_13/validation_predictions.jsonl --output runs/pilot/seed_13/validation_evaluation` | 🔲 |
| 2.6 | Inspect pilot metrics — exec accuracy, SQL validity, learning curve | `cat runs/pilot/seed_13/validation_evaluation/metrics.json` | 🔲 |

---

## Phase 3 — Baseline Review Gate (⚠️ Human required)

| # | Step | How | Status |
|---|------|-----|--------|
| 3.1 | Fill `docs/baseline_review.json` — set `baseline_is_suitable: true`, `pilot_run`, accuracy numbers | Manual review of pilot metrics from 2.6 | ⚠️ |

---

## Phase 4 — Augmentation Proposal + Human Annotation

**Goal:** Generate 1000 accepted paraphrases + 1000 accepted counterfactuals.

| # | Step | How | Status |
|---|------|-----|--------|
| 4.1 | Generate candidates | `python -m nl2sql propose` | 🔲 |
| 4.2 | Copy candidates to reviewed file | `cp artifacts/augmentation/candidates.jsonl artifacts/augmentation/reviewed.jsonl` | 🔲 |
| 4.3 | ⚠️ Manually annotate reviewed.jsonl | Follow `docs/annotation_guide.md` — write questions, set review booleans, accept/reject | ⚠️ |
| 4.4 | Validate augmentation | `python -m nl2sql validate-augmentation` | 🔲 |
| 4.5 | Check counts — need 1000 accepted paraphrases + 1000 counterfactuals | `cat artifacts/augmentation/validation_manifest.json` | 🔲 |

---

## Phase 5 — Augmentation Pilot Review Gate (⚠️ Human required)

| # | Step | How | Status |
|---|------|-----|--------|
| 5.1 | Fill `docs/augmentation_pilot_review.json` — set positive `pilot_examples_reviewed` count | Manual | ⚠️ |

---

## Phase 6 — Build Conditions + Diagnostics

| # | Step | How | Status |
|---|------|-----|--------|
| 6.1 | Build all 5 training condition files | `python -m nl2sql build-conditions` | 🔲 |
| 6.2 | Build diagnostic set | `python -m nl2sql build-diagnostics` | 🔲 |
| 6.3 | ⚠️ Review diagnostics candidates | `cp artifacts/diagnostics/candidates.jsonl artifacts/diagnostics/reviewed.jsonl` then manual review per `docs/annotation_guide.md` | ⚠️ |
| 6.4 | Freeze diagnostics | `python -m nl2sql freeze-diagnostics --reviewed artifacts/diagnostics/reviewed.jsonl` | 🔲 |

---

## Phase 7 — Final Freeze (⚠️ Human required)

| # | Step | How | Status |
|---|------|-----|--------|
| 7.1 | Fill `docs/final_evaluation_review.json` | Set both booleans true, confirm planned seeds/conditions | ⚠️ |
| 7.2 | Freeze protocol | `python -m nl2sql freeze --datasets artifacts/prepared/test.jsonl artifacts/diagnostics/test.jsonl` | 🔲 |

---

## Phase 8 — Full Training (all conditions × 3 seeds)

**~15 runs total. GPU required. Run via script.**

| # | Step | How | Status |
|---|------|-----|--------|
| 8.1 | Run all experiments | `bash scripts/run_experiments.sh configs/experiment.yaml` | 🔲 |
| 8.2 | Monitor — check `runs/` for complete status in each `run.json` | `grep -r '"status"' runs/*/seed_*/run.json` | 🔲 |

---

## Phase 9 — Evaluation

| # | Step | How | Status |
|---|------|-----|--------|
| 9.1 | Decode + evaluate all frozen datasets | `bash scripts/evaluate_all.sh configs/experiment.yaml` | 🔲 |
| 9.2 | Summarize across seeds/conditions | `python -m nl2sql summarize --evaluations runs/A*/seed_*/evaluation/dataset_0 --output runs/summary/spider` | 🔲 |
| 9.3 | (Optional) Run official Spider evaluator if checkout available | `python -m nl2sql official-evaluate --kind spider ...` | 🔲 |

---

## Phase 10 — Report

| # | Step | How | Status |
|---|------|-----|--------|
| 10.1 | Fill `docs/report_template.md` with actual numbers | Manual — populate results table, error analysis, discussion | ⚠️ |

---

## Blockers / Notes

- **Spider data** must be at `data/spider/` on remote. If absent → cannot proceed past 2.1.
- **T5-small model** must be at `models/t5-small/` on remote (all tokenizer + weight files). If absent → cannot proceed past 2.3.
- **Annotation (4.3, 6.3)** is fully manual — no LLM can safely auto-approve without reading each SQL+question pair.
- **Gate files (1.1, 3.1, 5.1, 7.1)** require a human reviewer identity — pipeline enforces this.
- Pilot (Phase 2) intentionally runs before novelty gate to allow feasibility debugging.
- Do NOT run final training until all gates approved.
