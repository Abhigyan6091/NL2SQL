# Semantic robustness in Text to SQL

This repository implements the research pipeline described in
`NL2SQL_Semantic_Robustness_Research_Proposal.docx`. It compares ordinary T5
fine-tuning (A0), paraphrase augmentation (A1), counterfactual augmentation (A2),
and both (A3). A3_matched adds the same total augmentation count as A1/A2.

The code was written without running experiments, tests, package installation,
or downloads. Runtime compatibility and experiment outcomes have not been
verified on your SSH machine. No accuracy, novelty, or error-free execution
claim is implied. All commands below are for you to run manually.

The pipeline has no dataset/model acquisition code or remote inference client.
Hugging Face is forced into offline mode and all model loads require local
files. The official evaluator command runs only the local script you provide.

## Files

```text
configs/experiment.yaml      All input paths and controlled hyperparameters
src/nl2sql/common.py         IO, hashes, reviews, offline policy, run metadata
src/nl2sql/schema.py         Stable original-identifier schema serialization
src/nl2sql/sql.py            SQL parsing and read-only bounded SQLite execution
src/nl2sql/data.py           Spider preparation, splits, quarantine and audits
src/nl2sql/augment.py        AST proposals, semantic validation, A0–A3 construction
src/nl2sql/diagnostics.py    Controlled schemas, fixtures, families and freeze
src/nl2sql/freeze.py         Immutable final protocol and fixture fingerprints
src/nl2sql/provenance.py     Shared prediction and scoring archive checks
src/nl2sql/orchestrate.py    Sequential runs with verified artifact reuse
src/nl2sql/model.py          Local T5 fine-tuning, checkpointing and decoding
src/nl2sql/evaluate.py       Execution, paired metrics, official evaluator bridge
src/nl2sql/analysis.py       Seed summaries and paired family bootstrap intervals
src/nl2sql/external.py       Local robustness/composition interchange adapter
src/nl2sql/cli.py            Explicit stage commands
scripts/                    Optional sequential Linux orchestration
tests/                      Tests supplied for manual execution
docs/                       Reviews, research protocol, report and annotation guide
```

## SSH environment and local inputs

Use Linux, Python 3.10–3.12, a working SQLite build, and sufficient memory for
the selected token lengths and batch size. Training supports one process per
run on CPU or a CUDA GPU; do not launch with torchrun. fp32 is the default;
select fp16/bf16 only for a suitable CUDA device. A larger model can be supplied
through the same local seq2seq interface after a new baseline pilot.

Transfer this whole repository, the dataset, the pretrained model/tokenizer,
and optionally the official evaluator checkouts and test suites to your SSH
system yourself. Do not transfer a Windows virtual environment.

Expected data layout (edit configuration if your layout differs):

```text
data/spider/train_spider.json
data/spider/train_others.json
data/spider/dev.json
data/spider/tables.json
data/spider/database/<db_id>/<db_id>.sqlite
models/t5-small/config.json
models/t5-small/model.safetensors         # or pytorch_model.bin
models/t5-small/tokenizer_config.json
models/t5-small/spiece.model              # tokenizer assets from same checkpoint
models/t5-small/...                       # retain all original local assets
```

List only training files actually present in `data.train_files`. Do not put
Spider dev in that list. Record the dataset version, access/license terms and
the original pretrained model commit in `configs/experiment.yaml`. Keep SQLite
inputs checkpointed and closed; immutable read-only queries do not read pending
WAL transactions. Run all commands from the repository root. Paths are relative
to that directory, not relative to the YAML file.

Dependencies are pinned in `requirements.txt` and `pyproject.toml`. Use the
appropriate already available PyTorch build for your GPU driver. If your SSH
environment is already configured, expose the package without installing it:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -m nl2sql --help
```

Alternatively, install only from a wheelhouse that you provision yourself:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --no-index --find-links /path/to/wheelhouse -r requirements.txt
python -m pip install --no-index --find-links /path/to/wheelhouse setuptools==75.8.0 wheel==0.45.1
python -m pip install --no-index --no-deps --no-build-isolation -e .
# Optional tests, using an already provisioned pytest wheel and its dependencies:
python -m pip install --no-index --find-links /path/to/wheelhouse pytest==8.3.4
python -m pytest
```

No setup or test command has been run by the authoring agent. The supplied
versions form an explicit starting environment rather than an executed lockfile
with platform-specific hashes. Every actual run records installed package
versions, model file hashes, hardware, configuration and source commit.

## Manual workflow

### 1 Read the protocol and record the literature decision

Read `docs/research_protocol.md`, complete `docs/related_work.csv`, and fill in
`docs/novelty_review.json` after reading/searching the relevant literature.
The proposal's literature references are supplied in `docs/references.md`.
No literature review has been performed or approved for you. The review files
start with `approved: false` deliberately; only you can record real evidence.

### 2 Prepare local Spider data once

```bash
python -m nl2sql prepare
```

This freezes train/internal-validation/final-test JSONL files, schema strings,
hashes, database-disjoint splits, and a leakage audit under artifacts/prepared.
Families and near-duplicate questions cannot cross partitions. Near-duplicate
train/dev wording is quarantined, and training databases connected by similar
questions are kept together when assigning internal validation. The achieved
validation fraction may differ from the requested fraction because groups are
indivisible. Shared abstract SQL structures are reported; ordinary Spider is
not falsely described as a compositional holdout.

Official development rows are preserved, including any repeated records, so
evaluation keeps the published denominator. Exact duplicate training examples
are removed before the training split is frozen. Changing split or dataset
configuration after preparation requires a new prepared-data version.

If grouping connects every training database, preparation fails with a clear
message. Inspect the data and threshold before freezing a justified protocol.
Prepared data is never silently regenerated or overwritten.

### 3 Run a baseline pilot and inspect its internal validation errors

```bash
python -m nl2sql train --condition A0 --seed 13 --pilot
python -m nl2sql predict --checkpoint runs/pilot/seed_13/model \
  --dataset artifacts/prepared/validation.jsonl \
  --output runs/pilot/seed_13/validation_predictions.jsonl --condition A0 --seed 13
python -m nl2sql evaluate --dataset artifacts/prepared/validation.jsonl \
  --predictions runs/pilot/seed_13/validation_predictions.jsonl \
  --output runs/pilot/seed_13/validation_evaluation
```

If an input is too long, the default is to stop instead of silently removing
schema information. Increase max_input_length, reduce batch size if necessary,
and choose a new run root before the final protocol is frozen. Gold SQL targets
are never silently truncated. Inspect validation execution accuracy, SQL validity,
learning curves in trainer_state.json, and error_review.jsonl. Complete
docs/baseline_review.json only after the baseline is suitable. Set
baseline_is_suitable to true and pilot_run to the completed pilot directory
(for example, runs/pilot/seed_13) or its run.json. Freeze checks its model and
validation identities. Pilot runs are
stored separately from final A0 runs and cannot replace them in the summary.

### 4 Prepare and review training augmentation

```bash
python -m nl2sql propose
cp artifacts/augmentation/candidates.jsonl artifacts/augmentation/reviewed.jsonl
```

Follow `docs/annotation_guide.md`. Candidate questions are intentionally empty:
an SQL operator edit cannot safely invent a correct English gold question.
Write each new question, check the SQL against its schema and original intent,
complete the review checklist, and explicitly accept/reject it. Proposals cover
paraphrases, simple numeric comparisons and simple AND/OR; manually authored
negation/quantifier counterfactuals can use the same record format. Never turn
all review booleans on without actually reviewing the examples.

```bash
python -m nl2sql validate-augmentation
```

Every input record is retained in accepted.jsonl or rejected.jsonl with reasons.
Unknown/ambiguous/unchecked gold does not enter training. A counterfactual must
produce different gold results on at least one supplied instance. For natural
Spider data with no witness, reject the item or supply a verified schema-compatible
test suite consistently to the source pipeline; do not bypass this check.

Review a small pilot before scaling, record findings in
docs/augmentation_pilot_review.json with a positive pilot_examples_reviewed
count, and obtain the configured budget of accepted
paraphrases and counterfactuals. The default budget is 1,000 per type; choose a
smaller justified pilot budget in configuration if necessary. The builder refuses
to silently underfill or rebalance an unequal budget.

```bash
python -m nl2sql build-conditions
```

The combined condition uses both full budgets; A3_matched uses one total budget,
split approximately equally. All conditions retain all original supervised
training examples. Optimizer steps are fixed by default; actual nonpadding input
and target token exposure is logged because equal steps alone do not equalize
tokens exactly. Avoid enabling early stopping for the main matched-step comparison.

All five condition files are built for reproducibility. training.conditions
selects the runs: A0–A3 are required; A3_matched is an optional control enabled
by default. If you omit it for compute reasons, update planned_conditions in
the final review before freezing. Never omit an unfavorable run after evaluation.

### 5 Build and manually review the independent diagnostic set

```bash
python -m nl2sql build-diagnostics
cp artifacts/diagnostics/candidates.jsonl artifacts/diagnostics/reviewed.jsonl
```

The default creates 200 source families, 600 questions, and three SQLite
instances. Every family has an original, a paraphrase, and a counterfactual.
Its schema is independent of Spider; all questions are final evaluation only.
Review every question, SQL, scope and NULL interpretation before proceeding.
Threshold variants share templates; report that limitation prominently.

Each record includes plausible_incorrect_sql alternatives. Fixtures must
distinguish all of them, including boundary mistakes, dropped constraints and
COUNT(total) versus COUNT(*). Customer counts include NULL-only, non-paid
orders at the quantifier boundaries. Fixture instances use different IDs and
assignments; validation records the distinguishing database witnesses.

```bash
python -m nl2sql freeze-diagnostics --reviewed artifacts/diagnostics/reviewed.jsonl
```

This writes the frozen test file and hashes only if all records pass review,
SQL validity, execution and family/contrast checks. Failed items are logged.
For more lexical variety, manually revise and review the pending questions while
preserving their family metadata before freezing; never revise after viewing
model predictions.

### 6 Optional local Dr.Spider or Spider-CG evaluation

Use the explicit interchange format in `docs/external_benchmarks.md` and the
import-benchmark command. No raw benchmark layout is guessed or downloaded.
Keep Dr.Spider NLQ original/perturbed questions together and record that they
derive from Spider dev. Database-changing perturbations are outside this
adapter's first-version scope. A composition benchmark is optional and requires
its published split policy; generic import alone does not establish composition
generalization.

### 7 Freeze final choices and datasets

Complete docs/final_evaluation_review.json, including the planned seeds and both
review booleans. Keep the configuration, annotations and evaluation sets fixed.

```bash
python -m nl2sql freeze --datasets artifacts/prepared/test.jsonl artifacts/diagnostics/test.jsonl
# Include any imported external test JSONL paths in this same command.
```

The freeze requires both Spider and controlled diagnostic tests. It verifies
every training-condition file, accepted annotation fingerprints, pilot evidence
and dataset producer manifests. It registers final data/database hashes and
binds them to the configuration, planned runs, pretrained model, Python source,
research protocol, literature matrix and completed reviews. Bytecode caches
are excluded from source fingerprints. Adding another
dataset or changing hyperparameters requires a separately versioned protocol
and output roots, not edits to a completed final experiment.

### 8 Train every condition and seed

```bash
bash scripts/run_experiments.sh configs/experiment.yaml
# Equivalent direct command:
python -m nl2sql run-experiments
# Or train one explicit run:
python -m nl2sql train --condition A0 --seed 13
```

All conditions begin from the same local pretrained checkpoint. Validation loss
selects the checkpoint; final datasets are never used for selection. The pilot
and completed final runs are preserved separately. Failed/partial runs are not
silently resumed; inspect them and use a new runs root for a fresh run. A change
to a frozen configuration requires a new final protocol.

The orchestration commands verify existing checkpoints, predictions and scoring
archives before reuse. They stop on stale, edited, partial or mismatched outputs.
Completed summaries can be reused only when all input/output hashes and analysis
settings agree. Each training/prediction stage runs in a separate process to
release GPU memory between runs.

The configured ordinary added tokens preserve SQL comparison symbols such as
`<` that plain T5 tokenizers can otherwise encode as unknown tokens. Embeddings
are resized after seeded initialization; all conditions with the same seed use
the same initialization. Gold targets containing any remaining unknown symbols
stop training with source IDs so you can extend the token list before freeze.

### 9 Decode and evaluate all frozen datasets

```bash
bash scripts/evaluate_all.sh configs/experiment.yaml
# Equivalent direct command:
python -m nl2sql evaluate-all
```

For one explicit dataset/run:

```bash
python -m nl2sql predict --checkpoint runs/A0/seed_13/model \
  --dataset artifacts/diagnostics/test.jsonl --output runs/A0/seed_13/diagnostic_predictions.jsonl \
  --condition A0 --seed 13 --final
python -m nl2sql evaluate --dataset artifacts/diagnostics/test.jsonl \
  --predictions runs/A0/seed_13/diagnostic_predictions.jsonl \
  --output runs/A0/seed_13/diagnostic_evaluation
```

Inspect metrics.json, items.jsonl, pairs.jsonl, error_review.jsonl, and
demonstration.jsonl. These archive question, gold/predicted SQL, run identity,
execution errors, instance-level correctness and suggested error categories.
Suggestions require manual review; they do not automatically establish
paraphrase meaning drift, scope mistakes, or causal explanations.

### 10 Run the official evaluators you supplied locally

```bash
python -m nl2sql official-evaluate --kind spider \
  --dataset artifacts/prepared/test.jsonl \
  --predictions runs/A0/seed_13/predictions/dataset_0.jsonl \
  --script /path/to/local/spider/evaluation.py \
  --database-root data/spider/database --tables data/spider/tables.json \
  --output runs/A0/seed_13/official_spider

python -m nl2sql official-evaluate --kind test-suite \
  --dataset artifacts/prepared/test.jsonl \
  --predictions runs/A0/seed_13/predictions/dataset_0.jsonl \
  --script /path/to/local/test-suite-sql-eval/evaluation.py \
  --database-root /path/to/local/distilled_test_suites --tables data/spider/tables.json \
  --output runs/A0/seed_13/official_test_suite
```

Use an actual compatible official checkout and its required local dependencies
and test-suite SQLite instances. The bridge uses the common --gold, --pred,
--db, --table, --etype flags, archives stdout/stderr, the exact command and source
hashes, and raises on nonzero exit. SQLite test-suite instances are fingerprinted
as well, so the evaluator data version is recorded. Consult the supplied
checkout's own CLI if using a fork with a different interface. Repeat for all
seeds/conditions.
Official aggregate scores remain in the evaluator's stdout; they are not
guessed by parsing an undocumented console format. Incorporate them into the
report with the evaluator identity and denominator. Official paired per-item
scores require a compatible per-item export from your evaluator; the built-in
paired metrics use the explicit strict SQLite policy instead.

The bridge checks the same prediction/configuration/fixture provenance as the
local evaluator. Empty model outputs are exported as invalid SQL placeholders
so evaluators that skip blank lines cannot shift subsequent example IDs.
item_order.jsonl records the exact export alignment. SQL comments and external
whitespace are folded while quoted spaces are preserved. Literal newlines or
tabs inside quoted SQL values cannot be represented by the official line-based
protocol and cause an explicit error instead of silently changing the query.

### 11 Aggregate and write the report

The evaluation shell script produces one summary per frozen dataset. For manual
aggregation, pass every condition/seed evaluation directory:

```bash
python -m nl2sql summarize --evaluations runs/A*/seed_*/evaluation/dataset_0 \
  --output runs/summary/manual_spider
```

The summarizer refuses an incomplete configured condition/seed matrix. It verifies
scoring archive hashes and rejects mixed configurations or scoring policies.
It records means, sample SD across seeds, denominators, per-phenomenon results
and matched-seed/family bootstrap differences from A0. A3_matched is required
when it is in the configured plan. Use docs/report_template.md to write the
final research report and cite actual
official results separately. Report negative/mixed results and the full error
taxonomy. Novelty and publication potential remain research questions.

## Metric interpretation

The built-in comparator uses exact SQLite values, strict column order,
duplicate-preserving row bags, and row order when the gold query has a top-level
ORDER BY. It can disagree with official Spider equivalence policies (column
permutations, numeric handling and normalization). Its syntactic match diagnostic
is not official exact match. A one-instance strict suite score is just an
execution score on that instance, not a distinguishing semantic test suite.

Primary execution accuracy and SQL validity use the first database instance;
strict suite accuracy and suite SQL validity require success on every instance.
Per-instance validity/errors are archived independently, so a later execution
failure cannot erase a correct primary-instance result. For a family containing
multiple perturbation categories, paired per-phenomenon metrics use each
sibling's category and preserve its original link.

Paraphrase family correctness requires all family members to be correct;
consistency alone can be high when both predictions are wrong. Counterfactual
contrast accuracy requires both members correct and predictions observably
different on supplied fixtures. Empty or invalid gold aborts evaluation rather
than being counted as a model error. Final diagnostics and Spider/Dr.Spider are
complementary evidence; template-controlled successes do not prove broad transfer.
