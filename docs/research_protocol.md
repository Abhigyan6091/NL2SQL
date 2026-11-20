# Research protocol

This protocol implements the supplied proposal as an experiment to be conducted.
No literature decision, annotations, baseline feasibility check or result is
asserted to have been completed. The user requested implementation only, so all
execution and research reviews remain manual stages.

## Questions and hypotheses

RQ1 measures the baseline's paraphrase invariance and sensitivity to changed
meaning. RQ2 tests whether paraphrase augmentation improves equivalent-question
correctness. RQ3 tests whether counterfactual augmentation improves logical
contrasts. RQ4 tests whether the interventions complement or interfere with one
another. These are falsifiable hypotheses, not promised improvements.

The contribution is provisionally a controlled comparison of the two
augmentation types with matched budgets and family-level semantic evaluation.
Read the closest work and record a gap before claiming novelty. If the exact
comparison already exists, revise the scope and protocol before final training.

## Data and partitions

Spider training files provide supervised examples and the sole source of
training augmentation. Split them by database, keeping similar-question
database groups together. Keep all descendants of a source in one partition.
Official Spider development examples are locked final evaluation data. Do not
select checkpoints or augmentation prompts with final Spider dev results.

Automated exact/high-Jaccard near-duplicate checks are conservative lexical
checks. They do not establish the absence of all semantic duplicates. Inspect
provenance and the audit; record the threshold and quarantine counts. Abstract
SQL overlap is reported. The default split does not claim abstract-structure
disjointness or compositional generalization.

The independent controlled shop schema is final diagnostic data. The default
200 families target comparisons, AND/OR, negation and quantifiers, with three
questions per family. The number is a planning parameter, not a quality
threshold. Boundary and NULL cases are explicit. Every question must be
manually reviewed before freeze. Numeric thresholds share templates, which
limits effective linguistic diversity and the interpretation of confidence
intervals.

Dr.Spider NLQ perturbations may be imported as complementary paired evaluation.
They derive from Spider dev, so do not count them as independent domain evidence.
Spider-CG is optional; inspect and honor its published split and verify held-out
combinations before claiming compositional generalization.

## Model and training controls

Use local pretrained T5-small as the initial baseline, with the same original
identifier schema serialization and foreign-key format in every condition.
The configuration fixes token limits, tokenizer, optimizer, learning rate,
batch sizes, number of optimizer steps, seed list and beam decoding settings.
Train each condition anew from the same pretrained files, not from another
condition's fine-tuned model.

Use the same configured ordinary SQL symbol tokens and embedding resize policy
for every condition. Added comparison symbols are not special tokens and are
retained during generation decoding. Gold SQL containing unknown-token symbols
fails preparation for training rather than silently discarding an operator.

| Condition | Added examples |
|---|---|
| A0 | None |
| A1 | B reviewed paraphrases |
| A2 | B reviewed counterfactuals |
| A3 | B paraphrases plus B counterfactuals |
| A3_matched | floor(B/2) paraphrases plus the remaining counterfactual budget |

A0–A3 are mandatory. A3_matched is an optional control selected through
training.conditions before final freeze; it is enabled by default. The final
review and summary must use exactly the configured condition and seed matrix.

All conditions retain the same original training set. B is fixed before final
training. Candidate selection is seeded, and selected rows are frozen in
condition files. Quality thresholds are identical for both augmentation types.
Actual input/target token exposure is logged because fixed steps and batch size
do not guarantee exactly matched tokens or effective passes through each set.
If exposure differs materially, discuss this confound or predefine an additional
token-matched study; do not retrofit a favorable control after final results.

The initial configuration uses 3,000 optimizer steps as an explicit matched
budget, rather than treating the proposal's rough 3–5 epoch pilot suggestion as
a hard requirement. Adjust it using only internal validation before final
freeze. The minimum internal validation loss selects the saved checkpoint.
Early stopping is disabled by default to preserve matched steps. If enabled,
report actual step/token exposure and the changed fairness interpretation.

Run the three configured seeds when possible. If compute is limited, change the
planned seeds before freezing and acknowledge the loss of variability evidence.
Do not select a single favorable seed. There is no custom contrastive loss.

## Milestone evidence

| Milestone | Evidence to complete |
|---|---|
| Literature review | related_work.csv and novelty_review.json |
| Baseline feasibility | Pilot checkpoint, validation predictions, metrics and baseline_review.json |
| Gold quality | Reviewed pilot, acceptance/rejection logs and augmentation_pilot_review.json |
| Data freeze | Split/condition manifests, leakage audit and reviewed diagnostics |
| Final protocol | final_evaluation_review.json and final_freeze.json |
| Training | Every condition/seed checkpoint, run.json and trainer logs |
| Evaluation | Official evaluator archives and local item/pair metric archives |
| Report | Complete methods, uncertainty, error analysis, limitations and demo |

The code supplies review records but does not self-approve them. You must fill
in actual evidence and reviewer identity. A baseline pilot may run before the
full-study gates, allowing feasibility debugging without pretending the
research decisions are already settled.

The baseline review identifies the completed pilot directory/run.json and
explicitly confirms suitability. Freeze verifies the pilot's pretrained model,
input policy and validation identity. The augmentation review records a positive
reviewed example count. Freeze also verifies every condition's original-example
membership, added-example budgets, annotation hashes and reviewed diagnostic
producer manifest, then binds the model, source and protocol to final data.

## Outcomes and interpretation

Where compatible local official tools and suites exist, official test-suite
execution accuracy is the principal ordinary semantic accuracy measure. Also
report official execution accuracy, official exact match and SQL validity.
The built-in strict SQLite comparator is explicitly a separate implementation;
its normalized syntax equality is only a syntactic diagnostic.

Primary execution and validity refer to the first supplied database. Suite
scores require success across all supplied instances; later instance failures
are archived without overwriting primary correctness. The official bridge
uses the same prediction provenance checks, preserves example alignment even
for empty outputs, and records hashes of the actual evaluator SQLite instances.

For paired diagnostics, report all-member paraphrase family correctness,
pair consistency and counterfactual contrast accuracy. Correctness prevents
two equally wrong predictions from appearing robust. Counterfactual success
requires an observable gold difference and both members correct. Report
per-phenomenon denominators as well as proportions.

Summaries require the full planned A0–A3 matrix. Report mean and sample SD across
seeds. Bootstrap intervals resample matched seeds and source families with
replacement, using the same family draw for the compared conditions. Delta
intervals weight families equally; for unequal family sizes this differs from
item-weighted aggregate accuracy. Controlled threshold families share templates;
the intervals do not capture uncertainty over entirely new linguistic templates.
Intervals are exploratory and not adjusted for multiple comparisons.

Paraphrase-specific gains support invariance under this setup. Counterfactual
gains support sensitivity to the tested changes. Overall accuracy gains without
paired gains do not establish the proposed semantic mechanism. Generated-only
gains do not establish transfer. Negative/mixed outcomes remain valid if the
baseline, annotation quality and uncertainty are documented. Do not guarantee
publication potential or novelty.

## Reproducibility and outputs

Archive local dataset/model versions and terms, source hashes, split files,
schemas, candidate/review/accepted/rejected records, frozen condition manifests,
configuration and gate evidence. Save every checkpoint identity, package
version, hardware description, seed, token limit, optimizer setting, runtime,
checkpoint-selection history and actual token exposure.

Archive all gold/predicted SQL and execution failures. Run the official tools
with their provided local data and dependencies; retain their commands and
source hashes. Manually review representative failures using the full taxonomy,
and prepare a few family demonstrations showing both success and failure.
Redistribute benchmark data only if its license and access terms permit it.
