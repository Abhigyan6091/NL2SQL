# Semantic robustness in Text to SQL

Author, institution, date, protocol version, code commit and artifact location:
complete after running the study. This is a report structure, not experimental
findings. Replace instructions with evidence and retain null/negative outcomes.

## Abstract

State the research question, dataset/model, four controlled conditions,
principal paired outcomes, measured uncertainty and limitations. Supply numbers
only after running the frozen experiment. Do not claim novelty without the
completed related-work comparison.

## Problem and hypotheses

Explain paraphrase invariance and sensitivity to meaning-changing edits. State
RQ1–RQ4 and the hypotheses frozen before final evaluation.

## Related work and contribution boundary

Summarize the completed comparison matrix, including Spider-CG, Dr.Spider,
natural-language variation augmentation, ambiguity and test-suite evaluation.
State the precise remaining gap or explain the replication/negative-result
value if the comparison is not novel. Include search dates and scope decisions.

## Data and annotation

Record dataset versions, terms, source hashes, database-disjoint split counts,
near-duplicate threshold, achieved validation fraction and quarantined items.
Give augmentation candidates/accepted/rejected counts by type and phenomenon,
reviewer procedure, generator/tool versions and common rejection reasons.
Describe diagnostic schemas, fixtures, family counts, boundary/NULL cases and
manual verification. Distinguish Dr.Spider's relation to Spider dev. Do not claim
an optional compositional holdout without validating its combination split.

## Methods and controls

Give pretrained model revision/hash, serialization version, token limits,
optimizer, learning rate, precision, batch and accumulation sizes, optimizer
steps, beam settings, seeds and checkpoint-selection policy. Explain A0–A3 and
A3_matched budgets. Report actual step/input/target token exposure, runtime and
hardware, and discuss any mismatch between conditions.

## Results

Populate every condition and planned seed, including worse-performing runs.
Identify the exact official evaluator checkout and distinguishing test-suite
version. Keep built-in strict SQLite scores separate from official scores.

| Condition | Seeds | Official test-suite accuracy | Official execution accuracy | Official exact match | SQL validity |
|---|---:|---:|---:|---:|---:|
| A0 | pending | pending | pending | pending | pending |
| A1 | pending | pending | pending | pending | pending |
| A2 | pending | pending | pending | pending | pending |
| A3 | pending | pending | pending | pending | pending |
| A3_matched | pending | pending | pending | pending | pending |

Include paraphrase family correctness, consistency and counterfactual contrast
accuracy with denominators, means/SD, matched-family/seed bootstrap deltas and
intervals. Present per-phenomenon results with counts. Mark unavailable official
metrics unavailable; never substitute syntactic matching silently.

## Error analysis and demonstrations

Review errors under paraphrase meaning drift, operator confusion, dropped
constraint, incorrect scope, schema linking, invalid SQL and evaluation ambiguity.
Show representative original/paraphrase/counterfactual families, gold/predicted
SQL, fixture witness and interpretation. Include failures as well as successes;
automated error hints need manual review.

## Discussion and limitations

Separate overall accuracy changes from changes to the hypothesized paired
mechanism. Discuss annotation quality, generation-template similarity,
near-duplicate detection limits, token exposure, weak-baseline risk, fixture
collisions, official comparator differences, small categories, shared-source
benchmark dependence, and uncertainty over seeds versus unseen templates.
Do not infer broad transfer from controlled templates alone.

## Conclusion

State what the measured evidence supports, what remains unresolved, and the
most useful next experiment. A null or mixed result is valid. Do not guarantee
publication or claim causality beyond the controls actually implemented.

## Reproducibility appendix

Link split/condition/diagnostic manifests, configuration, review evidence,
accepted/rejected annotation logs, checkpoints, package/hardware metadata,
predictions, local/official evaluation archives and summary files. Respect
redistribution terms when packaging data or benchmark labels.
