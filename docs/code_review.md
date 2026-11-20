# Static review against the research proposal

The implementation was reviewed again against the supplied Word proposal and
its implementation, data-quality, fairness, evaluation and reproducibility
requirements. This review used file reading and editing only. No Python,
training, SQL query, test, package installation, evaluator, or network download
was executed. The regression tests listed below are supplied for manual use.

## Corrections

| Area | Correction |
|---|---|
| Execution metrics | A later database failure no longer erases a correct primary result; every instance has validity, correctness and errors |
| Pair scoring | Shared source/schema/fixture links are validated; mixed perturbation categories use sibling categories |
| Diagnostic sensitivity | NULL-only/non-paid orders at count boundaries expose COUNT(column) and dropped/status-filtered constraints |
| Alternative gold validation | Controlled examples contain plausible wrong queries; each needs a distinguishing fixture witness |
| Fixture independence | IDs are consistently renumbered between instances; construction hashes are verified before review freeze |
| Official benchmark size | Repeated development rows remain in the official denominator with unique IDs |
| Leakage checks | Equal IDs in different records no longer suppress near-duplicate comparisons; numeric signs/decimals survive normalization |
| Training augmentation | Redundant original-question copies are rejected; accepted rows are revalidated and audited before condition construction |
| AST transformation scope | Comparison proposals edit only the selected WHERE predicate, even if the identical expression also occurs in SELECT or a CTE |
| Full-study freeze | Every condition is checked for frozen content, original membership, type budget and approved additions |
| Dataset provenance | Reviewed diagnostic/import manifests and fixture hashes are required, not just status strings |
| Research evidence | Final freeze verifies an identified suitable baseline pilot and a positive augmentation pilot review count |
| Protocol immutability | Source Python, model files, protocol, literature matrix and reviews are bound to final data/configuration; bytecode is excluded |
| Compute controls | A0–A3 remain required; A3_matched can be omitted in the predeclared plan when compute is limited |
| Artifact reuse | Existing run/prediction/evaluation/summary outputs are verified before reuse; partial/mixed artifacts stop the stage |
| Official export | Empty predictions preserve row alignment; quoted spaces survive export; unrepresentable literal line breaks fail explicitly |
| Evaluator provenance | Official commands archive item alignment, Python sources, SQLite instance fingerprints and stdout/stderr |
| Configuration and errors | Numeric/batch/seed/condition/token settings, offline reporting and schema references have explicit validation |
| Memory and reproducibility | SQL result cache is bounded; subprocess stages release GPU memory; inference requests deterministic behavior |
| Demonstrations | A complete family per phenomenon is selected without consulting prediction correctness |

## Manual regression coverage

Tests cover read-only execution, NULL/boundary comparisons, bags/order, query
limits, primary/suite failure separation, consistently wrong paraphrases,
quoted-value export, AST predicate scope, family links, numeric deduplication, official dev row
preservation, COUNT(column) diagnostic witnesses, changed condition files,
modified scoring archives and bytecode-independent source fingerprints.

The README, research protocol, annotation guide, external benchmark instructions
and requirement map were updated to match the corrected behavior. Old partial
artifacts or outputs lacking the new provenance/completion manifests should be
regenerated through the documented stages before a new final protocol is frozen.

## Remaining manual research work

Local datasets, weights, tokenizer assets and compatible official evaluator
checkouts must be supplied by the researcher. The literature search, annotation
review, suitability decisions and actual experimental findings are still manual
research tasks. Optional Spider-CG composition claims need a validated published
composition split. Static review does not establish runtime compatibility,
correctness on every possible SQL query, performance improvement or novelty.
