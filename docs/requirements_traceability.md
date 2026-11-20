# Proposal implementation map

| Proposal requirement | Implementation or explicit manual deliverable |
|---|---|
| Sections 1–4 research framing and hypotheses | research_protocol.md and report_template.md |
| Section 5 full literature/novelty check | related_work.csv, references.md, novelty_review.json; manual research pending |
| Section 6 narrow first-version scope | Offline seq2seq training and four phenomenon catalogue; no agent/UI or custom loss |
| Section 7 Spider input and schema | data.py and schema.py |
| Database-disjoint internal validation | Frozen grouped database split in data.prepare |
| Family and near-duplicate leakage checks | data.audit_splits, augmentation validation and final freeze |
| Independent controlled test set | diagnostics.py; default 200 reviewed source families |
| SQL parser, schema references and execution | sql.py; SQLite compilation resolves actual schema names |
| Human semantic correctness | Mandatory record checklist and validation gate |
| Meaning-changing labels and witnesses | AST proposals plus reviewed questions; contrast execution checks |
| Training-only augmentation | Frozen source membership and record hash validation |
| Dr.Spider and optional Spider-CG | Explicit local interchange adapter and documented scope limits |
| Section 8 pretrained T5 and fixed input format | model.py, schema.model_input and experiment.yaml |
| Checkpoint selection on internal validation | Minimum internal validation loss; pilot review on internal errors |
| Multiple seeds and controlled updates | Configured seed matrix and fixed optimizer-step budget |
| Section 9 A0–A3 and data-matched combined | augment.build_conditions; A3_matched |
| Token/step fairness documentation | Actual training exposure counters and manifest controls |
| Frozen common evaluation | freeze.py hashes questions, SQLite fixtures and configuration |
| Section 10 primary official evaluation | official-evaluate local CLI bridge with source/command/stdout archives |
| Execution, validity and syntax diagnostic | evaluate.py with explicit comparator policy |
| Paraphrase correctness and consistency | Family correctness and paired strict execution metrics |
| Counterfactual contrast accuracy | Both outputs correct and observably different |
| Per-phenomenon counts | Metrics denominators and per_phenomenon output |
| Seed variability and intervals | analysis.py; complete matrix and paired bootstrap |
| Error taxonomy and examples | error_review.jsonl, demonstration.jsonl and annotation_guide.md |
| Sections 11–12 milestones and risks | Research gates, protocol and report limitations |
| Section 13 code and reproducibility | Package, configs, manifests, run metadata and scripts |
| Final research report | Report template supplied; actual findings require manual runs |
| Appendix A record provenance and quality | Rich JSONL records, validation fingerprints and review checklist |
| Distinguishing fixtures for plausible errors | Per-record incorrect alternatives and database witness archives |
| Preserve official benchmark denominators | Development rows retain unique IDs even when content repeats |
| First-instance versus test-suite execution | Independent per-instance correctness, validity and errors |
| Optional compute-sensitive controls | Configured condition matrix; A3_matched optional before freeze |
| Reuse of completed experiments | orchestrate.py verifies checkpoints, predictions, scoring archives and summaries |
| Official export reproducibility | Stable item_order.jsonl, empty-output alignment and SQLite instance hashes |

No dataset, pretrained weights, evaluator checkout, completed annotation,
experiment result or literature approval is fabricated. These require your
local inputs and manual research/execution. All requested runnable code and
supporting templates are supplied; no project code or tests were run during
authoring.
