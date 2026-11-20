# Local external benchmark interchange

Raw Dr.Spider and Spider-CG layouts may differ by release. This implementation
does not guess raw field names or acquire files. Prepare a local JSONL mapping
from the files and documentation of the exact benchmark version you hold.
Map original and perturbed natural-language questions into this format:

```json
{"id":"dr-original-001","source_id":"dr-original-001","family_id":"dr-family-001","db_id":"your_db","question":"Original question","gold_sql":"SELECT id FROM your_table","database_paths":["data/external/database/your_db/your_db.sqlite"],"transformation_type":"original","phenomenon":"dr_nlq_perturbation_name"}
{"id":"dr-variant-001","source_id":"dr-original-001","family_id":"dr-family-001","db_id":"your_db","question":"Published equivalent question","gold_sql":"SELECT id FROM your_table","database_paths":["data/external/database/your_db/your_db.sqlite"],"transformation_type":"paraphrase","phenomenon":"dr_nlq_perturbation_name"}
```

Replace illustrative identifiers and queries with the actual published records.
Provide the matching Spider-format tables.json and local SQLite instances.
Every family needs exactly one original, and each sibling's source_id references
it. One original may have multiple paraphrases. Schema/instance paths must agree
within each family. If no pairs exist (for an optional ordinary benchmark), give
each original a distinct family and source_id equal to its own id.

```bash
python -m nl2sql import-benchmark --source data/external/mapped.jsonl \
  --tables data/external/tables.json --output artifacts/external/dr_nlq.jsonl \
  --name Dr.Spider-NLQ --benchmark-version YOUR_LOCAL_VERSION \
  --license 'Record the actual benchmark access and redistribution terms'
```

This parses and executes gold, serializes schemas identically to training,
checks paraphrase equivalence on supplied fixtures, and records source hashes
and benchmark provenance. It does not override published labels or infer that
changed schemas can be paired in the same fixture. The first-version adapter
supports NLQ perturbations with unchanged schema, not database-changing
Dr.Spider perturbations.

Counterfactual imports must have an observable gold contrast on at least one
supplied instance. Original/variant links, schema policy and fixture lists are
checked before import; all fixture hashes are saved in the import manifest.
Freeze verifies that manifest rather than accepting a benchmark_gold label
without provenance. Mixed perturbation categories within a family are retained
in pair details and attributed to the sibling's category during scoring.

Supply the resulting dataset in the same final freeze as Spider and controlled
diagnostics. Train/validation overlap is forbidden; cross-test overlap between
Spider dev and Dr.Spider is expected and must be reported. Do not count repeated
Dr.Spider original questions as independent examples in a pooled estimate. If
the same source appears in multiple perturbation categories, keep a consistent
family or analyze the categories separately with the shared-source dependence
documented.

For Spider-CG, use its actual intended training/evaluation split rather than
blindly importing it as a new independent test. Before any composition claim,
verify that the specific evaluated combinations and near-duplicate questions
are absent from training while their components are present. The generic
interchange adapter cannot establish that property and the core project does
not require this optional experiment.
