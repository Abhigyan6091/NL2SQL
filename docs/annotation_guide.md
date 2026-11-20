# Annotation and quality validation

Candidates are annotation tasks, never automatic gold. Keep the original JSONL
archive; edit a separate reviewed.jsonl. Each line is a complete JSON object.
Use a text editor or your own annotation tool; the pipeline supplies no online
annotation service or model-based judge.

## Record requirements

Every record has a unique id, source_id, family_id, db_id, question, gold_sql,
transformation_type, phenomenon, split, schema, schema_serialization_version,
database_paths, semantic_review_status and provenance. Training variants retain
the frozen source's family_id, source_id, db_id, split, schema and database_paths.
original_question and original_sql must match the frozen source exactly.
provenance.source_record_sha256 must be retained from the generated candidate.
Do not change source metadata to make a check pass.

For a manually authored counterfactual not in the automatic catalogue, copy a
candidate derived from that same original source. Give it a new unique id,
change transformation_type to counterfactual, set the phenomenon, and write the
coordinated question and SQL. Keep the original fields and source hash. Record
the manual generation procedure and author in provenance. The validator applies
the same review and execution standard to every augmentation type.

The default AST catalogue proposes simple numeric comparisons and a simple
top-level AND/OR swap. It deliberately leaves question text blank. Changing a
SQL operator does not imply that a natural-language edit is valid. NOT EXISTS,
grouping and quantifier transformations require hand-verified SQL and questions.
SQL text is parsed structurally rather than rewritten with regex replacements.

## Semantic review

Set semantic_review_status to reviewed only after checking all of the following:

1. Naturalness: a reader can understand the question without reconstructing SQL.
2. Unambiguous intent: there is one intended logical interpretation.
3. SQL alignment: tables, columns, joins, aggregation and projection express it.
4. Constraint preservation: values, entities, quantifiers, scope and all other
   predicates survive in paraphrases; a counterfactual changes only its intended
   logical aspect and preserves the remaining intent.
5. NULL semantics: missing values, COUNT(column) versus COUNT(*), NOT IN versus
   NOT EXISTS, and zero-related-row cases have been considered explicitly.
6. Diagnostic sensitivity: fixture rows distinguish gold from plausible wrong
   operators, boundaries, scopes or dropped constraints.
7. Provenance: generator/author, tool/model version if used, source family and
   procedure are recorded. A model output still requires human review.

The review object must contain:

```json
{
  "decision": "accept",
  "reviewer": "your-name-or-reviewer-id",
  "reviewed_at": "2026-11-10T10:00:00+05:30",
  "notes": "Describe the constraints, scope and boundary/NULL cases actually checked.",
  "natural": true,
  "unambiguous": true,
  "sql_matches_question": true,
  "constraints_checked": true,
  "null_semantics_checked": true
}
```

The date is an illustration, not a prescribed annotation date. For a rejected
item, use decision reject and explain why in notes; unchecked/uncertain items
remain pending. Both are logged as rejected by validation and cannot enter
training. Retain these records to measure the candidate acceptance rate.

## Automated checks and their limits

SQLGlot validates a single read-only query. SQLite compilation/execution resolves
schema references and enforces the actual database schema; time and result-size
limits prevent runaway recursive queries or huge results. A valid SQL query can
still be the wrong annotation. Human review is a separate required check.

A paraphrase must agree with original gold on all supplied fixtures. A
counterfactual must differ on at least one fixture. These checks are evidence
under the provided instances; they do not prove equivalence on every possible
database. For order-sensitive questions, ensure fixtures avoid unspecified tie
order or give a deterministic tie-breaker consistent with the English question.

Controlled records also contain plausible_incorrect_sql. Every listed
alternative must execute and differ from the reviewed gold on at least one
fixture. Validation saves alternative_witnesses and counterfactual
contrast_witnesses. Review these alternatives alongside the intended meaning;
do not delete a plausible alternative merely because the current fixture fails
to distinguish it. Strengthen the fixture in a new construction version instead.

Training candidates are checked against frozen training source IDs, source
record hashes, schema policy and held-out near-duplicates. Exact duplicate
questions in an annotation batch are rejected. Accepted records carry a
validation fingerprint and are revalidated when conditions are built. Editing
an accepted record invalidates its fingerprint; edit the reviewed input and run
validation again before freezing conditions.

Candidates that duplicate any original training question in the same database
are rejected as redundant augmentation. Exact normalization preserves numeric
signs and decimal values, so -1.5, 1.5 and 15 are not collapsed into one label.
Near-duplicate screening remains lexical evidence rather than a semantic proof.

## Controlled diagnostic review

Every family contains original, paraphrase and counterfactual records. Check
all three; do not assume the generated original is already trusted. The
controlled database includes price and total boundaries, paid/pending/NULL
statuses, NULL numeric values, and customers with zero orders. EXISTS/NOT EXISTS
operates over customers; quantifiers use correlated COUNT(*) so zero-order
customers are not accidentally excluded by an inner join.

NULL-only, non-paid orders are present at the counting boundaries to expose
COUNT(total) and inappropriate status filters. Separate instances renumber IDs
consistently, preventing a prediction that lists fixed fixture IDs from passing
multiple copies of the same database. Keep the construction manifest unchanged;
freeze checks its schema and fixture identities before accepting annotations.

All diagnostic records are test-only. They must never enter the training
augmentation file or checkpoint selection. Keep every sibling in its family.
Each reviewed file must have complete triples. Freeze once before final model
predictions. If a later annotation error is discovered, document it and issue a
new benchmark version with all conditions reevaluated, rather than changing only
the failed model's questions.

## Error taxonomy for prediction analysis

Use one reviewed primary category and optionally secondary notes:

| Category | Meaning |
|---|---|
| paraphrase_meaning_drift | Candidate wording changed intent; annotation failure |
| operator_confusion | Comparison, AND/OR, negation or quantifier misinterpreted |
| dropped_constraint | Required predicate or condition absent |
| incorrect_scope | Grouping, quantifier or negation scope incorrect |
| schema_linking_error | Wrong table, column or relationship |
| invalid_sql | Parser/execution failure |
| evaluation_ambiguity | Fixture collision, unspecified order or evaluator limitation |
| other | Explain an error not captured above |

The automatic error_category_suggestion is only a triage hint. Fill
error_review.jsonl manually; AST counts cannot distinguish all linguistic causes.
