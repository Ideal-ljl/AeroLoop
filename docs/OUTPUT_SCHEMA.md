# JSONL output schema

Each run writes, in order:

1. one `run_config` row;
2. one `episode` row per evaluated episode;
3. one `environment_summary` per environment;
4. one `overall_summary` row.

Episode rows contain the termination reason, aggregate metrics, and optionally
the complete step trace. Every step stores the requested canonical action,
pose before/after execution, collision flag, all distance variants, inference
call/action indices, latency, and model metadata.

Non-finite optional distances are serialized as JSON `null`, never as the
non-standard `Infinity` literal.

Custom metric values use `<metric-name>/<field>` keys (for example,
`energy/total`) so extensions cannot overwrite the standard benchmark fields.
Numeric custom fields are averaged into environment and overall summaries;
non-numeric fields remain episode-level metadata.
