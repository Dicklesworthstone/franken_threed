# Test Evidence Specification

Conforming evidence archive layout, JSON-lines event schema, and retention rules for FrankenThreeD test runs.

## Archive Layout

```text
evidence/<bead-key>/<run-id>/events.jsonl
evidence/<bead-key>/<run-id>/summary.json
```

- `<bead-key>`: Bead key (e.g. `01.8`) or full bead identifier.
- `<run-id>`: Unique execution run identifier.

## Event Schema (`events.jsonl`)

One JSON object per line with the following fields:

- `ts_wall` (number, required): Wall-clock timestamp in milliseconds.
- `ts_app` (number | null): Logical application clock timestamp when applicable.
- `lane` (string, required): Test lane (`unit`, `integration`, `fuzz`, `e2e-functional`, `e2e-forced-new-backend`, `e2e-exact`, `bench`).
- `bead` (string, required): Bead key or identifier under test.
- `test` (string, required): Test function or case name.
- `step` (string | null): Test step identifier.
- `level` (string, required): Severity level (`info`, `warn`, `error`, `metric`).
- `owner` (string, required): Implementation owner tag (`new-rust`, `retained-js`, `exact-backend`, `general-webgpu`, etc.). Must not be empty.
- `route` (string | null): Submission route when rendering (`specialized-webgpu`, `general-webgpu`, etc.).
- `browser` (object | null): Browser environment metadata (`{name, version, os}`).
- `device_generation` (number | null): Monotonic device generation counter.
- `scene_generation` (number | null): Monotonic scene generation counter.
- `msg` (string, required): Descriptive event message.
- `data` (object | null): Bounded structured diagnostic payload.

## Summary Schema (`summary.json`)

Written upon run completion with aggregated run metadata and outcomes:

- `commit` (string): Repository commit hash at HEAD.
- `upstream_commit` (string): Pinned Three.js r186 commit (`148ef33ecb6d2502ff796d4554abd1549c95d519`).
- `bead` (string): Bead identifier.
- `run_id` (string): Execution run identifier.
- `pass` (number): Count of passing test cases.
- `fail` (number): Count of failing test cases.
- `first_failing_event` / `first_failure` (object | string | null): First failing event or diagnostic.

## Retention Rules (`KEEP/`)

- The `evidence/` hierarchy is git-ignored by default to prevent temporary run pollution.
- Runs cited in gate decisions, milestone verifications, or audits are preserved by storing them under `evidence/**/KEEP/`.
- Gitignore negates `!evidence/**/KEEP/**` so retained evidence artifacts remain tracked in version control.
