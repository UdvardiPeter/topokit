# benchmarks

Tier-3 reference regression for TopoKit: the 2D MBB beam and cantilever at
60x20 and 150x50, run with OC and MMA, validated against frozen reference
fields and a published 88-line compliance figure.

Run: `npx nx run benchmarks:test-regression`.

Reference `.npz` under `tests/data/` are regenerated only by deliberate
maintainer action: `uv run python scripts/regenerate.py` (commit with a
changelog note; never run in CI).
