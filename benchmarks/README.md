# benchmarks

Tier-3 reference regression for TopoKit: the 2D MBB beam and cantilever at
60x20 and 150x50, run with OC and MMA, validated against frozen reference
fields.

Run: `npx nx run benchmarks:test-regression`.

## What "matches the literature" means here

The suite asserts each optimized design against a committed reference `.npz`
(compliance within 1%, density field within a coarse `1e-2`, volume on target,
iteration band), plus OC/MMA agreement within 5%.

The reference designs *are* the 88-line method's designs: TopoKit's Q4
plane-stress element stiffness (eigenvalues), modified SIMP (`E = Emin +
x^p (E0 - Emin)`), compliance functional, and boundary conditions match the
88-line (Andreassen et al. 2011) exactly; `OC` is the clean-room 88-line
optimality-criteria update (validated in WP-1.8a); and `RadialDensityFilter`
reproduces the 88-line density filter (`ft = 2`). So a faithful single-stage
run reproduces the 88-line, and the frozen `.npz` pins that result against
drift.

No external published *compliance number* is asserted: the 88-line paper prints
none, and cross-method papers report their own method's value with the 88-line
only "visually indistinguishable" (e.g. Biyikli & To 2015) — so there is no
exact published figure to reproduce within 1%. The literature claim rests on the
method-level faithfulness above; the frozen regression carries the precision.

## Regenerating references

Reference `.npz` under `tests/data/` are regenerated only by deliberate
maintainer action: `uv run python scripts/regenerate.py` (commit with a
changelog note; never run in CI).
