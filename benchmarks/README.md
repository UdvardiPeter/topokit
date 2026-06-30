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

No external *compliance number* is asserted in this per-PR gate. The 88-line
paper prints none, and cross-method papers report only their own method's value
with the 88-line "visually indistinguishable" (e.g. Biyikli & To 2015). An
*independent* end-to-end cross-check — running topopt.py or a clean-room top88
and matching its compliance within 1% — is achievable (the method matches) and
is deliberately deferred to the forthcoming public benchmark suite, which
compares accuracy against the literature. Here the claim rests on method-level
faithfulness — the exact element-level KE match above, the clean-room 88-line
OC, and the radial filter reproducing the 88-line filter — plus the
independently-validated FE / optimizer / filter pieces; the frozen regression
carries the precision and guards drift.

## Regenerating references

Reference `.npz` under `tests/data/` are regenerated only by deliberate
maintainer action: `uv run python scripts/regenerate.py` (commit with a
changelog note; never run in CI). Use `--only {2d,full,all}` to limit the set —
e.g. `--only full` regenerates just the 3D + Michell references below.

## Nightly full suite

`test-regression-full` (nightly only, `pytest -m regression_full`) adds 3D
correctness on top of the per-PR 2D gate:

- **3D cantilever** (`24x12x12`, OC) and **Michell** (`90x30`, OC + MMA) —
  frozen-reference regression with the same reference assertions as the 2D suite
  (volume, compliance, density field, iteration band) plus OC/MMA agreement, and
  a mirror-symmetry topology check on the Michell reference.
- **Robustness sweep** (`test_robustness.py`) — `volfrac x rmin x resolution`
  over the 2D builders plus small 3D points, asserting *clean convergence*
  (finite field in `[0, 1]`, volume on target, net objective progress, no
  blow-up) rather than frozen values.

3D / Michell references are regenerated with `scripts/regenerate.py --only full`
(maintainer-only). The anchor is the same as the 2D suite — method-level lineage
(hex8 element, SIMP, compliance, BCs) plus the frozen `.npz` guarding drift — not
a published 3D compliance number.

## Perf baseline

`bench` (nightly only, never cached) runs a scaling study — 3D cantilever at
`20^3 / 40^3 / 60^3` plus the 2D `150x50` full run — recording per-iteration wall
time, peak RSS, solver, and AMG CG iterations to `bench/baseline.json` (committed;
refreshed by `uv run python scripts/bench.py`). It is a **soft baseline**: no
assertions, no gating. Hard perf budgets land in WP-2.2. The 1M-element target is
aspirational / dedicated-hardware; the committed study stops at CI-feasible sizes.

The nightly workflow (`.github/workflows/nightly.yml`) runs these tiers on a daily
cron against `main` and uploads `baseline.json` as an artifact for drift
inspection.
