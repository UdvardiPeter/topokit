# benchmarks

Tier-3 reference regression for TopoKit: the 2D MBB beam and cantilever at
60x20 and 150x50, plus a tri3 cantilever at 60x20, run with OC and MMA,
validated against frozen reference fields.

Run: `npx nx run benchmarks:test-regression`.

## What "matches the literature" means here

The suite asserts each optimized design against a committed reference `.npz`
(compliance within 1%, density field within a coarse `1e-2`, volume on target,
iteration band), plus OC/MMA agreement within 5%.

The reference designs *are* the 88-line method's designs: TopoKit's Q4
plane-stress element stiffness (eigenvalues), modified SIMP (`E = Emin +
x^p (E0 - Emin)`), compliance functional, and boundary conditions match the
88-line (Andreassen et al. 2011) exactly; `OC` is the clean-room 88-line
optimality-criteria update, validated against scipy on convex problems; and `RadialDensityFilter`
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

The suite also includes one unstructured case per-PR: `cantilever_tri`, the
same cantilever problem on a 60x20 tri3 mesh (deterministic quad-split
triangulation, `UnstructuredElasticity`) instead of the quad4 grid. It gets
the same frozen-reference assertions plus a dedicated cross-check
(`test_tri3_vs_quad4_cantilever_agree`) against the quad4 `cantilever_60x20`
reference. The two cases differ only in discretization — same box, supports,
load point, plane-stress material, radial filter radius, SIMP exponent — so
their compliance is close but not equal. The measured gap is 2.08%, tri3
lower: constant-strain triangles are stiffer than bilinear quads at equal
nodes, the split doubles the design variables, and its fixed diagonal biases
the design. The gap and the tolerance derived from it are recorded in that
test.

The four unstructured references (`cantilever_tri_60x20_{oc,mma}`,
`cantilever_tet4_16x8x8_oc`, `cantilever_tet10_10x5x5_oc`) were frozen on
macOS arm64 — recorded in the files — not on the Linux platform CI runs on, so
they spend part of the suite's cross-platform margin instead of leaving all of
it for genuine numerical change. The tolerances are coarse enough to absorb
that: the OC iteration is contractive, so a 1e-10 perturbation moves the final
field by 1e-10 against a `1e-2` gate. The tri3 pair is validated
cross-platform by the per-PR gate, the tet pair only by the nightly. Refreeze
them on Linux at the next deliberate baseline reset.

## Regenerating references

Reference `.npz` under `tests/data/` are regenerated only by deliberate
maintainer action: `uv run python scripts/regenerate.py` (commit with a
changelog note; never run in CI). Use `--only {2d,full,all}` to limit the set —
e.g. `--only full` regenerates just the 3D + Michell references below — and
`--case SUBSTRING` to further limit it to reference files whose name contains
the substring, e.g. `--only full --case cantilever_tet` touches only the
tet4/tet10 nightly references, leaving every other frozen reference untouched.
A `--case` matching nothing (a typo, or the wrong `--only` set) exits nonzero
instead of reporting success over an unchanged tree.

Each reference holds the frozen `density`, `compliance`, `volume` and
`iterations`, plus the provenance to read them against: `element_kind`,
`n_elements`, `optimizer`, `penal`, `max_iter`, `tol`, the `platform` it was
frozen on and the `topokit` version. Only the unstructured references carry
the provenance keys so far — the earlier baselines predate them and pick them
up on their next regeneration.

## Nightly full suite

`test-regression-full` (nightly only, `pytest -m regression_full`) adds 3D
correctness on top of the per-PR 2D gate:

- **3D cantilever** (`24x12x12`, OC) and **Michell** (`90x30`, OC + MMA) —
  frozen-reference regression with the same reference assertions as the 2D suite
  (volume, compliance, density field, iteration band) plus OC/MMA agreement, and
  a mirror-symmetry topology check on the Michell reference.
- **Unstructured 3D cantilever** — the same 3D cantilever problem on
  `UnstructuredElasticity` simplex meshes: `cantilever_tet4` (`16x8x8`, OC) and
  `cantilever_tet10` (`10x5x5`, OC; shrunk from `12x6x6` — that size measured at
  2.27x the hex `cantilever_3d_24x12x12` case's wall time, over the ~2x nightly
  runtime budget). Same frozen-reference assertions as the other nightly cases,
  no separate literature anchor.
- **Robustness sweep** (`test_robustness.py`) — `volfrac x rmin x resolution`
  over the 2D builders plus small 3D points, asserting *clean convergence*
  (finite field in `[0, 1]`, volume on target, net objective progress, no
  blow-up) rather than frozen values.

3D / Michell / unstructured references are regenerated with
`scripts/regenerate.py --only full` (maintainer-only; add `--case SUBSTRING` to
target one case). The anchor is the same as the 2D suite — method-level lineage
(element formulation, SIMP, compliance, BCs) plus the frozen `.npz` guarding
drift — not a published 3D compliance number.

## Scaling study

`bench` (local and manual, never cached) runs a scaling study — 3D cantilever at
`20^3 / 40^3` plus the 2D `150x50` full run — recording per-iteration wall time,
peak RSS, solver, and AMG CG iterations to `bench/latest.json` (gitignored;
written by every run). `bench-check` runs the same study and gates the numbers
against the committed `bench/baseline.json`; that is the target the nightly runs,
and the only one it runs — see below.

`60^3` (~6.6 GB peak) is a **heavy** case: it is skipped by default so the study
fits a 7 GB CI runner, and is included only with `TOPOKIT_BENCH_HEAVY=1` (needs
~8+ GB free). The 1M-element target is aspirational / dedicated-hardware.

## Performance baseline

`bench/baseline.json` holds the committed reference numbers. The nightly workflow
(`.github/workflows/nightly.yml`) runs `bench-check` on a daily cron against
`main` and uploads the resulting `bench/latest.json` as the `bench-latest`
artifact. It runs in a step of its own, after the test tiers rather than beside
them, because its timings *are* the measurement and a 2-vCPU runner shared with
`pytest -n auto` charges every case a large size-independent cost. A regression
fails the job: peak RSS more than 10% above the baseline, AMG iterations above the
+10%/+2 band, or per-iteration wall time more than 30% above. Wall time gets the
loose gate because the runner is a shared box whose timings vary by about 20%
between runs; the tight gates ride the near-deterministic metrics. A case whose
`elements`, `dof` or `solver` no longer matches the baseline fails too — the old
numbers describe a different problem, so comparing them proves nothing.

Numbers are only comparable within one platform, so the check refuses to compare a
baseline generated elsewhere. Regenerate after an intended performance change, and
after any **pyamg version change**: AMG iteration counts are version sensitive, so
the check drops the AMG gate whenever the baseline's pyamg differs from the run's,
and it stays off until the baseline is regenerated on the new version.

The check prints only what failed. A case the baseline does not carry, a metric
neither report carries, and a disabled AMG gate are all silent, so every line it
prints is a number that moved; `bench-check` reports how many of the baseline's
cases the run covered, which is what shows a baseline gone partially stale.

To regenerate: trigger the nightly workflow, download its `bench-latest` artifact
from a run whose *test* steps are green, check no case in it carries an `error`
field, commit it as `bench/baseline.json`, and say in the PR what changed and why
the numbers moved. When the change legitimately moves the numbers the perf-gate
step itself is red on that run — it compared the new numbers against the old
baseline — and that is expected; a red *test* step is not, and its artifact must
not be committed.

A baseline holds exactly the cases of the run that produced it, and its `meta`
describes that one platform and that one run. The nightly never sets
`TOPOKIT_BENCH_HEAVY`, so the `bench-latest` artifact covers the non-heavy cases
and `cantilever_3d_60` is consequently not gated. Rows must not be merged in from
another machine or another commit: they would sit under a `meta` that does not
describe them. Gating the heavy case means running the whole study with
`TOPOKIT_BENCH_HEAVY=1` on Linux x86_64 hardware that can hold it, matching the
runner the check compares against, and committing that entire file as the
baseline.
