# Benchmarks

TopoKit's reference regression suite pins the 2D MBB beam and cantilever, plus (nightly) the 3D cantilever and Michell cantilever, against frozen designs. Rendered versions of the same problems are in the [gallery](gallery/index.md): [MBB beam](gallery/mbb.md), [cantilever](gallery/cantilever.md), [Michell cantilever](gallery/michell.md), [3D cantilever](gallery/cantilever-3d.md).

The suite runs in two tiers:

- **Per-PR**: the 2D MBB beam and cantilever, at 60×20 and 150×50, run with both OC and MMA and checked against a committed reference `.npz` (compliance within 1%, density field within a coarse tolerance, volume on target, iteration count in-band), plus OC/MMA agreement within 5%. The reference designs are TopoKit's clean-room reproduction of the 88-line method (Andreassen et al. 2011): matching element stiffness, modified SIMP, compliance functional, boundary conditions, and density filter, so a faithful run reproduces the 88-line result and the frozen `.npz` guards it against drift.
- **Nightly**: the per-PR suite plus 3D correctness (3D cantilever, Michell) with the same reference assertions and a mirror-symmetry check on the Michell design, plus a robustness sweep over volume fraction, filter radius, and resolution asserting clean convergence rather than frozen values. A performance gate runs alongside it: per-iteration wall time, peak memory, and AMG solver iteration counts, compared against a committed baseline generated on Linux x86_64; a regression in any of them fails the job.

From a checkout, in `benchmarks/`:

    uv run pytest -m regression        # per-PR 2D suite
    uv run pytest -m regression_full   # nightly 3D + Michell + robustness suite
    python scripts/bench.py --check    # performance comparator against the committed baseline

See `benchmarks/README.md` in the repository for the full detail: what "matches the literature" means for the per-PR gate, how references are regenerated, and how the performance baseline is maintained.
