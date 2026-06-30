# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Perf scaling study: per-iteration wall time, peak RSS, AMG CG iterations.

MAINTAINER / nightly tool. Writes ``bench/baseline.json``. No assertions, no
gating (WP-2.1 soft baseline; WP-2.2 reads these numbers and adds gates). Run on
the CI platform (Linux) for comparable numbers; needs pyamg (topokit dev group)
for the larger 3D sizes.

Each case runs in its own subprocess so peak RSS is per-case, not a monotonic
process-lifetime high-water mark. Heavy cases (``heavy: True``, e.g. 60^3 at
~6.6 GB peak) are skipped unless ``TOPOKIT_BENCH_HEAVY`` is set, so the default
study fits a 7 GB CI runner; the committed baseline carries the heavy cases from
dedicated hardware. The 1M-element gate (doc 08) is aspirational / dedicated-hardware.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from topokit.optimizers import OC
from topokit.problem import Schedule, Study

from topokit_bench.problems import cantilever_3d, mbb

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "baseline.json"
BENCH_ITERS = 8

# 3D cantilevers timed over a fixed BENCH_ITERS (timing, not convergence); the 2D
# case runs to convergence for an end-to-end number.
CASES: list[dict[str, Any]] = [
    {"label": "cantilever_3d_20", "build": "c3d", "n": 20, "iters": BENCH_ITERS, "tol": 0.0},
    {"label": "cantilever_3d_40", "build": "c3d", "n": 40, "iters": BENCH_ITERS, "tol": 0.0},
    {
        "label": "cantilever_3d_60",
        "build": "c3d",
        "n": 60,
        "iters": BENCH_ITERS,
        "tol": 0.0,
        "heavy": True,
    },
    {"label": "mbb_150x50_full", "build": "mbb", "n": 0, "iters": 100, "tol": 1e-3},
]


def _build(spec: dict[str, Any]) -> Any:
    if spec["build"] == "c3d":
        n = int(spec["n"])
        return cantilever_3d(n, n, n, optimizer=OC(move=0.2))
    return mbb(150, 50, optimizer=OC(move=0.2))


def run_case(spec: dict[str, Any]) -> dict[str, Any]:
    """Run one case in-process; return its metrics (the worker adds peak RSS)."""
    problem = _build(spec)
    solver = type(problem.solver).__name__
    study = Study(
        problem,
        schedule=Schedule.single(p=3.0, max_iter=int(spec["iters"]), tol=float(spec["tol"])),
    )
    result = study.run()
    amg = getattr(problem.solver, "last_iterations", 0)
    iters = max(1, int(result.iterations))
    return {
        "label": spec["label"],
        "elements": int(problem.model.mesh.n_elements),
        "dof": int(problem.model.n_dof),
        "solver": solver,
        "iterations": int(result.iterations),
        "wall_total_s": round(float(result.timing), 4),
        "wall_per_iter_s": round(float(result.timing) / iters, 4),
        "amg_iterations": int(amg) if amg else None,
    }


def _peak_rss_kb() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":  # macOS reports bytes; Linux reports KiB
        rss //= 1024
    return int(rss)


def _pyamg_version() -> str | None:
    # the AMG iteration counts are pyamg-version-sensitive; record it for drift.
    try:
        import pyamg  # type: ignore[import-untyped]
    except ImportError:
        return None
    return str(pyamg.__version__)


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=ROOT
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip()


def _worker(label: str) -> None:
    spec = next(c for c in CASES if c["label"] == label)
    rec = run_case(spec)
    rec["peak_rss_kb"] = _peak_rss_kb()
    sys.stdout.write(json.dumps(rec))


def main() -> None:
    """Run each case in a subprocess; aggregate into the committed baseline.

    Heavy cases are skipped unless ``TOPOKIT_BENCH_HEAVY`` is set (CI runners cap
    at ~7 GB; 60^3 peaks at ~6.6 GB). The committed baseline carries the heavy
    cases from a dedicated-hardware run.
    """
    include_heavy = os.environ.get("TOPOKIT_BENCH_HEAVY", "") not in ("", "0")
    cases = [c for c in CASES if include_heavy or not c.get("heavy")]
    records: list[dict[str, Any]] = []
    for spec in cases:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", spec["label"]],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            records.append({"label": spec["label"], "error": proc.stderr[-2000:].strip()})
            print(f"{spec['label']}: FAILED (see error field)")
            continue
        try:
            rec = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            records.append({"label": spec["label"], "error": f"unparseable worker output: {exc}"})
            print(f"{spec['label']}: FAILED (unparseable worker output)")
            continue
        records.append(rec)
        print(
            f"{rec['label']}: dof={rec['dof']} {rec['solver']} "
            f"{rec['wall_per_iter_s']:.3f}s/it rss={rec['peak_rss_kb'] / 1e3:.0f}MB "
            f"amg={rec['amg_iterations']}"
        )
    baseline = {
        "meta": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pyamg": _pyamg_version(),
            "git_sha": _git_sha(),
            "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "cases": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _worker(sys.argv[2])
    else:
        main()
