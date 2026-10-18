# Contributing

## Dev setup

```
uv sync --all-packages
uv run --directory topokit pytest
```

Requires uv, which installs Python 3.12 itself. That is a 270 MB environment
and runs 524 of the 570 tests; `test_viz.py` and `test_jax.py` skip, because
PyVista and JAX are 850 MB between them and serve those two files. Lint and
typecheck pass either way.

Add them when you touch visualization or the JAX backend, which is also what
CI installs:

```
uv sync --all-packages --extra viz --extra jax
```

## Rules

- Tests first. Every change starts with a failing test.
- Every analytic gradient needs a finite-difference test. Use
  `topokit.testing.assert_gradient_matches`. PRs adding a `Response` or
  `ChainLink` without one will not be merged.
- mypy strict, ruff, and import-linter gate every PR. Module boundaries are
  enforced by `lint-imports` against the contract in `topokit/pyproject.toml`.
- New `ArrayBackend` implementations must pass the conformance suite. Subclass
  `topokit.backend.conformance.ArrayBackendConformance` in your tests and set
  `backend`. Register a module-level instance (not the class) via an entry
  point so the registry can discover it:

  ```toml
  [project.entry-points."topokit.backends"]
  mybackend = "mypkg.backend:BACKEND"
  ```
- Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`.
  Scope when useful, e.g. `feat(fields): ...`.

## DCO

Every commit needs a `Signed-off-by` line (`git commit -s`), certifying the
[Developer Certificate of Origin](https://developercertificate.org/).
There is no CLA. Contributions are MIT, permanently.

## What the PR gate runs

In this order, and nothing else. Run all of it and green locally means green in
CI; the test suite alone does not, because it leaves out the regression tier,
the wheel build and the two documentation gates.

```
# lint and format
uv run --directory topokit ruff check src tests
uv run --directory topokit ruff format --check src tests
uv run --directory benchmarks ruff check src tests scripts
uv run --directory benchmarks ruff format --check src tests scripts
# types and module boundaries
uv run --directory topokit mypy
uv run --directory benchmarks mypy
uv run --directory topokit lint-imports
# tests
uv run --directory topokit pytest
uv run --directory benchmarks pytest -m "not regression and not regression_full"
uv run --directory benchmarks pytest -m regression -n auto
# packaging and docs
uv build --package topokit
uv run --group docs mkdocs build --strict
uv run --directory benchmarks pytest -q tests/test_docs_pages.py
```

To autofix instead of checking, drop `--check` from `ruff format` and add `--fix`
to `ruff check`. The nightly adds the heavy tiers (`-m regression_full`) and the
perf gate (`uv run --directory benchmarks python scripts/bench.py --check`, Linux
x86_64 only; see `benchmarks/README.md`).
