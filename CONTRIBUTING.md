# Contributing

## Dev setup

```
npm ci
uv sync --all-packages
npx nx run-many -t lint typecheck boundaries test test-fd
```

The last command is what CI runs (affected-only on PRs). Green locally means green in CI.

Requires Node >= 20.19 and uv. Python 3.12 is installed by uv automatically.

## Rules

- Tests first. Every change starts with a failing test.
- Every analytic gradient needs a finite-difference test. Use
  `topokit.testing.assert_gradient_matches`. PRs adding a `Response` or
  `ChainLink` without one will not be merged.
- mypy strict, ruff, and import-linter gate every PR. Module boundaries are
  enforced by the `boundaries` target.
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
There is no CLA. Contributions are LGPL-2.1-or-later, permanently.

## Common commands

| What | Command |
|---|---|
| Everything CI checks | `npx nx run-many -t lint typecheck boundaries test test-fd` |
| Unit tests only | `npx nx run topokit:test` |
| FD gradient tests | `npx nx run topokit:test-fd` |
| Autofix lint + format | `npx nx run topokit:format` |
| Build the wheel | `npx nx run topokit:build` |
| Build the docs | `npx nx run docs:docs-build` |

Nx caches results. Second runs are instant. `npx nx affected -t test` runs only
what your change touched.
