# Contributing

See [CONTRIBUTING.md](https://github.com/UdvardiPeter/topokit/blob/main/CONTRIBUTING.md)
in the repository.

Quick start:

```
uv sync --all-packages
uv run --directory topokit pytest
```

Add `--extra viz --extra jax` to run the visualization and JAX tests too;
without them those two files skip. Lint and typecheck pass either way.

```
uv sync --all-packages --extra viz --extra jax
```

The full list of gates the PR workflow runs is in `CONTRIBUTING.md`.
