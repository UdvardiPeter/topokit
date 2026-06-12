"""Meta-tests for the import-linter layer contract.

import-linter silently ignores modules that are not listed in a contract;
the fem module was unlisted for a whole merge cycle before anyone noticed.
This test makes that omission loud.
"""

import pkgutil
import tomllib
from pathlib import Path

import topokit


def test_layers_contract_covers_all_modules() -> None:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as fh:
        cfg = tomllib.load(fh)
    layered: set[str] = set()
    for contract in cfg["tool"]["importlinter"]["contracts"]:
        for layer in contract.get("layers", []):
            layered |= {m.strip() for m in layer.split(":")}
    found = {f"topokit.{m.name}" for m in pkgutil.iter_modules(topokit.__path__)}
    missing = found - layered
    assert not missing, f"modules missing from the layers contract: {sorted(missing)}"
