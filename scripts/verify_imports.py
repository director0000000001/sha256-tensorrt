"""Import every module in the package and report failures.

Catches the class of breakage that tests miss: a module nothing currently
imports (a new backend, a panel behind a feature flag) can carry a syntax error,
a circular import, or a stale reference indefinitely without a single test
failing.

    uv run python scripts/verify_imports.py

Exits non-zero if anything fails to import.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import sys

# UI modules construct Qt types at import time; without this the check fails on
# a machine or CI runner with no display, for reasons unrelated to the code.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sha256


def main() -> int:
    module_names = sorted(
        module.name for module in pkgutil.walk_packages(sha256.__path__, "sha256.")
    )
    failures: list[tuple[str, str]] = []

    for name in module_names:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append((name, f"{type(exc).__name__}: {exc}"))

    print(f"sha256 {sha256.__version__}")
    print(f"modules discovered: {len(module_names)}")

    for name in module_names:
        marker = "FAIL" if any(name == failed for failed, _ in failures) else "ok"
        print(f"  [{marker:>4}] {name}")

    if failures:
        print(f"\n{len(failures)} module(s) failed to import:")
        for name, error in failures:
            print(f"  {name}: {error}")
        return 1

    print("\nAll modules import cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
