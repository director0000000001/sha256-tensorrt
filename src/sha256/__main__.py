"""Application entry point.

Run with ``python -m sha256`` or the installed ``sha256`` command.

Deliberately thin: it parses arguments, builds the context, and hands off. All
wiring lives in :mod:`sha256.app.bootstrap`, and it stays importable without Qt
so ``--check`` works in a headless CI environment.
"""

from __future__ import annotations

import argparse
import sys

from sha256 import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sha256",
        description="sha256 — local-first real-time AI avatar (foundation build).",
    )
    parser.add_argument("--version", action="version", version=f"sha256 {__version__}")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Wire the application, print a diagnostic report, and exit without a UI.",
    )
    parser.add_argument(
        "--set-owner",
        action="store_true",
        help="Ask for the owner's passphrase (twice, unseen) and store it in "
        "configs/local.owner.json; Login in the application takes it from then on.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override the configured log level.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Process entry point. Returns a process exit code."""
    args = _build_parser().parse_args(argv)
    if args.set_owner:
        return _set_owner()

    # Imported here, not at module scope: `--check` must work on a machine or CI
    # runner where Qt cannot open a display.
    from sha256.app.bootstrap import build_application

    overrides = {"log_level": args.log_level} if args.log_level else None
    # 2026-10-09: the start takes twenty seconds, and minutes the first time
    # on a machine (the fast engine built for its graphics card), so a small
    # window says what is happening before the main one exists. Not for
    # --check, which must run where Qt cannot open a display.
    splash = None
    if not args.check:
        from sha256.ui.splash import StartupSplash

        splash = StartupSplash.open()
    progress = splash.say if splash is not None else None
    try:
        if overrides:
            from sha256.config.loader import load_settings
            from sha256.config.paths import AppPaths

            paths = AppPaths.resolve()
            context = build_application(
                settings=load_settings(paths, overrides=overrides), paths=paths, progress=progress
            )
        else:
            context = build_application(progress=progress)
    finally:
        if splash is not None:
            splash.close()

    try:
        if args.check:
            _print_diagnostics(context)
            return 0

        from sha256.ui.application import run_ui

        return run_ui(context)
    finally:
        context.shutdown()


def _set_owner() -> int:
    """The owner's passphrase set from the terminal, unseen, twice (day 15,
    back on 2026-10-16): the window never sets one."""
    import getpass

    from sha256.app.owner import OWNER_FILENAME, OwnerAccess
    from sha256.config.paths import AppPaths

    first = getpass.getpass("The owner's passphrase: ")
    again = getpass.getpass("Once more: ")
    if first != again:
        print("The two entries differ; nothing was set.")
        return 2
    owner = OwnerAccess(AppPaths.resolve().configs / OWNER_FILENAME)
    try:
        owner.set_passphrase(first)
    except ValueError as exc:
        print(exc)
        return 2
    print(f"Passphrase set and kept in {owner.path}.")
    return 0


def _print_diagnostics(context: object) -> None:
    """Print a wiring report. Used by ``--check`` and by CI smoke tests."""
    from sha256.app.bootstrap import ApplicationContext

    assert isinstance(context, ApplicationContext)

    print(f"sha256 {__version__} — foundation build")
    print(f"  environment    : {context.settings.environment}")
    print(f"  app root       : {context.paths.root}")
    print(f"  development    : {context.paths.is_development}")
    print(f"  avatar library : {context.paths.avatars}")
    print()

    print("Registered backends:")
    for category, ids in context.registries.summary().items():
        print(f"  {category:<15}: {', '.join(ids)}")
    print()

    avatars = context.avatars.statuses()
    print(f"Avatars discovered: {len(avatars)}")
    for status in avatars:
        marker = "*" if status.avatar_id == context.avatars.active_id else " "
        detail = f" ({status.error})" if status.error else ""
        print(f"  {marker} {status.avatar_id:<16} {status.name:<24} [{status.state.value}]{detail}")

    print()
    print(f"Pipeline state: {context.controller.state.value}")


if __name__ == "__main__":
    sys.exit(main())
