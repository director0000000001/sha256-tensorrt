"""The courtesy kill switch (day 16, block D1).

The 2026-08-15 decision, unchanged: no licence server, no payments, no
accounts -- the application is free to use -- and one thing kept, because
the owner wants to be able to turn it off: a static file on a host he
controls, read at start and then hourly, that says whether the application
may run. Three rules the decision set, all kept here:

* **It fails open.** No network, a slow host, a missing file, a file that
  is not what is expected -- every one of these means "allowed". The only
  thing that stops the application is a file that plainly says so.
* **It stays off the frame path.** The check runs on its own thread with
  a short timeout; the pipeline never waits for it. What the check finds
  is handed to the window through a callback, which decides what to do
  (stop the pipeline, say why, disable Start).
* **It is a courtesy, described as one.** Anyone with the files can
  delete the URL from the configuration; the guide says so.

The file is JSON: ``{"enabled": true}`` allows, ``{"enabled": false,
"message": "..."}`` stops with those words; or plain text, ``on`` / ``off``
on the first line. With no URL configured (the committed default) nothing
is ever fetched -- ``docs/PRIVACY.md``'s rule that the application makes
no network calls unless told to stands.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from sha256.utils.logging import get_logger

__all__ = ["KillSwitch", "KillSwitchState", "read_switch"]

_LOG = get_logger(__name__)

DEFAULT_MESSAGE = "This application has been switched off by its owner."
_USER_AGENT = "sha256/0.1"


@dataclass(frozen=True, slots=True)
class KillSwitchState:
    allowed: bool
    message: str
    reason: str  # how the state was decided, for the log


def parse_switch(body: bytes) -> KillSwitchState:
    """The state a fetched file says -- and ``allowed`` for anything that is
    not plainly a stop."""
    text = body.decode("utf-8", errors="replace").strip()
    if not text:
        return KillSwitchState(True, "", "empty file")
    try:
        data = json.loads(text)
    except ValueError:
        first = text.splitlines()[0].strip().lower()
        if first in ("off", "disabled", "stop", "false", "0"):
            return KillSwitchState(False, DEFAULT_MESSAGE, "the file says off")
        return KillSwitchState(True, "", f"text {first[:20]!r}")
    if isinstance(data, dict):
        enabled = data.get("enabled", True)
        if enabled is False or (isinstance(enabled, str) and enabled.lower() in ("false", "off")):
            message = str(data.get("message") or DEFAULT_MESSAGE)
            return KillSwitchState(False, message, "the file says disabled")
        return KillSwitchState(True, "", "the file says enabled")
    if data is False:
        return KillSwitchState(False, DEFAULT_MESSAGE, "the file says false")
    return KillSwitchState(True, "", "the file is not a switch")


def read_switch(url: str, *, timeout: float = 5.0) -> KillSwitchState:
    """Fetch and parse the switch at ``url``; every failure is ``allowed``."""
    if not url:
        return KillSwitchState(True, "", "no url")
    request = urllib.request.Request(
        url, headers={"User-Agent": _USER_AGENT, "Cache-Control": "no-cache"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if getattr(response, "status", 200) != 200:
                return KillSwitchState(True, "", f"status {response.status}")
            body = response.read(64 * 1024)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return KillSwitchState(True, "", f"unreachable: {exc}")
    return parse_switch(body)


class KillSwitch:
    """Checks the switch on its own thread: once at :meth:`start`, then every
    ``interval_seconds``; calls ``on_state`` with each result (from the
    thread -- the caller marshals to its own)."""

    def __init__(
        self,
        url: str,
        *,
        interval_seconds: float = 3600.0,
        timeout: float = 5.0,
        on_state: Callable[[KillSwitchState], None] | None = None,
        fetch: Callable[[str], KillSwitchState] | None = None,
    ) -> None:
        self._url = url.strip()
        self._interval = max(float(interval_seconds), 1.0)
        self._timeout = timeout
        self._on_state = on_state
        self._fetch = fetch or (lambda u: read_switch(u, timeout=self._timeout))
        self._stop = threading.Event()
        # Day 17: a check asked for now (every Start presses it), without
        # waiting for the interval -- the thread wakes on this.
        self._nudge = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = KillSwitchState(True, "", "not checked")

    @property
    def configured(self) -> bool:
        return bool(self._url)

    @property
    def state(self) -> KillSwitchState:
        return self._state

    def check_now(self) -> KillSwitchState:
        """One check, on the calling thread."""
        state = self._fetch(self._url) if self._url else KillSwitchState(True, "", "no url")
        self._state = state
        _LOG.info("Kill switch: %s (%s)", "allowed" if state.allowed else "STOPPED", state.reason)
        if self._on_state is not None:
            try:
                self._on_state(state)
            except Exception:
                _LOG.exception("The kill switch's listener failed")
        return state

    def start(self) -> None:
        """Begin checking in the background; nothing at all without a URL."""
        if not self._url or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="sha256-kill-switch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._nudge.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    def request_check(self) -> None:
        """Ask the background thread to check now rather than at the next
        interval; nothing without a URL or a running thread."""
        if self._url and self._thread is not None:
            self._nudge.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._nudge.clear()
            self.check_now()
            self._nudge.wait(self._interval)
