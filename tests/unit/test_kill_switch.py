"""The courtesy kill switch (day 16, D1): a file that plainly says off
stops; everything else -- no URL, no host, a missing or strange file --
allows; the check runs off the frame path and tells its listener."""

from __future__ import annotations

import http.server
import json
import threading
import time

import pytest

from sha256.app.kill_switch import (
    DEFAULT_MESSAGE,
    KillSwitch,
    KillSwitchState,
    parse_switch,
    read_switch,
)


class TestParse:
    @pytest.mark.parametrize(
        "body",
        [
            b"",
            b"on",
            b"enabled",
            b'{"enabled": true}',
            b"{}",
            b"[1, 2]",
            b"hello there",
            b"\xff\xfe",
        ],
    )
    def test_anything_that_is_not_a_stop_allows(self, body):
        assert parse_switch(body).allowed

    @pytest.mark.parametrize("body", [b"off", b"OFF\nmore", b"disabled", b"stop"])
    def test_plain_text_off(self, body):
        state = parse_switch(body)
        assert not state.allowed and state.message == DEFAULT_MESSAGE

    def test_json_off_carries_its_message(self):
        state = parse_switch(json.dumps({"enabled": False, "message": "Back on Monday."}).encode())
        assert not state.allowed and state.message == "Back on Monday."
        assert not parse_switch(b'{"enabled": "off"}').allowed
        assert not parse_switch(b"false").allowed
        assert parse_switch(b'{"enabled": false, "message": ""}').message == DEFAULT_MESSAGE


class _Server(http.server.ThreadingHTTPServer):
    body: bytes = b""
    status: int = 200
    delay: float = 0.0


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        server = self.server
        assert isinstance(server, _Server)
        time.sleep(server.delay)
        self.send_response(server.status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(server.body)

    def log_message(self, *args) -> None:  # quiet
        return None


@pytest.fixture
def server():
    srv = _Server(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()


def url_of(srv: _Server) -> str:
    return f"http://127.0.0.1:{srv.server_address[1]}/switch.json"


class TestRead:
    def test_no_url_is_allowed_without_a_request(self):
        assert read_switch("").allowed

    def test_a_served_off_stops_and_a_served_on_allows(self, server):
        server.body = json.dumps({"enabled": False, "message": "Stopped."}).encode()
        state = read_switch(url_of(server), timeout=2.0)
        assert not state.allowed and state.message == "Stopped."
        server.body = b'{"enabled": true}'
        assert read_switch(url_of(server), timeout=2.0).allowed

    def test_a_missing_file_a_dead_host_and_a_slow_host_fail_open(self, server):
        server.status = 404
        server.body = b"off"
        assert read_switch(url_of(server), timeout=2.0).allowed
        assert read_switch("http://127.0.0.1:9/nothing", timeout=0.5).allowed
        server.status = 200
        server.delay = 1.5
        state = read_switch(url_of(server), timeout=0.3)
        assert state.allowed and "unreachable" in state.reason


class TestSwitch:
    def test_unconfigured_never_starts_a_thread(self):
        heard: list[KillSwitchState] = []
        switch = KillSwitch("", on_state=heard.append)
        switch.start()
        assert not switch.configured and switch._thread is None
        assert switch.check_now().allowed and heard[-1].reason == "no url"
        switch.stop()

    def test_the_listener_hears_each_check_and_a_failing_listener_is_survived(self):
        heard: list[KillSwitchState] = []
        answers = iter(
            [
                KillSwitchState(True, "", "on"),
                KillSwitchState(False, "Enough.", "off"),
                KillSwitchState(True, "", "on again"),
            ]
        )

        def listen(state: KillSwitchState) -> None:
            heard.append(state)
            if len(heard) == 2:
                raise RuntimeError("listener broke")

        switch = KillSwitch(
            "http://example.invalid/switch",
            interval_seconds=0.05,
            on_state=listen,
            fetch=lambda url: next(answers),
        )
        switch.start()
        deadline = time.monotonic() + 3.0
        while len(heard) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        switch.stop()
        assert [s.allowed for s in heard[:3]] == [True, False, True]
        assert switch.state.allowed


class TestRequestCheck:
    """Day 17: every Start asks the switch again, without waiting for the
    interval; nothing without a URL or a thread."""

    def test_a_request_wakes_the_thread_before_the_interval(self):
        heard: list[KillSwitchState] = []
        switch = KillSwitch(
            "http://example.invalid/switch",
            interval_seconds=60.0,
            on_state=heard.append,
            fetch=lambda url: KillSwitchState(True, "", "on"),
        )
        switch.start()
        deadline = time.monotonic() + 3.0
        while len(heard) < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(heard) == 1
        switch.request_check()
        deadline = time.monotonic() + 3.0
        while len(heard) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        switch.stop()
        assert len(heard) >= 2

    def test_no_url_or_no_thread_ignores_the_request(self):
        switch = KillSwitch("", on_state=None)
        switch.request_check()
        assert not switch._nudge.is_set()
        idle = KillSwitch("http://example.invalid/switch")
        idle.request_check()  # never started
        assert not idle._nudge.is_set()
