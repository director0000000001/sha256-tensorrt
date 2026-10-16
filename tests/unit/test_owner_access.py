"""The owner's passphrase (day 15, back on 2026-10-16): set from the
command line, verified in the window, never stored in the clear, and
nothing unlocked by a wrong or a missing one."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sha256.app.owner import MIN_PASSPHRASE_LENGTH, OwnerAccess


class TestOwnerAccess:
    def test_nothing_is_configured_or_unlocked_at_first(self, tmp_path: Path):
        owner = OwnerAccess(tmp_path / "local.owner.json")
        assert not owner.is_configured and not owner.is_unlocked
        assert owner.unlock("anything") is False
        assert not owner.is_unlocked

    def test_setting_stores_a_salted_key_and_unlocks(self, tmp_path: Path):
        path = tmp_path / "configs" / "local.owner.json"
        owner = OwnerAccess(path)

        owner.set_passphrase("  open sesame ")

        assert owner.is_configured and owner.is_unlocked
        record = json.loads(path.read_text(encoding="utf-8"))
        assert set(record) == {"salt", "key", "iterations"}
        assert "open sesame" not in path.read_text(encoding="utf-8")
        assert len(record["salt"]) == 32 and len(record["key"]) == 64

    def test_a_fresh_session_unlocks_with_the_right_passphrase_only(self, tmp_path: Path):
        path = tmp_path / "local.owner.json"
        OwnerAccess(path).set_passphrase("open sesame")

        owner = OwnerAccess(path)
        assert not owner.is_unlocked
        assert owner.unlock("wrong") is False and not owner.is_unlocked
        assert owner.unlock("open sesame") is True and owner.is_unlocked
        owner.lock()
        assert not owner.is_unlocked

    def test_a_short_passphrase_is_refused(self, tmp_path: Path):
        owner = OwnerAccess(tmp_path / "o.json")
        with pytest.raises(ValueError, match=str(MIN_PASSPHRASE_LENGTH)):
            owner.set_passphrase("abc")
        assert not owner.is_configured

    def test_two_salts_differ_for_the_same_passphrase(self, tmp_path: Path):
        a, b = OwnerAccess(tmp_path / "a.json"), OwnerAccess(tmp_path / "b.json")
        a.set_passphrase("open sesame")
        b.set_passphrase("open sesame")
        ra = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
        rb = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))
        assert ra["salt"] != rb["salt"] and ra["key"] != rb["key"]

    def test_a_damaged_file_counts_as_not_configured(self, tmp_path: Path):
        path = tmp_path / "o.json"
        path.write_text("{not json", encoding="utf-8")
        owner = OwnerAccess(path)
        assert not owner.is_configured and owner.unlock("x") is False
        path.write_text(json.dumps({"salt": "zz", "key": "zz"}), encoding="utf-8")
        assert owner.unlock("x") is False
