"""Offline tests for harness levels L1-L4: policy resolution, netguard enforcement, prompt
augmentation, and folding the network audit log into the Agent Process plane. The guard raises at
getaddrinfo before any connection, so blocked hosts never reach the wire."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from biopulse_lg.middleware import RunRecorder
from biopulse_lg.netguard import NetPolicy
from biopulse_lg.run import _augment_for_level, _fold_network_events, _resolve_level
from biopulse_lg.score import score_process
from biopulse_lg.tools import execute_python

_BENCHMARK_ROOT = Path(os.environ.get("BIOPULSE_BENCHMARK_ROOT", "../biopulse-core/benchmark_packs"))
_DKD = _BENCHMARK_ROOT / "op_label_projection_dkd"


@pytest.fixture
def pack() -> Path:
    if not (_DKD / "blacklist.yaml").exists():
        pytest.skip(f"dkd pack not built at {_DKD}")
    return _DKD


def test_resolve_levels(pack: Path):
    net1, menu1, research1 = _resolve_level("L1", pack)
    assert net1.mode == "block_all" and menu1 is None and research1 is False

    net2, menu2, research2 = _resolve_level("L2", pack)
    assert net2.mode == "blacklist" and "github.com" in net2.blacklist and menu2 is None and research2 is False

    net3, menu3, research3 = _resolve_level("L3", pack)
    assert net3.mode == "blacklist" and menu3 and "knn" in menu3 and research3 is False

    _, menu4, research4 = _resolve_level("L4", pack)
    assert menu4 and research4 is True


def test_unknown_level_raises(pack: Path):
    with pytest.raises(ValueError):
        _resolve_level("L9", pack)


def test_l1_blocks_network_and_folds_into_process(tmp_path: Path):
    recorder = RunRecorder()
    log = tmp_path / "audit.jsonl"
    code = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('github.com', 443), timeout=3); print('CONNECTED')\n"
        "except OSError:\n"
        "    print('blocked')\n"
    )
    out = execute_python(tmp_path, recorder, code, net=NetPolicy("block_all", ()), net_log=log)
    assert "blocked" in out
    assert json.loads(log.read_text().splitlines()[0])["blocked"] is True

    _fold_network_events(log, recorder)
    process = score_process(recorder, finished=True, max_iterations=6)
    assert process["n_blocked_fetches"] >= 1 and process["n_web_fetches"] == 0


def test_l2_blocks_blacklisted_host(tmp_path: Path):
    recorder = RunRecorder()
    log = tmp_path / "audit.jsonl"
    code = (
        "import socket\n"
        "for h in ('github.com', 'example.org'):\n"
        "    try:\n"
        "        socket.getaddrinfo(h, 80)\n"
        "    except OSError:\n"
        "        pass\n"
        "print('done')\n"
    )
    execute_python(tmp_path, recorder, code, net=NetPolicy("blacklist", ("github.com",)), net_log=log)
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    blocked = [r for r in rows if r["blocked"]]
    assert blocked and all("github.com" in r["host"] for r in blocked)  # only blacklisted host blocked

    _fold_network_events(log, recorder)
    process = score_process(recorder, finished=True, max_iterations=6)
    assert process["n_blocked_fetches"] >= 1


def test_augment_for_level():
    base = "TASK STATEMENT"
    with_menu = _augment_for_level(base, "- knn (KNN): nearest neighbours", False)
    assert "method menu" in with_menu.lower() and "knn" in with_menu
    with_research = _augment_for_level(base, None, True)
    assert "fitness analysis" in with_research.lower()
    assert _augment_for_level(base, None, False) == base  # L1/L2 leave message unchanged
