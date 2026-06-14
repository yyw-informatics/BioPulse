"""Placement tests against built leaderboards: assert exact ranks and neighbors.

Covers both metric directions: dkd (accuracy, maximize) and denoising (MSE, minimize)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from biopulse_lg.placement import load_boards, placement

_BENCHMARK_ROOT = Path(os.environ.get("BIOPULSE_BENCHMARK_ROOT", "../biopulse-core/benchmark_packs"))
_DKD = _BENCHMARK_ROOT / "op_label_projection_dkd"
_DENOISE = _BENCHMARK_ROOT / "op_denoising_immune"


@pytest.fixture
def dkd_boards():
    if not (_DKD / "leaderboard.yaml").exists():
        pytest.skip(f"dkd pack not built at {_DKD}")
    return load_boards(_DKD)


def test_placement_mid_board(dkd_boards):
    leaderboard, menu_ids = dkd_boards
    p = placement(0.95, "logistic_regression", leaderboard, menu_ids=menu_ids)
    assert p["dataset_id"] == "cellxgene_census/dkd"
    assert p["metric"] == "accuracy" and p["maximize"] is True
    assert p["n_methods"] == 12  # non-control methods scored on dkd
    assert p["rank"] == 8
    assert p["nearest_better"]["method_id"] == "cellmapper_linear"
    assert p["nearest_worse"]["method_id"] == "mlp"
    assert p["method_known_to_op"] is True
    assert abs(p["op_score_for_method"] - 0.9572) < 1e-3


def test_placement_top_and_bottom(dkd_boards):
    leaderboard, menu_ids = dkd_boards
    top = placement(0.99, "x", leaderboard, menu_ids=menu_ids)
    assert top["rank"] == 1 and top["nearest_better"] is None
    bottom = placement(0.0, "x", leaderboard, menu_ids=menu_ids)
    assert bottom["rank"] == bottom["n_methods"] + 1 and bottom["nearest_worse"] is None


def test_method_known_via_menu_without_score(dkd_boards):
    leaderboard, menu_ids = dkd_boards
    p = placement(0.9, "scgpt_zeroshot", leaderboard, menu_ids=menu_ids)
    assert p["method_known_to_op"] is True and p["op_score_for_method"] is None


def test_unknown_method(dkd_boards):
    leaderboard, menu_ids = dkd_boards
    assert placement(0.9, "BioPulse_custom_thing", leaderboard, menu_ids=menu_ids)["method_known_to_op"] is False


@pytest.fixture
def denoise_boards():
    if not (_DENOISE / "leaderboard.yaml").exists():
        pytest.skip(f"denoising pack not built at {_DENOISE}")
    return load_boards(_DENOISE)


def test_denoising_minimize_direction(denoise_boards):
    """Verify lower MSE ranks higher: a low MSE ranks above the no-op floor."""
    leaderboard, menu_ids = denoise_boards
    assert leaderboard["dataset"]["maximize"] is False and leaderboard["dataset"]["metric_primary"] == "mse"
    strong = placement(0.250, "magic", leaderboard, menu_ids=menu_ids)  # good-method cluster
    floor = placement(0.315, "identity", leaderboard, menu_ids=menu_ids)  # no_denoising
    assert strong["rank"] < floor["rank"]  # lower MSE => better rank
