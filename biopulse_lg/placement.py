"""Place an agent run on the Open Problems leaderboard for a dataset.

Consume-only: the board (``leaderboard.yaml``) and method menu (``methods.yaml``) are built in
biopulse-core from OP's published scores; no methods are rerun. Placement ranks the agent's score
against the real (non-control) methods OP scored on the same dataset and reports its nearest
neighbors. The ranking respects the metric direction from the board header (``maximize: true`` for
accuracy, ``false`` for MSE).
"""

from __future__ import annotations

from pathlib import Path

import yaml


def load_boards(benchmark: Path | str) -> tuple[dict | None, set[str]]:
    """Return (leaderboard dict or None, set of menu method ids) from the pack root."""
    base = Path(benchmark)
    lb_path = base / "leaderboard.yaml"
    leaderboard = yaml.safe_load(lb_path.read_text(encoding="utf-8")) if lb_path.exists() else None
    menu_ids: set[str] = set()
    methods_path = base / "methods.yaml"
    if methods_path.exists():
        menu = yaml.safe_load(methods_path.read_text(encoding="utf-8")) or {}
        menu_ids = {m["id"] for m in menu.get("methods", []) if m.get("id")}
    return leaderboard, menu_ids


def placement(
    agent_score: float,
    method_id: str | None,
    leaderboard: dict,
    *,
    menu_ids: set[str] | None = None,
    include_controls: bool = False,
) -> dict:
    """Rank ``agent_score`` against the leaderboard's real methods, respecting the metric direction.

    Controls are excluded unless ``include_controls`` is set.
    """
    header = leaderboard.get("dataset", {})
    metric = header.get("metric_primary", "accuracy")
    maximize = bool(header.get("maximize", True))
    entries = leaderboard.get("entries", [])
    pool = [e for e in entries if (include_controls or not e.get("is_control")) and e.get(metric) is not None]
    n = len(pool)

    def is_better(value: float) -> bool:
        return value > agent_score if maximize else value < agent_score

    better = [e for e in pool if is_better(e[metric])]
    worse = [e for e in pool if not is_better(e[metric])]
    rank = len(better) + 1  # 1-based: methods strictly better, plus the agent
    percentile = round(100.0 * len(worse) / n, 1) if n else None

    def neighbor(group):
        if not group:
            return None
        entry = min(group, key=lambda e: abs(e[metric] - agent_score))
        return {"method_id": entry["method_id"], metric: entry[metric]}

    by_id = {e["method_id"]: e for e in entries}
    known = bool(method_id) and (method_id in by_id or (menu_ids is not None and method_id in menu_ids))

    return {
        "dataset_id": header.get("dataset_id"),
        "metric": metric,
        "maximize": maximize,
        "agent_method_id": method_id,
        "agent_score": agent_score,
        "method_known_to_op": known,
        "op_score_for_method": by_id.get(method_id, {}).get(metric) if method_id else None,
        "rank": rank,
        "n_methods": n,  # real (non-control) methods scored on this dataset
        "percentile": percentile,
        "nearest_better": neighbor(better),
        "nearest_worse": neighbor(worse),
        "beats": [e["method_id"] for e in worse if e["method_id"] != method_id],
    }
