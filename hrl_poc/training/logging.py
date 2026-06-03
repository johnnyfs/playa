from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_execution_record(
    record: dict[str, Any],
    trajectory: list[dict[str, Any]],
    output_dir: str | Path,
    episode_idx: int,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    trajectory_dir = output_path / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = trajectory_dir / f"episode_{episode_idx:06d}.json"
    trajectory_path.write_text(json.dumps(trajectory, separators=(",", ":")) + "\n")

    record = dict(record)
    record["trajectory_path"] = str(trajectory_path)
    records_path = output_path / "execution_records.jsonl"
    with records_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    return record
