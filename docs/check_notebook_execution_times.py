#!/usr/bin/env python
"""Execute notebooks and report slow cells.

This script executes each notebook, measures per-cell wall-clock runtime,
and emits a JSON report. It can optionally fail when slow cells are found.

(this is AI-written, copilot probably using GPT codex 5.3)
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook-dir",
        default="notebooks",
        help="Directory containing notebooks to execute.",
    )
    parser.add_argument(
        "--notebook",
        action="append",
        default=[],
        help="Execute only the given notebook path (can be passed multiple times).",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help=(
            "Skip the given notebook (basename or path). Repeatable. "
            "Useful for notebooks that require infrastructure unavailable in CI."
        ),
    )
    parser.add_argument(
        "--threshold-seconds",
        type=float,
        default=30.0,
        help="Cell runtime threshold in seconds for slow-cell flagging.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="Per-cell execution timeout passed to nbclient.",
    )
    parser.add_argument(
        "--report-file",
        default="docs/notebook_timing_report.json",
        help="Path to write JSON report.",
    )
    parser.add_argument(
        "--fail-on-slow",
        action="store_true",
        help="Return non-zero exit code if any slow cells are found.",
    )
    return parser.parse_args()


def execute_with_timing(notebook_path: Path, timeout_seconds: int) -> tuple[list[dict], str | None]:
    with notebook_path.open("r", encoding="utf-8") as fh:
        notebook = nbformat.read(fh, as_version=4)

    client = NotebookClient(
        notebook,
        timeout=timeout_seconds,
        kernel_name="python3",
        allow_errors=True,
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()

    error_summary: str | None = None
    for cell_index, cell in enumerate(notebook.cells):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                error_summary = (
                    f"cell {cell_index + 1}: "
                    f"{output.get('ename', 'Error')}: {output.get('evalue', '')}"
                )
                break
        if error_summary:
            break

    timings: list[dict] = []
    for cell_index, cell in enumerate(notebook.cells):
        if cell.get("cell_type") != "code":
            continue

        execution_meta = cell.get("metadata", {}).get("execution", {})
        timestamp_keys = (
            "iopub.status.busy",
            "iopub.execute_input",
            "shell.execute_reply",
            "iopub.status.idle",
        )
        timestamps = [
            datetime.datetime.fromisoformat(execution_meta[key].replace("Z", "+00:00"))
            for key in timestamp_keys
            if key in execution_meta
        ]
        duration_seconds = (
            (max(timestamps) - min(timestamps)).total_seconds()
            if len(timestamps) >= 2
            else 0.0
        )

        timings.append(
            {
                "cell_number": cell_index + 1,
                "duration_seconds": duration_seconds,
                "source_preview": str(cell.get("source", "")).strip().splitlines()[:1],
            }
        )

    return timings, error_summary


def build_report(
    notebook_dir: Path,
    notebook_paths: list[Path],
    threshold_seconds: float,
    timeout_seconds: int,
    excluded: set[str] | None = None,
) -> dict:
    excluded = excluded or set()
    notebooks = sorted(notebook_paths) if notebook_paths else sorted(notebook_dir.glob("*.ipynb"))
    notebooks = [
        nb for nb in notebooks
        if nb.name not in excluded and str(nb) not in excluded
    ]
    report = {
        "notebook_dir": str(notebook_dir),
        "threshold_seconds": threshold_seconds,
        "excluded": sorted(excluded),
        "results": [],
        "slow_cells": [],
        "failed_notebooks": [],
    }

    for notebook_path in notebooks:
        try:
            timings, cell_error = execute_with_timing(
                notebook_path, timeout_seconds=timeout_seconds,
            )
            execution_error = cell_error
        except Exception as exc:
            timings = []
            execution_error = f"{type(exc).__name__}: {exc}"

        notebook_result = {
            "notebook": str(notebook_path),
            "cell_timings": timings,
            "total_seconds": sum(item["duration_seconds"] for item in timings),
            "error": execution_error,
        }
        report["results"].append(notebook_result)

        if execution_error:
            report["failed_notebooks"].append(
                {"notebook": str(notebook_path), "error": execution_error}
            )

        for timing in timings:
            if timing["duration_seconds"] >= threshold_seconds:
                report["slow_cells"].append(
                    {
                        "notebook": str(notebook_path),
                        "cell_number": timing["cell_number"],
                        "duration_seconds": timing["duration_seconds"],
                        "source_preview": timing["source_preview"],
                        "flag": "slow",
                    }
                )

    return report


def main() -> int:
    args = parse_args()
    notebook_dir = Path(args.notebook_dir)
    notebook_paths = [Path(path) for path in args.notebook]
    report_path = Path(args.report_file)

    report = build_report(
        notebook_dir=notebook_dir,
        notebook_paths=notebook_paths,
        threshold_seconds=args.threshold_seconds,
        timeout_seconds=args.timeout_seconds,
        excluded=set(args.exclude),
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    slow_cells = report["slow_cells"]
    failed = report["failed_notebooks"]
    print(f"Executed {len(report['results'])} notebooks.")
    if report["excluded"]:
        print(f"Excluded {len(report['excluded'])}: {', '.join(report['excluded'])}")
    print(f"Found {len(slow_cells)} slow cells with threshold {args.threshold_seconds:.1f}s.")
    print(f"{len(failed)} notebooks raised errors during execution.")

    for entry in failed:
        print(f"FAILED {entry['notebook']} :: {entry['error']}")

    for entry in slow_cells:
        preview = entry["source_preview"][0] if entry["source_preview"] else ""
        print(
            "SLOW "
            f"{entry['notebook']} "
            f"cell {entry['cell_number']} "
            f"({entry['duration_seconds']:.2f}s) :: {preview}"
        )

    if args.fail_on_slow and slow_cells:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
