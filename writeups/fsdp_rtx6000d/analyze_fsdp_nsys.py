"""Summarize and visualize FSDP AllGather/GEMM overlap from Nsight SQLite."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _duration(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def _overlap(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> int:
    i = j = total = 0
    while i < len(left) and j < len(right):
        total += max(
            0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0])
        )
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def main() -> None:
    connection = sqlite3.connect(ROOT / "fsdp_trace_clean.sqlite")
    processes = [
        (int(global_pid), int(pid), name)
        for global_pid, pid, name in connection.execute(
            "SELECT globalPid, pid, name FROM PROCESSES"
        )
    ]
    kernels = []
    query = """SELECT deviceId, globalPid, start, end, StringIds.value
               FROM CUPTI_ACTIVITY_KIND_KERNEL
               JOIN StringIds ON demangledName = StringIds.id"""
    for device, global_pid, start, end, name in connection.execute(query):
        lower_name = name.lower()
        if "allgather" in lower_name:
            category = "AllGather"
        elif "reducescatter" in lower_name:
            category = "ReduceScatter"
        elif "cutlass" in lower_name or "gemm" in lower_name:
            category = "GEMM"
        else:
            category = "other"
        kernels.append(
            (int(device), int(global_pid), int(start), int(end), category)
        )

    forward_ranges = []
    for start, end, global_tid in connection.execute(
        "SELECT start, end, globalTid FROM NVTX_EVENTS "
        "WHERE text = 'forward' ORDER BY start"
    ):
        matches = [
            process
            for process in processes
            if 0 <= int(global_tid) - process[0] < 1_000_000
        ]
        if matches:
            global_pid, pid, _name = max(matches, key=lambda process: process[0])
            forward_ranges.append(
                (global_pid, pid, int(start), int(end))
            )

    # Use the last forward range for each worker; the profile includes one
    # measured iteration after the initial allocation-measurement iteration.
    selected = []
    for global_pid in sorted({row[0] for row in forward_ranges}):
        selected.append(
            max(
                (row for row in forward_ranges if row[0] == global_pid),
                key=lambda row: row[3],
            )
        )

    results = []
    figure, axis = plt.subplots(figsize=(13, 4.6), layout="constrained")
    colors = {"AllGather": "#2563eb", "GEMM": "#16a34a"}
    tick_positions = []
    tick_labels = []

    for row_index, (global_pid, pid, start, end) in enumerate(selected):
        device = next(
            device
            for device, process_pid, *_ in kernels
            if process_pid == global_pid
        )
        events = [
            event
            for event in kernels
            if event[0] == device
            and event[1] == global_pid
            and event[3] > start
            and event[2] < end
        ]
        intervals = {}
        for category in ("AllGather", "GEMM", "ReduceScatter"):
            clipped = [
                (max(event[2], start), min(event[3], end))
                for event in events
                if event[4] == category
            ]
            intervals[category] = _merge(clipped)

        overlap = _overlap(intervals["AllGather"], intervals["GEMM"])
        all_gather_ns = _duration(intervals["AllGather"])
        results.append(
            {
                "pid": pid,
                "gpu": device,
                "forward_ms": (end - start) / 1e6,
                "all_gather_union_ms": all_gather_ns / 1e6,
                "gemm_union_ms": _duration(intervals["GEMM"]) / 1e6,
                "all_gather_gemm_overlap_ms": overlap / 1e6,
                "all_gather_overlap_percent": 100 * overlap / max(all_gather_ns, 1),
                "reduce_scatter_union_ms": _duration(
                    intervals["ReduceScatter"]
                )
                / 1e6,
            }
        )

        base_y = row_index * 1.0
        for lane, category in enumerate(("AllGather", "GEMM")):
            bars = [
                ((event_start - start) / 1e6, (event_end - event_start) / 1e6)
                for event_start, event_end in intervals[category]
            ]
            axis.broken_barh(
                bars,
                (base_y + lane * 0.34, 0.27),
                facecolors=colors[category],
                label=category if row_index == 0 else None,
            )
            tick_positions.append(base_y + lane * 0.34 + 0.135)
            tick_labels.append(f"GPU {device} {category}")

    axis.set_yticks(tick_positions)
    axis.set_yticklabels(tick_labels)
    axis.invert_yaxis()
    axis.set_xlabel("Time from start of measured forward (ms)")
    axis.set_title("Nsight Systems: FSDP XL forward AllGather and GEMM kernels")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="upper right")
    figure.savefig(ROOT / "nsys_fsdp_forward_overlap.png", dpi=180)
    (ROOT / "nsys_overlap.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
