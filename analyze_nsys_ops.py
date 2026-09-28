"""Analyze kernel time or operation-level CUDA memory allocation.

Run first:
  nsys export --type sqlite --output report.sqlite report.nsys-rep
Then:
  python analyze_nsys_ops.py report.sqlite

For memory allocation attribution, first generate the profiler JSON:
  python benchmark_memory_requirements.py --mode full \
      --operation_memory_json operation_memory.json
Then run:
  python analyze_nsys_ops.py --memory_json operation_memory.json

The denominator is total GPU kernel duration, matching Nsight's GPU operation
summary.  This is deliberately different from PyTorch allocator traffic.
"""

import argparse
import json
import sqlite3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sqlite_file", nargs="?")
    p.add_argument("--memory_json", help="operation_memory.json from torch.profiler")
    args = p.parse_args()

    if args.memory_json:
        with open(args.memory_json, encoding="utf-8") as f:
            result = json.load(f)
        print("metric: self_cuda_memory_usage")
        print("rank\toperation\tallocation_mib\tshare_of_memory_allocation")
        for row in result.get("top5", []):
            print(f'{row["rank"]}\t{row["operation"]}\t'
                  f'{row["mib"]:.3f}\t{row["share_percent"]:.2f}%')
        return

    if not args.sqlite_file:
        p.error("provide sqlite_file for kernel analysis or --memory_json for memory analysis")
    con = sqlite3.connect(args.sqlite_file)
    table = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%KERNEL%'"
    ).fetchone()
    if not table:
        raise RuntimeError("No kernel activity table found in exported Nsight SQLite")
    table = table[0]
    cols = {r[1] for r in con.execute(f'PRAGMA table_info("{table}")')}
    name_col = "shortName" if "shortName" in cols else "demangledName"
    if name_col not in cols:
        name_col = "name"
    # duration is in ns in CUPTI activity exports.
    rows = con.execute(
        f'''SELECT COALESCE("{name_col}", 'UNKNOWN'), SUM("end" - "start")
            FROM "{table}" GROUP BY 1 ORDER BY 2 DESC'''
    ).fetchall()
    total = sum(r[1] for r in rows)
    print("metric: gpu_kernel_duration")
    print("rank\toperation\ttotal_us\tshare_of_gpu_kernel_time")
    for i, (name, duration) in enumerate(rows[:5], 1):
        print(f"{i}\t{name}\t{duration / 1000:.3f}\t{duration / total * 100:.2f}%")
    con.close()


if __name__ == "__main__":
    main()
