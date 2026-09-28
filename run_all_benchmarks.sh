#!/bin/bash
echo "=== 开始基准测速 ===" > benchmark_results.txt
for size in small medium large xl; do
    for dtype in fp32 bf16; do
        for mode in forward backward full; do
            echo "Running: size=$size, dtype=$dtype, mode=$mode"
            uv run python benchmark.py --model_size $size --mode $mode --dtype $dtype >> benchmark_results.txt 2>&1
        done
    done
done
echo "=== 全部完成 ==="
