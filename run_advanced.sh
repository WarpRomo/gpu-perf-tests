#!/bin/bash
echo ">>> Compiling..."
nvcc -arch=sm_75 -O3 bench_bandwidth.cu -o bench_bw
nvcc -arch=sm_75 -O3 bench_overlap.cu -o bench_overlap
nvcc -arch=sm_75 -O3 bench_fp16.cu -o bench_fp16

echo ""
./bench_bw
echo ""
./bench_overlap
echo ""
echo "--- Precision Test ---"
./bench_fp16
