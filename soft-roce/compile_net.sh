#!/bin/bash
echo "Compiling Network Benchmark..."
nvcc -o bench_net bench_net_gpu.cu -libverbs
chmod +x bench_net
echo "Done."