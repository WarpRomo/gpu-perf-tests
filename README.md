# AWS G4dn CUDA Benchmarks

[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Platform](https://img.shields.io/badge/platform-AWS%20EC2%20G4dn-orange.svg)](https://aws.amazon.com/ec2/instance-types/g4/)
[![Stack](https://img.shields.io/badge/Language-C%2B%2B%20%7C%20CUDA-blue.svg)](https://developer.nvidia.com/cuda-zone)

This repository contains C++ and CUDA scripts designed to benchmark the NVIDIA T4 GPU on AWS EC2 `g4dn` instances. The suite includes tests for memory bandwidth, compute throughput, stream overlap, and precision scaling.

## Files

| File | Description |
| :--- | :--- |
| `setup_cuda.sh` | Verifies the NVIDIA driver, installs the `nvidia-cuda-toolkit` (nvcc), and installs Python dependencies for plotting. |
| `cuda_benchmark.sh` | The main interface script. It compiles the standard kernels (`vector`, `matrix`) and executes the benchmark loops. |
| `bench_vector_add.cu` | CUDA kernel for Vector Addition ($C = A + B$). Measures execution time for array sizes ranging from small to large. |
| `bench_matrix_mul.cu` | CUDA kernel for Matrix Multiplication. Comparisons are run against a naive CPU implementation for validation. |
| `run_advanced.sh` | Compiles and executes the micro-benchmarks (`bandwidth`, `overlap`, `fp16`). |
| `bench_bandwidth.cu` | Measures PCIe throughput using Pageable vs. Pinned (`cudaMallocHost`) memory. |
| `bench_overlap.cu` | Demonstrates and measures the latency impact of using CUDA Streams to overlap compute with memory transfer. |
| `bench_fp16.cu` | Compares the throughput of FP32 `float` operations against FP16 `__half` intrinsic operations. |
| `generate_plot.py` | Python script that parses CSV output from the benchmarks and generates log-log performance graphs. |

## Benchmark Results

**Vector Addition Performance**
<div align="center">
  <img width="1000" src="./vector_benchmark.png" alt="Vector Benchmark Graph" />
</div>

**Matrix Multiplication Performance**
<div align="center">
  <img width="1000" src="./matrix_benchmark.png" alt="Matrix Benchmark Graph" />
</div>

## Micro-Benchmark Results

Sample output from `run_advanced.sh` on a `g4dn.xlarge` instance.

#### PCIe Bandwidth (Host to Device)
Comparing standard `malloc` (Pageable) against `cudaMallocHost` (Pinned/DMA) transfers.

| Transfer Size | Pageable Speed | Pinned Speed |
| :--- | :--- | :--- |
| **10 MB** | 5.68 GB/s | 6.25 GB/s |
| **100 MB** | 6.15 GB/s | 6.27 GB/s |
| **500 MB** | 6.20 GB/s | 6.27 GB/s |

#### Precision Scaling (FP32 vs FP16)
Measuring the throughput advantage of Tensor Cores using `__half` intrinsics.

| Precision | Execution Time | Speedup |
| :--- | :--- | :--- |
| **FP32** (Float) | 0.0328 s | 1.0x |
| **FP16** (Half) | 0.0180 s | **1.82x** |

#### Stream Overlap
*   **Pipeline Latency:** 9430.69 µs
*   **Efficiency:** The CPU successfully executed ~500 Billion operations *simultaneously* while the GPU was processing the kernel, demonstrating effective latency hiding via CUDA Streams.

## Prerequisites

*   **Instance:** AWS EC2 `g4dn.xlarge` (or larger).
*   **AMI:** Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Ubuntu 22.04 or 24.04).
*   **Drivers:** Pre-installed on the AMI (NVIDIA 535+).

## Installation

Run the setup script to install the compiler and plotting libraries.

```bash
chmod +x setup_cuda.sh
./setup_cuda.sh
```

## Usage

### Standard Benchmarks
Run the Vector Addition or Matrix Multiplication suites. These scripts automatically compile the binaries and generate visualization images.

```bash
# Syntax: ./cuda_benchmark.sh <type> <limit> <iterations>

# Run Vector Addition
./cuda_benchmark.sh vector 10000000 50

# Run Matrix Multiplication
./cuda_benchmark.sh matrix 2048 20
```

**Output:**
*   `vector_benchmark.png` / `vector_results.csv`
*   `matrix_benchmark.png` / `matrix_results.csv`

### Advanced Micro-Benchmarks
Run the hardware-specific diagnostic tests (Bandwidth, Overlap, FP16).

```bash
chmod +x run_advanced.sh
./run_advanced.sh
```
