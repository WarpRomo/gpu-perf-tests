# AWS G4dn CUDA Benchmarks

[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Platform](https://img.shields.io/badge/platform-AWS%20EC2%20G4dn-orange.svg)](https://aws.amazon.com/ec2/instance-types/g4/)
[![Stack](https://img.shields.io/badge/Language-C%2B%2B%20%7C%20CUDA-blue.svg)](https://developer.nvidia.com/cuda-zone)

This repository contains C++ and CUDA scripts designed to benchmark the NVIDIA T4 GPU on AWS EC2 `g4dn` instances. The suite includes tests for memory bandwidth, compute throughput, stream overlap, precision scaling, and multi-node network latency (Soft-RoCE vs TCP).

## Files

| File | Description |
| :--- | :--- |
| `setup_cuda.sh` | Verifies the NVIDIA driver, installs the `nvidia-cuda-toolkit` (nvcc), and installs Python dependencies for plotting. |
| `cuda_benchmark.sh` | The main interface script. It compiles the standard kernels (`vector`, `matrix`) and executes the benchmark loops. |
| `bench_vector_add.cu` | CUDA kernel for Vector Addition ($C = A + B$). Measures execution time for array sizes ranging from small to large. |
| `bench_matrix_mul.cu` | CUDA kernel for Matrix Multiplication. Comparisons are run against a naive CPU implementation for validation. |
| `run_advanced.sh` | Compiles and executes the single-node micro-benchmarks (`bandwidth`, `overlap`, `fp16`). |
| `soft-roce/setup_roce.sh` | Installs RDMA drivers (`ibverbs`, `rdma-core`) and configures the Soft-RoCE (RXE) network interface. |
| `soft-roce/bench_net_gpu.cu` | Network benchmark comparing TCP sockets against RDMA Verbs (Soft-RoCE) for transferring data to GPU memory. |
| `soft-roce/compile_net.sh` | Compiles the network benchmark with `ibverbs` linking. |

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

#### Network Transport (Soft-RoCE vs TCP)
Comparison of transferring 100MB buffers between two `g4dn.xlarge` instances.
*Note: On instances without hardware RDMA (like g4dn), Soft-RoCE is emulated via CPU, resulting in lower throughput than native TCP.*

| Protocol | Transport Mechanism | Throughput (Approx) |
| :--- | :--- | :--- |
| **TCP** | Standard ENA TCP/IP Stack | **~0.56 GB/s** |
| **Soft-RoCE** | RXE Driver (UDP Encapsulation) | ~0.30 GB/s |

## Prerequisites

*   **Instance:** AWS EC2 `g4dn.xlarge` (or larger).
*   **AMI:** Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Ubuntu 22.04 or 24.04).
*   **Drivers:** Pre-installed on the AMI (NVIDIA 535+).
*   **Network:** For RoCE tests, Security Groups must allow all traffic (or specific TCP/UDP ports) between instances.

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

### Advanced Micro-Benchmarks
Run the hardware-specific diagnostic tests (Bandwidth, Overlap, FP16).

```bash
chmod +x run_advanced.sh
./run_advanced.sh
```

### Soft-RoCE Network Benchmark (Multi-Node)
This test requires **two** AWS instances in the same VPC/Subnet.

1.  **Configure Security Groups:** Ensure the security group attached to both instances allows **All Traffic** from itself (Self-referencing rule) or from `0.0.0.0/0`.
2.  **Setup & Compile (Run on BOTH nodes):**
    ```bash
    cd soft-roce
    chmod +x setup_roce.sh compile_net.sh
    
    # Installs IBVerbs and loads RXE kernel module
    ./setup_roce.sh 
    
    # Compiles the binary
    ./compile_net.sh
    ```
3.  **Run the Benchmark:**
    *   **On Instance A (Server):**
        ```bash
        ./bench_net server
        ```
    *   **On Instance B (Client):**
        ```bash
        # Replace with Instance A's Private IP
        ./bench_net client 172.31.X.X
        ```