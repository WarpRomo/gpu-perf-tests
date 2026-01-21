#!/bin/bash
# =================================================================
# AWS G4dn Setup - Fix Missing Compiler
# =================================================================
set -e

echo ">>> 1. Verifying Driver..."
if ! nvidia-smi; then
    echo "ERROR: NVIDIA Driver is missing. Wrong Instance?"
    exit 1
fi

echo ">>> 2. Installing CUDA Compiler (NVCC)..."
# This package installs nvcc and standard headers without breaking the driver
sudo apt-get update
sudo apt-get install -y nvidia-cuda-toolkit

echo ">>> 3. Installing Python Dependencies..."
sudo apt-get install -y python3-pip python3-matplotlib

echo "==========================================================="
echo "SETUP COMPLETE. Run ./cuda_benchmark.sh now."
echo "==========================================================="
