#!/bin/bash
set -e

echo ">>> [1/5] Installing Dependencies (MPI, Build Tools)..."
sudo apt-get update
sudo apt-get install -y git build-essential autoconf libtool g++ \
    openmpi-bin libopenmpi-dev libnccl2 libnccl-dev

echo ">>> [2/5] Installing AWS EFA Drivers..."
# Download and run EFA installer
if [ ! -d "aws-efa-installer" ]; then
    curl -O https://s3-us-west-2.amazonaws.com/aws-efa-installer/aws-efa-installer-latest.tar.gz
    tar -xf aws-efa-installer-latest.tar.gz
fi
cd aws-efa-installer
# -y argument runs it non-interactively
sudo ./efa_installer.sh -y
cd ..

echo ">>> [3/5] Installing AWS-OFI-NCCL Plugin..."
if [ ! -d "aws-ofi-nccl" ]; then
    git clone https://github.com/aws/aws-ofi-nccl.git
fi
cd aws-ofi-nccl
./autogen.sh
./configure --with-libfabric=/opt/amazon/efa --with-cuda=/usr/local/cuda --with-mpi=/usr/lib/x86_64-linux-gnu/openmpi
make -j
sudo make install
cd ..

echo ">>> [4/5] Compiling NCCL Tests (The Benchmark)..."
if [ ! -d "nccl-tests" ]; then
    git clone https://github.com/NVIDIA/nccl-tests.git
fi
cd nccl-tests
make clean
make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda
cd ..

echo ">>> [5/5] Disabling Ptrace (Ubuntu Security Blocking EFA)..."
sudo sysctl -w kernel.yama.ptrace_scope=0

echo "=========================================================="
echo " SETUP COMPLETE."
echo " IMPORTANT: You MUST reboot this instance now to load the new kernel!"
echo " Run: sudo reboot"
echo "=========================================================="