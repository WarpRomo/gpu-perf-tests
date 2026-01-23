#!/bin/bash
set -e

echo ">>> 1. Installing RDMA/Infiniband libraries..."
sudo apt-get update
sudo apt-get install -y ibverbs-utils libibverbs-dev librdmacm-dev rdma-core iproute2

echo ">>> 2. Loading RXE Kernel Module..."
sudo modprobe rdma_rxe

echo ">>> 3. Configuring Soft-RoCE (RXE)..."
# Find the primary network interface (usually ens5 or eth0 on AWS)
NET_IF=$(ip route get 8.8.8.8 | awk '{print $5; exit}')
echo "Detected Interface: $NET_IF"

# create the rxe link
sudo rdma link add rxe0 type rxe netdev $NET_IF || echo "Link likely already exists"

echo ">>> 4. Verifying RDMA Device..."
ibv_devinfo -d rxe0

echo ">>> 5. Enforcing MTU (Jumbo Frames) for performance..."
sudo ip link set dev $NET_IF mtu 9001

echo "==========================================================="
echo "Soft-RoCE Setup Complete on device: rxe0"
echo "==========================================================="