#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: ./run_nccl.sh <LOCAL_IP> <REMOTE_IP>"
    echo "Example: ./run_nccl.sh 172.31.15.52 172.31.10.248"
    exit 1
fi

IP1=$1
IP2=$2

echo ">>> Verifying Ptrace is disabled..."
sudo sysctl -w kernel.yama.ptrace_scope=0

echo ">>> Running NCCL All-Reduce Benchmark over EFA..."
echo "Nodes: $IP1, $IP2"

mpirun \
--host $IP1,$IP2 \
-x FI_PROVIDER=efa \
-x NCCL_P2P_DISABLE=1 \
-x NCCL_IB_DISABLE=0 \
-x NCCL_DEBUG=INFO \
--mca plm_rsh_no_tree_spawn 1 \
~/nccl-tests/build/all_reduce_perf -b 100M -e 1G -f 2 -g 1