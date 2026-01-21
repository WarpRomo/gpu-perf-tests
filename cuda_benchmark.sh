#!/bin/bash

# =========================================================
# CUDA Master Benchmark Interface
# Usage: ./cuda_benchmark.sh <type> <limit> <iters>
# Types: vector, matrix
# =========================================================

TYPE=$1
LIMIT=$2
ITERS=$3

# Default Args
if [ -z "$TYPE" ]; then TYPE="vector"; fi
if [ -z "$LIMIT" ]; then 
    if [ "$TYPE" == "vector" ]; then LIMIT=10000000; else LIMIT=2048; fi
fi
if [ -z "$ITERS" ]; then ITERS=50; fi

# 1. Compile
echo ">>> Compiling Suite (CUDA)..."
# sm_75 is the architecture for NVIDIA T4 (g4dn instances)
if [ ! -f bench_vector ]; then
    nvcc -arch=sm_75 -O3 bench_vector_add.cu -o bench_vector
fi
if [ ! -f bench_matrix ]; then
    nvcc -arch=sm_75 -O3 bench_matrix_mul.cu -o bench_matrix
fi

# 2. Check Python Deps
if ! python3 -c "import matplotlib" &> /dev/null; then
    echo "WARNING: Matplotlib not found. Installing..."
    sudo apt-get install -y python3-matplotlib
fi

# 3. Setup Loop Params
RESULTS_FILE="${TYPE}_results.csv"
echo "" > $RESULTS_FILE

echo "==========================================================="
echo " Starting CUDA $TYPE benchmark"
echo " Max Size: $LIMIT | Iterations: $ITERS"
echo " Saving raw data to: $RESULTS_FILE"
echo "==========================================================="

if [ "$TYPE" == "vector" ]; then
    for BASE in 10 100 1000 10000 100000 1000000; do
        STEP=$(( BASE / 2 )) 
        for (( i=BASE; i<BASE*10; i+=STEP )); do
            if [ "$i" -gt "$LIMIT" ]; then break; fi
            
            # Run Binary
            OUTPUT=$(./bench_vector $i $ITERS)
            echo "$OUTPUT"
            echo "$OUTPUT" >> $RESULTS_FILE
        done
    done

elif [ "$TYPE" == "matrix" ]; then
    for (( i=64; i<=LIMIT; i+=64 )); do
        CURRENT_ITERS=$ITERS
        if [ "$i" -gt 1000 ]; then CURRENT_ITERS=5; fi

        # Run Binary
        OUTPUT=$(./bench_matrix $i $CURRENT_ITERS)
        echo "$OUTPUT"
        echo "$OUTPUT" >> $RESULTS_FILE
    done
else
    echo "Error: Type must be 'vector' or 'matrix'"
    exit 1
fi

# 4. Generate Graph
echo ">>> Generating Visualization..."
IMG_FILE="${TYPE}_benchmark.png"
TITLE="CUDA ${TYPE^} Benchmark (NVIDIA T4)"

# We can reuse the same python script, just need to make sure the file exists
if [ ! -f generate_plot.py ]; then
    echo "Error: generate_plot.py missing."
    exit 1
fi

python3 generate_plot.py "$RESULTS_FILE" "$IMG_FILE" --title "$TITLE"

echo "==========================================================="
echo " DONE."
echo " Raw Data: $RESULTS_FILE"
echo " Graph:    $IMG_FILE"
echo "==========================================================="
