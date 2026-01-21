#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

// Dummy heavy kernel
__global__ void heavyMath(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for(int k=0; k<1000; k++) val = sinf(val) * cosf(val); // Burn cycles
        data[idx] = val;
    }
}

int main() {
    int N = 1000000;
    size_t bytes = N * sizeof(float);
    float *h_a, *d_a;
    
    cudaMallocHost(&h_a, bytes); // Must be pinned for Async
    cudaMalloc(&d_a, bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto start = std::chrono::high_resolution_clock::now();

    // 1. Launch Async Copy
    cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream);
    
    // 2. Launch Kernel (in same stream, queues after copy)
    heavyMath<<<N/256, 256, 0, stream>>>(d_a, N);

    // 3. Launch Async Copy Back
    cudaMemcpyAsync(h_a, d_a, bytes, cudaMemcpyDeviceToHost, stream);

    // 4. CPU does work WHILE GPU is working (Overlap)
    long cpu_work = 0;
    for(int i=0; i<1000000; i++) cpu_work += i; 

    cudaStreamSynchronize(stream); // Wait for GPU
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double, std::micro>(end - start).count();
    std::cout << "Async Pipeline Time: " << elapsed << " us" << std::endl;
    std::cout << "(CPU did work: " << cpu_work << " during GPU execution)" << std::endl;

    cudaFree(d_a); cudaFreeHost(h_a); cudaStreamDestroy(stream);
    return 0;
}
