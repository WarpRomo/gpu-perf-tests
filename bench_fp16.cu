#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <chrono>

__global__ void fp32Math(float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = a[i] * 2.0f + 1.0f;
}

__global__ void fp16Math(__half *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // FP16 intrinsic math
    if (i < n) a[i] = __hadd(__hmul(a[i], __float2half(2.0f)), __float2half(1.0f));
}

int main() {
    int N = 10000000; // 10 Million
    float *d_fp32;
    __half *d_fp16;

    cudaMalloc(&d_fp32, N * sizeof(float));
    cudaMalloc(&d_fp16, N * sizeof(__half));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // --- FP32 Test ---
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<100; i++) fp32Math<<<gridSize, blockSize>>>(d_fp32, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double time32 = std::chrono::duration<double>(end - start).count();

    // --- FP16 Test ---
    start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<100; i++) fp16Math<<<gridSize, blockSize>>>(d_fp16, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double time16 = std::chrono::duration<double>(end - start).count();

    std::cout << "FP32 Time: " << time32 << " s" << std::endl;
    std::cout << "FP16 Time: " << time16 << " s" << std::endl;
    std::cout << "Speedup: " << time32 / time16 << "x" << std::endl;

    cudaFree(d_fp32); cudaFree(d_fp16);
    return 0;
}
