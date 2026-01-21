#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cmath>

// --- Error Handling Macro ---
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        std::exit(1); \
    } \
}

// CPU Implementation
void cpuAdd(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
}

// GPU Kernel
__global__ void gpuAdd(const float *a, const float *b, float *c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./bench_vector <size> <iters>" << std::endl;
        return 1;
    }
    
    int N = std::atoi(argv[1]);
    int iterations = std::atoi(argv[2]);
    size_t bytes = N * sizeof(float);

    // Host Alloc
    std::vector<float> h_a(N, 1.0f);
    std::vector<float> h_b(N, 2.0f);
    std::vector<float> h_c_cpu(N);
    std::vector<float> h_c_gpu(N);

    // Device Alloc
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // Copy to Device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // --- CPU Bench ---
    double cpu_total = 0.0;
    for(int i=0; i<iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        cpuAdd(h_a.data(), h_b.data(), h_c_cpu.data(), N);
        auto end = std::chrono::high_resolution_clock::now();
        cpu_total += std::chrono::duration<double, std::micro>(end - start).count();
    }
    double cpu_avg = cpu_total / iterations;

    // --- GPU Bench ---
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Warmup
    gpuAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double gpu_total = 0.0;
    for(int i=0; i<iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        gpuAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        gpu_total += std::chrono::duration<double, std::micro>(end - start).count();
    }
    double gpu_avg = gpu_total / iterations;

    // --- VERIFICATION ---
    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    for(int i=0; i<N; i++) {
        if (std::abs(h_c_cpu[i] - h_c_gpu[i]) > 1e-4) {
            errors++;
            if (errors < 5) std::cerr << "MISMATCH index " << i << std::endl;
        }
    }

    if (errors > 0) {
        std::cerr << "VERIFICATION FAILED" << std::endl;
        std::cout << N << "," << cpu_avg << ",-1.0" << std::endl;
    } else {
        std::cout << N << "," << cpu_avg << "," << gpu_avg << std::endl;
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
