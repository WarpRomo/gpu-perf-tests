#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        std::exit(1); \
    } \
}

// CPU Naive
void cpuMatMul(const float *a, const float *b, float *c, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += a[i * N + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

// GPU Naive
__global__ void gpuMatMul(const float *a, const float *b, float *c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) return 1;
    int N = std::atoi(argv[1]);
    int iterations = std::atoi(argv[2]);

    size_t bytes = N * N * sizeof(float);
    std::vector<float> h_a(N * N, 1.0f);
    std::vector<float> h_b(N * N, 1.0f);
    std::vector<float> h_c_cpu(N * N);
    std::vector<float> h_c_gpu(N * N);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // --- CPU Bench ---
    double cpu_avg = 0.0;
    if (N <= 1024) { 
        int cpu_iters = (N > 512) ? 1 : iterations;
        auto start = std::chrono::high_resolution_clock::now();
        for(int i=0; i<cpu_iters; i++) {
            cpuMatMul(h_a.data(), h_b.data(), h_c_cpu.data(), N);
        }
        auto end = std::chrono::high_resolution_clock::now();
        cpu_avg = std::chrono::duration<double, std::micro>(end - start).count() / cpu_iters;
    }

    // --- GPU Bench ---
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    gpuMatMul<<<blocks, threads>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iterations; i++) {
        gpuMatMul<<<blocks, threads>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();
    double gpu_avg = std::chrono::duration<double, std::micro>(end - start).count() / iterations;

    std::cout << N << "," << cpu_avg << "," << gpu_avg << std::endl;

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
