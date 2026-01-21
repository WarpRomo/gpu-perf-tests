#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        std::exit(1); \
    } \
}

void bench(size_t bytes, bool pinned) {
    float *h_data, *d_data;
    
    // Allocation
    if (pinned) {
        CUDA_CHECK(cudaMallocHost(&h_data, bytes)); // PINNED
    } else {
        h_data = (float*)malloc(bytes);             // PAGEABLE
    }
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    // Warmup
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // Measure H2D (Host to Device)
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<50; i++) {
        CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_sec = std::chrono::duration<double>(end - start).count() / 50.0;
    double gb_per_sec = (bytes / 1e9) / elapsed_sec;

    std::cout << (pinned ? "Pinned  " : "Pageable") << " | Size: " << bytes/1024/1024 << " MB | Speed: " << gb_per_sec << " GB/s" << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    if (pinned) cudaFreeHost(h_data);
    else free(h_data);
}

int main() {
    std::cout << "--- PCIe Bandwidth Test (NVIDIA T4) ---" << std::endl;
    // Test 10MB, 100MB, 500MB
    std::vector<size_t> sizes = {10*1024*1024, 100*1024*1024, 500*1024*1024};
    
    for (size_t s : sizes) {
        bench(s, false); // Pageable
        bench(s, true);  // Pinned
        std::cout << "-----------------------------------" << std::endl;
    }
    return 0;
}
