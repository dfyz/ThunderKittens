#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <openssl/sha.h>
#include <openssl/evp.h>
#include <iomanip>
#include <sstream>

// Error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

std::string sha256(const uint8_t* data, size_t size) {
    unsigned char hash[SHA256_DIGEST_LENGTH];

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);
    EVP_DigestUpdate(ctx, data, size);
    EVP_DigestFinal_ex(ctx, hash, nullptr);
    EVP_MD_CTX_free(ctx);

    // Convert to hex string
    std::stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    return ss.str();
}

// Function to convert float to __nv_bfloat16
void convertFloatToBF16(const std::vector<float>& src, std::vector<__nv_bfloat16>& dst) {
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = __float2bfloat16(src[i]);
    }
}

// Function to perform mixed-precision matrix multiplication using cuBLAS
void matrixMultiplyMixedPrecision(cublasHandle_t handle, const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int m, int n, int k) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Both matrices are row-major, hence transposed from the point of view of cuBLAS
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k,
                 &alpha, A, CUDA_R_16BF, m,
                 B, CUDA_R_16BF, k,
                 &beta, C, CUDA_R_16BF, m,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// Benchmark function
void benchmark(int m, int n, int k) {
    if (m != n || n != k) {
        std::cerr << "uh oh" << std::endl;
        return;
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate host memory
    std::vector<float> h_A_float(m * k);
    std::vector<float> h_B_float(k * n);
    std::vector<float> h_C(m * n);
    std::vector<__nv_bfloat16> h_A_bf16(m * k);
    std::vector<__nv_bfloat16> h_B_bf16(k * n);
    std::vector<__nv_bfloat16> h_C_bf16(k * n);

    // Initialize matrices
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // Initialize matrices with random values
    for (int i = 0; i < m * k; ++i) h_A_float[i] = dis(gen);
    for (int i = 0; i < k * n; ++i) h_B_float[i] = dis(gen);

    // Convert to BF16
    convertFloatToBF16(h_A_float, h_A_bf16);
    convertFloatToBF16(h_B_float, h_B_bf16);

    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, m * k * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_B, k * n * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_C, m * n * sizeof(__nv_bfloat16)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A_bf16.data(), m * k * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B_bf16.data(), k * n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // Warm-up run
    for (int i = 0; i < 10; ++i) {
        matrixMultiplyMixedPrecision(handle, d_A, d_B, d_C, m, n, k);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    const int NUM_ITERATIONS = 10;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        matrixMultiplyMixedPrecision(handle, d_A, d_B, d_C, m, n, k);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time = milliseconds / NUM_ITERATIONS;

    // Calculate GFLOPS
    double gflops = (2.0 * m * n * k) / (avg_time * 1e9);

    std::cout << "Matrix size: " << m << "x" << n << "x" << k << std::endl;
    std::cout << "Average time: " << avg_time << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl << std::endl;

    cudaMemcpy(h_C_bf16.data(), d_C, m*n*2, cudaMemcpyDeviceToHost);

    // Transpose the col-major result to row-major
    for (int i = 0; i < m; i++) {
        for (int j = i + 1; j < n; j++) {
            std::swap(h_C_bf16[i * n + j], h_C_bf16[j * n + i]);
        }
    }

    std::cout << "SHA256: " << sha256(reinterpret_cast<uint8_t*>(h_C_bf16.data()), m * n * sizeof(__nv_bfloat16)) << std::endl;

    // Clean up
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // Benchmark different matrix sizes
    // benchmark(1024, 1024, 1024);
    // benchmark(2048, 2048, 2048);
    // benchmark(4096, 4096, 4096);
    // benchmark(8192, 8192, 8192);
    benchmark(5376, 5376, 5376);

    return 0;
}