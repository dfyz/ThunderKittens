#include <openssl/sha.h>
#include <openssl/evp.h>
#include <iomanip>
#include <sstream>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <omp.h>
#include <chrono>

#include <vector>

#include <cuda_runtime.h>

using my_dtype = __nv_bfloat16; 

// Courtesy of Claude
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

#define NAIVE_SUM 1

void cpu_gemm(__nv_bfloat16* a, __nv_bfloat16* b, __nv_bfloat16* c, int M, int N, int K) {
    #pragma omp parallel for collapse(2) // otherwise the CPU version takes for everrrrrr
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
#if NAIVE_SUM
            for (int k = 0; k < K; k++) {
                sum += __bfloat162float(a[i * K + k]) * __bfloat162float(b[k * N + j]);
            }
#else
            struct Addend {
                int index;
                float value;
            };
            std::array<Addend, 16> addends;

            for (int k = 0; k < K; k += 16) {
                for (int kk = 0; kk < 16; kk++) {
                    float a_val = __bfloat162float(a[i * K + k + kk]);
                    float b_val = __bfloat162float(b[(k + kk) * N + j]);
                    addends[kk] = {
                        .index = kk,
                        .value = a_val * b_val,
                    };
                }
                std::sort(addends.begin(), addends.end(), [](const Addend& a, const Addend& b) {
                    return std::abs(a.value) > std::abs(b.value);
                });
                for (int kk = 0; kk < 16; kk++) {
                    int idx = addends[kk].index;
                    sum = std::fma(
                        __bfloat162float(a[i * K + k + idx]),
                        __bfloat162float(b[(k + idx) * N + j]),
                        sum
                    );
                }
            }
#endif
            c[i * N + j] = __float2bfloat16(sum);
        }
    }
}

int run_benchmark(size_t M, size_t N, size_t K) {
    cudaError_t cudaStatus;
    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";

    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    __nv_bfloat16 *h_C = new __nv_bfloat16[M * N];
    __nv_bfloat16 *h_C_ref = new __nv_bfloat16[M * N];
    std::cout << "Allocated host memory" << std::endl;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    std::mt19937 perm_gen(43);
    std::vector<size_t> perm(M);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), perm_gen);
    std::cout << "Applied permutation:";
    for (int i = 0; i < 20; i++) std::cout << " " << perm[i];
    std::cout << "..." << std::endl;

    // Initialize matrices with random values
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            h_A[perm[i] * K + j] = dis(gen);
        }
    }
    for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);
    std::cout << "Initialized matrices" << std::endl;

    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, K*N*sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, M*N*sizeof(__nv_bfloat16));
    // Check for CUDA errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }
    std::cout << "Allocated device memory" << std::endl;

    // Convert to __nv_bfloat16
    __nv_bfloat16 *h_A_bf16 = new __nv_bfloat16[M * K];
    __nv_bfloat16 *h_B_bf16 = new __nv_bfloat16[K * N];
    for (int i = 0; i < M * K; ++i) h_A_bf16[i] = __float2bfloat16(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_bf16[i] = __float2bfloat16(h_B[i]);

    // Perform CPU matrix multiplication for reference
    if(true) cpu_gemm(h_A_bf16, h_B_bf16, h_C_ref, M, N, K);
    std::cout << "Performed CPU matrix multiplication" << std::endl;

    // Copy matrices to device
    cudaMemcpy(d_A, h_A_bf16, M*K*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_bf16, K*N*2, cudaMemcpyHostToDevice);
    std::cout << "Copied matrices to device" << std::endl;
    printf("\n");

    // Launch kernel
    for(int i = 0; i < 2; i++) { // warmup
        matmul(d_A, d_B, d_C, M, K);
    }
    // Start timing
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    constexpr int ITERS = 1;
    for(int i = 0; i < ITERS; i++) {
        matmul(d_A, d_B, d_C, M, K);
    }
    cudaDeviceSynchronize();

    // End timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> diff = end - start;
    double useconds = diff.count() * 1e6 / ITERS;

    // Calculate TFLOPs
    double flops = double(2.0) * M * N * K; // 2 FLOPs per multiply-add
    double tflops = (flops / useconds) / 1e6;
    std::cout << "Avg Kernel execution time: " << useconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";
    
    // Check for CUDA errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    // Copy result back to host
    __nv_bfloat16 *h_C_bf16 = new __nv_bfloat16[M * N];
    cudaMemcpy(h_C_bf16, d_C, M*N*2, cudaMemcpyDeviceToHost);
    std::cout << "Copied result back to host" << std::endl;

    // Convert result back to float for comparison
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            h_C[i * N + j] = h_C_bf16[perm[i] * N + j];
        }
    }
    std::cout << "Converted result back to float" << std::endl;

    // Check result
    float max_error = 0.0f;
    int error_count = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float act = __bfloat162float(h_C[i * N + j]);
            float ref = __bfloat162float(h_C_ref[perm[i] * N + j]);
            float error = std::abs(act - ref);
            if( error > 0 ) {
                if(error_count < 20) std::cout << "Error at row " << i << " col " << j << ": " << act << " != " << ref << " (ref)" << std::endl;
                else if(error_count == 21) std::cout << "Too many errors to show them all.\n";
                error_count++;
            }
            max_error = std::max(max_error, error);
        }
    }

    std::cout << "SHA256: " << sha256(reinterpret_cast<uint8_t*>(h_C), M * N * sizeof(__nv_bfloat16)) << std::endl;
    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Error count: " << error_count << std::endl;
    std::cout << "Total count: " << int(N * N) << std::endl;

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    delete[] h_A_bf16;
    delete[] h_B_bf16;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

int main() {
    int N;
    // N = 4096;
    N = 5376;
    run_benchmark(N, N, N);
    return 0;
}

