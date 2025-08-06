

#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <omp.h>
#include <chrono>
#include <cuda_runtime.h>

#include "kittens.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE = 16;
#define NUM_WORKERS  (1)
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)

struct matmul_globals { 
    using sub_tile_a = st_bf<BLOCK_SIZE,16>;
    using tile_gl_a =  gl<bf16,  1, 1, -1, -1, sub_tile_a>;
    using sub_tile_b = st_bf<16,BLOCK_SIZE>;
    using tile_gl_b =  gl<bf16,  1, 1, -1, -1, sub_tile_b>;
    using sub_tile_c = st_bf<BLOCK_SIZE,BLOCK_SIZE>;
    using tile_gl_c =  gl<bf16,  1, 1, -1, -1, sub_tile_c>;
    tile_gl_a A;
    tile_gl_b B; 
    tile_gl_c C;
    int N;
    int K;
};

__global__ void kernel(const __grid_constant__ matmul_globals g) {

    extern __shared__ alignment_dummy __shm[]; 
    shared_allocator al((int*)&__shm[0]);
    matmul_globals::sub_tile_a &As = al.allocate<matmul_globals::sub_tile_a>(); 
    matmul_globals::sub_tile_b &Bs = al.allocate<matmul_globals::sub_tile_b>(); 
    
    rt_bf<BLOCK_SIZE,16> A_reg;
    rt_bf<16,BLOCK_SIZE> B_reg;
    rt_bf<BLOCK_SIZE,16,ducks::rt_layout::col> B_reg_col;
    rt_fl<BLOCK_SIZE,BLOCK_SIZE> C_accum;

    int col = blockIdx.x; 
    int row = blockIdx.y; 

    zero(C_accum);
    int num_tiles = (g.K + 16 - 1) / 16;
    for (int tile = 0; tile < num_tiles; ++tile) {
        load(As, g.A, {0, 0, row, tile});
        load(Bs, g.B, {0, 0, tile, col});
        __syncthreads();
        load(A_reg, As);
        load(B_reg, Bs);
        swap_layout(B_reg_col, B_reg);
        __syncthreads();
        mma_AB(C_accum, A_reg, B_reg_col, C_accum);
        __syncthreads(); 
    }
    store(g.C, C_accum, {0, 0, row, col});
}

// launch kernel
void matmul(bf16* A, bf16* B, bf16* C, int N, int K) { 

    // global pointers
    using a_gl = matmul_globals::tile_gl_a;
    using b_gl = matmul_globals::tile_gl_b; 
    using c_gl = matmul_globals::tile_gl_c;
    a_gl  a_arg{A, nullptr, nullptr, N, K};
    b_gl  b_arg{B, nullptr, nullptr, K, N};
    c_gl  c_arg{C, nullptr, nullptr, N, N};
    matmul_globals g{a_arg, b_arg, c_arg, N, K}; 

    // launch
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);  // Watch out for requesting too many!
    unsigned long mem_size = 100000;
    cudaDeviceSynchronize();
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    kernel<<<blocks, NUM_THREADS, mem_size>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
}

#include "launch.cu"
