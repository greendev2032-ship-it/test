#ifndef CUDA_KANGAROO_CUH
#define CUDA_KANGAROO_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "CUDAMath.h"
#include "CUDAStructures.h"

// Max jump points we keep in constant memory
#define MAX_JUMP_POINTS 256

extern __constant__ uint64_t c_SpX[MAX_JUMP_POINTS * 4];
extern __constant__ uint64_t c_SpY[MAX_JUMP_POINTS * 4];

#define DP_RECORD_TAME 0
#define DP_RECORD_WILD 1

struct DPRecord {
    uint32_t type;
    uint32_t threadId;
    uint64_t X[4];
    uint64_t distance[4];
};

__device__ __forceinline__ void add256_device(uint64_t a[4], const uint64_t b[4]) {
    unsigned __int128 cur = (unsigned __int128)a[0] + b[0];
    a[0] = (uint64_t)cur;
    uint64_t carry = (uint64_t)(cur >> 64);
    for (int i = 1; i < 4; ++i) {
        cur = (unsigned __int128)a[i] + b[i] + carry;
        a[i] = (uint64_t)cur;
        carry = (uint64_t)(cur >> 64);
    }
}

__global__ void kernel_add_pubkey(uint64_t* K_x, uint64_t* K_y, 
    uint64_t tX0, uint64_t tX1, uint64_t tX2, uint64_t tX3, 
    uint64_t tY0, uint64_t tY1, uint64_t tY2, uint64_t tY3, int N)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
    
    uint64_t x1[4] = {K_x[gid*4], K_x[gid*4+1], K_x[gid*4+2], K_x[gid*4+3]};
    uint64_t y1[4] = {K_y[gid*4], K_y[gid*4+1], K_y[gid*4+2], K_y[gid*4+3]};
    uint64_t px_i[4] = {tX0, tX1, tX2, tX3};
    uint64_t py_i[4] = {tY0, tY1, tY2, tY3};

    uint64_t dx[4]; ModSub256(dx, px_i, x1);
    uint64_t inv[5] = {dx[0], dx[1], dx[2], dx[3], 0};
    _ModInv(inv);

    uint64_t dy[4]; ModSub256(dy, py_i, y1);
    uint64_t lam[4]; _ModMult(lam, dy, inv);

    uint64_t x3[4]; _ModSqr(x3, lam);
    ModSub256(x3, x3, x1);
    ModSub256(x3, x3, px_i);

    uint64_t s[4], y3[4]; 
    ModSub256(s, x1, x3); 
    _ModMult(y3, s, lam); 
    ModSub256(y3, y3, y1);

    for(int i=0; i<4; ++i) { K_x[gid*4+i] = x3[i]; K_y[gid*4+i] = y3[i]; }
}

__launch_bounds__(256, 1)
__global__ void kernel_kangaroo_jump(
    uint64_t* K_x, 
    uint64_t* K_y,
    uint64_t* K_dist,
    uint32_t* K_type,
    uint32_t threadsTotal,
    uint32_t num_k_per_thread,
    uint32_t slices_per_launch,
    uint32_t pow2Jmax,
    uint32_t dp_mask_lsb,
    DPRecord* d_dp_buffer,
    unsigned int* d_dp_count,
    unsigned int max_dps
) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= threadsTotal) return;

    // We hardcode NUM_K array limits inside. Ensure host calls with num_k_per_thread <= 16
    const int NUM_K = 8; 

    uint64_t x[NUM_K][4];
    uint64_t y[NUM_K][4];
    uint64_t dist[NUM_K][4];
    uint32_t type[NUM_K];

    for(int k = 0; k < num_k_per_thread; ++k) {
        uint64_t idx = gid * num_k_per_thread + k;
        type[k] = K_type[idx];
        for(int i=0; i<4; ++i) {
            x[k][i] = K_x[idx*4 + i];
            y[k][i] = K_y[idx*4 + i];
            dist[k][i] = K_dist[idx*4 + i];
        }
    }

    uint64_t diff_x[NUM_K][4];
    uint64_t subp[NUM_K][4];
    uint64_t inverse[5];

    for (uint32_t slice = 0; slice < slices_per_launch; ++slice) {
        uint32_t pw[NUM_K];
        uint64_t SpX[NUM_K][4];
        uint64_t SpY[NUM_K][4];

        for(int k=0; k < num_k_per_thread; ++k) {
            pw[k] = (uint32_t)(x[k][0] % pow2Jmax);
            for(int i=0; i<4; ++i) {
                SpX[k][i] = c_SpX[pw[k]*4 + i];
                SpY[k][i] = c_SpY[pw[k]*4 + i];
            }
            ModSub256(diff_x[k], SpX[k], x[k]); 
        }

        for(int i=0; i<4; ++i) subp[num_k_per_thread-1][i] = diff_x[num_k_per_thread-1][i];
        
        for (int i = num_k_per_thread - 2; i >= 0; --i) {
            _ModMult(subp[i], subp[i+1], diff_x[i]);
        }

        for(int i=0; i<4; ++i) inverse[i] = subp[0][i];
        inverse[4] = 0;
        _ModInv(inverse);

        uint64_t inv_accum[4] = {inverse[0], inverse[1], inverse[2], inverse[3]};

        for (int k = 0; k < num_k_per_thread; ++k) {
            uint64_t dx_inv_k[4];

            if (k < num_k_per_thread - 1) {
                _ModMult(dx_inv_k, subp[k+1], inv_accum);
                _ModMult(inv_accum, inv_accum, diff_x[k]);
            } else {
                for(int i=0; i<4; ++i) dx_inv_k[i] = inv_accum[i];
            }

            uint64_t py_minus_y[4];
            ModSub256(py_minus_y, SpY[k], y[k]);

            uint64_t lam[4];
            _ModMult(lam, py_minus_y, dx_inv_k);

            uint64_t x3[4];
            _ModSqr(x3, lam);
            ModSub256(x3, x3, x[k]);
            ModSub256(x3, x3, SpX[k]);

            uint64_t s[4], y3[4];
            ModSub256(s, x[k], x3);
            _ModMult(y3, s, lam);
            ModSub256(y3, y3, y[k]);

            for(int i=0; i<4; ++i) { x[k][i] = x3[i]; y[k][i] = y3[i]; }

            uint64_t d_inc[4] = {0,0,0,0};
            d_inc[pw[k]/64] = 1ULL << (pw[k]%64); 
            add256_device(dist[k], d_inc);

            if ((x[k][0] & dp_mask_lsb) == 0) {
                unsigned int dp_idx = atomicAdd(d_dp_count, 1);
                if (dp_idx < max_dps) {
                    d_dp_buffer[dp_idx].type = type[k];
                    d_dp_buffer[dp_idx].threadId = gid * num_k_per_thread + k;
                    for(int i=0; i<4; ++i) {
                        d_dp_buffer[dp_idx].X[i] = x[k][i];
                        d_dp_buffer[dp_idx].distance[i] = dist[k][i];
                    }
                }
            }
        }
    }

    for(int k = 0; k < num_k_per_thread; ++k) {
        uint64_t idx = gid * num_k_per_thread + k;
        for(int i=0; i<4; ++i) {
            K_x[idx*4 + i] = x[k][i];
            K_y[idx*4 + i] = y[k][i];
            K_dist[idx*4 + i] = dist[k][i];
        }
    }
}

#endif
