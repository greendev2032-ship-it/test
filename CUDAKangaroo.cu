#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <chrono>
#include <thread>
#include <unordered_map>
#include <vector>

#include "CUDAKangaroo.cuh"

#include "CUDAUtils.h"

extern bool hexToLE64(const std::string& h_in, uint64_t w[4]);
extern std::string formatHex256(const uint64_t limbs[4]);
extern std::string formatCompressedPubHex(const uint64_t X[4], const uint64_t Y[4]);
extern std::string formatCompressedPubHex(const uint64_t X[4], const uint64_t Y[4]);
extern __global__ void scalarMulKernelBase(const uint64_t* scalars_in, uint64_t* outX, uint64_t* outY, int N);

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

int getPow2Jmax(long double optimalmeanjumpsize) {
    long double sumjumpsize = 0;
    for(int i = 1; i < 256; ++i) {
        sumjumpsize += std::pow(2.0, i - 1);
        long double now_mean = sumjumpsize / i;
        long double next_mean = (sumjumpsize + std::pow(2.0, i)) / i;
        if (optimalmeanjumpsize - now_mean <= next_mean - optimalmeanjumpsize) {
            return i;
        }
    }
    return 255;
}

int runKangaroo(const std::string& range_hex, const std::string& pubkey_hex, uint32_t a, uint32_t b, uint32_t slices) {
    std::cout << "======== Phase-1: Kangaroo Parameters =================\n";

    size_t colon_pos = range_hex.find(':');
    std::string start_hex = range_hex.substr(0, colon_pos);
    std::string end_hex   = range_hex.substr(colon_pos + 1);

    uint64_t range_start[4]{0}, range_end[4]{0};
    hexToLE64(start_hex, range_start);
    hexToLE64(end_hex, range_end);
    
    // W = end - start
    uint64_t W[4];
    uint64_t borrow = 0;
    for(int i=0; i<4; i++) {
        uint64_t bi = range_start[i] + borrow;
        if (range_end[i] < bi) { W[i] = (uint64_t)((((unsigned __int128)1)<<64) + range_end[i] - bi); borrow = 1; }
        else { W[i] = range_end[i] - bi; borrow = 0; }
    }

    long double W_ld = ld_from_u256(W);
    long double Wsqrt = std::sqrt(W_ld);
    long double midJsize = Wsqrt / 2.0;

    int pow2W = (int)std::round(std::log2(W_ld));
    int pow2Jmax = getPow2Jmax(midJsize);
    int pow2dp = (pow2W / 2) - 2;
    if (pow2dp < 0) pow2dp = 0;
    uint32_t dp_mask_lsb = (1ULL << pow2dp) - 1; 

    // M = start + W/2
    uint64_t halfW[4] = {W[0], W[1], W[2], W[3]};
    for(int i=3; i>=0; --i) {
        if (i>0) halfW[i-1] |= (halfW[i] & 1) ? (1ULL<<63) : 0;
        halfW[i] >>= 1;
    }
    uint64_t M[4];
    unsigned __int128 cc_M = (unsigned __int128)range_start[0] + halfW[0];
    M[0] = (uint64_t)cc_M; uint64_t cy_M = (uint64_t)(cc_M >> 64);
    for (int i=1; i<4; i++) { cc_M = (unsigned __int128)range_start[i] + halfW[i] + cy_M; M[i] = (uint64_t)cc_M; cy_M = (uint64_t)(cc_M >> 64); }
    
    unsigned __int128 cc = (unsigned __int128)M[0] + halfW[0]; 
    M[0] = (uint64_t)cc; uint64_t cy = (uint64_t)(cc >> 64);
    for (int i=1; i<4; i++) { cc = (unsigned __int128)M[i] + halfW[i] + cy; M[i] = (uint64_t)cc; cy = (uint64_t)(cc >> 64); }

    std::cout << std::left << std::setw(20) << "W (Keyspace)" << " : ~2^" << pow2W << "\n";
    std::cout << std::left << std::setw(20) << "pow2dp" << " : " << pow2dp << "\n";
    std::cout << std::left << std::setw(20) << "pow2Jmax" << " : " << pow2Jmax << "\n";

    uint64_t pubX[4]{0}, pubY[4]{0};
    if (pubkey_hex.length() == 130) {
        hexToLE64(pubkey_hex.substr(2, 64), pubX);
        hexToLE64(pubkey_hex.substr(66, 64), pubY);
    } else {
        std::cerr << "Uncompressed pubkey expected for Kangaroo, strictly 130 hex characters.\n";
        return 1;
    }

    uint32_t threadsTotal = 262144; 
    uint32_t KANGAROOS_PER_THREAD = 8;
    uint32_t TOTAL_KANGAROOS = threadsTotal * KANGAROOS_PER_THREAD;

    uint64_t* h_start_scalars;
    cudaHostAlloc(&h_start_scalars, TOTAL_KANGAROOS * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
    
    uint64_t* h_dist;
    cudaHostAlloc(&h_dist, TOTAL_KANGAROOS * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
    
    // TAME (First half) and WILD (Second half)
    for(uint32_t k=0; k < TOTAL_KANGAROOS/2; k++) {
        uint64_t rT[4] = { (uint64_t)rand() | ((uint64_t)rand()<<32), (uint64_t)rand() | ((uint64_t)rand()<<32), 0, 0 }; 
        uint64_t rW[4] = { (uint64_t)rand() | ((uint64_t)rand()<<32), (uint64_t)rand() | ((uint64_t)rand()<<32), 0, 0 }; 
        // Mask offset to be smaller than W. If W < 2^128, rT[2]=rT[3]=0 is mostly fine. We'll mask to pow2W/2 bits roughly.
        // It's just an offset.
        int wsqrt_bits = pow2W / 2;
        if (wsqrt_bits < 64) { rT[0] &= (1ULL << wsqrt_bits) - 1; rT[1]=0; rT[2]=0; rT[3]=0; }
        else if (wsqrt_bits < 128) { rT[1] &= (1ULL << (wsqrt_bits-64)) - 1; rT[2]=0; rT[3]=0; }
        
        if (wsqrt_bits < 64) { rW[0] &= (1ULL << wsqrt_bits) - 1; rW[1]=0; rW[2]=0; rW[3]=0; }
        else if (wsqrt_bits < 128) { rW[1] &= (1ULL << (wsqrt_bits-64)) - 1; rW[2]=0; rW[3]=0; }

        uint64_t mT[4] = {M[0], M[1], M[2], M[3]};
        unsigned __int128 cc_mT = (unsigned __int128)mT[0] + rT[0];
        mT[0] = (uint64_t)cc_mT; uint64_t cy_mT = (uint64_t)(cc_mT >> 64);
        for (int i=1; i<4; i++) { cc_mT = (unsigned __int128)mT[i] + rT[i] + cy_mT; mT[i] = (uint64_t)cc_mT; cy_mT = (uint64_t)(cc_mT >> 64); }

        for(int i=0; i<4; i++) {
            h_start_scalars[k*4 + i] = mT[i];
            h_dist[k*4 + i] = rT[i]; // Tame distance initialized to rT

            h_start_scalars[(TOTAL_KANGAROOS/2 + k)*4 + i] = rW[i]; 
            h_dist[(TOTAL_KANGAROOS/2 + k)*4 + i] = rW[i]; // Wild distance initialized to rW
        }
    }

    uint64_t *d_S, *d_Kx, *d_Ky, *d_Kdist;
    uint32_t *d_Ktype;
    cudaMalloc(&d_S, TOTAL_KANGAROOS * 4 * sizeof(uint64_t));
    cudaMalloc(&d_Kx, TOTAL_KANGAROOS * 4 * sizeof(uint64_t));
    cudaMalloc(&d_Ky, TOTAL_KANGAROOS * 4 * sizeof(uint64_t));
    cudaMalloc(&d_Kdist, TOTAL_KANGAROOS * 4 * sizeof(uint64_t));
    cudaMalloc(&d_Ktype, TOTAL_KANGAROOS * sizeof(uint32_t));

    cudaMemcpy(d_S, h_start_scalars, TOTAL_KANGAROOS * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Kdist, h_dist, TOTAL_KANGAROOS * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    std::vector<uint32_t> h_types(TOTAL_KANGAROOS, DP_RECORD_TAME);
    for(size_t i=TOTAL_KANGAROOS/2; i<TOTAL_KANGAROOS; i++) h_types[i] = DP_RECORD_WILD;
    cudaMemcpy(d_Ktype, h_types.data(), TOTAL_KANGAROOS * sizeof(uint32_t), cudaMemcpyHostToDevice);

    int blocks_scal = (TOTAL_KANGAROOS + 255) / 256;
    scalarMulKernelBase<<<blocks_scal, 256>>>(d_S, d_Kx, d_Ky, TOTAL_KANGAROOS);
    cudaDeviceSynchronize();

    kernel_add_pubkey<<<blocks_scal / 2 + 1, 256>>>(d_Kx + (TOTAL_KANGAROOS/2)*4, d_Ky + (TOTAL_KANGAROOS/2)*4, 
        pubX[0], pubX[1], pubX[2], pubX[3], pubY[0], pubY[1], pubY[2], pubY[3], TOTAL_KANGAROOS/2);
    cudaDeviceSynchronize();

    uint64_t* h_scalars_Sp;
    cudaHostAlloc(&h_scalars_Sp, 256 * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
    memset(h_scalars_Sp, 0, 256 * 4 * sizeof(uint64_t));
    for (int k = 0; k < 256; ++k) h_scalars_Sp[k*4 + (k/64)] = 1ULL << (k%64);
    
    uint64_t *d_scalars_Sp, *d_Gx_Sp, *d_Gy_Sp;
    cudaMalloc(&d_scalars_Sp, 256 * 4 * sizeof(uint64_t));
    cudaMalloc(&d_Gx_Sp, 256 * 4 * sizeof(uint64_t));
    cudaMalloc(&d_Gy_Sp, 256 * 4 * sizeof(uint64_t));
    cudaMemcpy(d_scalars_Sp, h_scalars_Sp, 256 * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);

    scalarMulKernelBase<<<1, 256>>>(d_scalars_Sp, d_Gx_Sp, d_Gy_Sp, 256);
    cudaDeviceSynchronize();

    uint64_t h_Gx_Sp[1024], h_Gy_Sp[1024];
    cudaMemcpy(h_Gx_Sp, d_Gx_Sp, 256 * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Gy_Sp, d_Gy_Sp, 256 * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpyToSymbol(c_SpX, h_Gx_Sp, 256 * 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(c_SpY, h_Gy_Sp, 256 * 4 * sizeof(uint64_t));

    unsigned int MAX_DPS = 10000;
    DPRecord* d_dp_buffer;
    unsigned int* d_dp_count;
    cudaMalloc(&d_dp_buffer, MAX_DPS * sizeof(DPRecord));
    cudaMalloc(&d_dp_count, sizeof(unsigned int));
    cudaMemset(d_dp_count, 0, sizeof(unsigned int));

    struct DPInfo { int type; uint64_t distance[4]; };
    std::unordered_map<uint64_t, DPInfo> CPU_DP_MAP;
    
    std::cout << "======== Phase-2: Execute Kangaroos ==================\n";
    int blocks = threadsTotal / 256;
    bool found = false;
    uint64_t total_jumps = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    while(!found) {
        kernel_kangaroo_jump<<<blocks, 256>>>(d_Kx, d_Ky, d_Kdist, d_Ktype, threadsTotal, KANGAROOS_PER_THREAD, slices, pow2Jmax, dp_mask_lsb, d_dp_buffer, d_dp_count, MAX_DPS);
        cudaDeviceSynchronize();
        total_jumps += TOTAL_KANGAROOS * slices;

        unsigned int dps = 0;
        cudaMemcpy(&dps, d_dp_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (dps > MAX_DPS) dps = MAX_DPS;

        if (dps > 0) {
            DPRecord* host_dps = new DPRecord[dps];
            cudaMemcpy(host_dps, d_dp_buffer, dps * sizeof(DPRecord), cudaMemcpyDeviceToHost);
            cudaMemset(d_dp_count, 0, sizeof(unsigned int));

            for(size_t i=0; i<dps; i++) {
                uint64_t key = host_dps[i].X[0];
                if (CPU_DP_MAP.find(key) != CPU_DP_MAP.end()) {
                    auto existing = CPU_DP_MAP[key];
                    if (existing.type != host_dps[i].type) {
                        found = true;
                        std::cout << "\nCOLLISION DETECTED! Tame and Wild paths converged.\n";
                        uint64_t tame_dist[4], wild_dist[4];
                        if (existing.type == DP_RECORD_TAME) {
                            for(int j=0; j<4; j++) tame_dist[j] = existing.distance[j];
                            for(int j=0; j<4; j++) wild_dist[j] = host_dps[i].distance[j];
                        } else {
                            for(int j=0; j<4; j++) wild_dist[j] = existing.distance[j];
                            for(int j=0; j<4; j++) tame_dist[j] = host_dps[i].distance[j];
                        }
                        
                        uint64_t target_key_calc[4];
                        // M + D_T = PrivateKey + D_W => PrivateKey = M + D_T - D_W
                        uint64_t MT[4] = {M[0], M[1], M[2], M[3]};
                        unsigned __int128 cc_MT = (unsigned __int128)MT[0] + tame_dist[0];
                        MT[0] = (uint64_t)cc_MT; uint64_t cy_MT = (uint64_t)(cc_MT >> 64);
                        for (int j=1; j<4; j++) { cc_MT = (unsigned __int128)MT[j] + tame_dist[j] + cy_MT; MT[j] = (uint64_t)cc_MT; cy_MT = (uint64_t)(cc_MT >> 64); }
                        
                        // We need sub256 for MT - D_W. Let's use CPU arithmetic.
                        uint64_t borrow = 0;
                        for(int j=0; j<4; ++j) {
                            uint64_t b_val = wild_dist[j] + borrow;
                            if (MT[j] < b_val) {
                                target_key_calc[j] = (uint64_t)((((unsigned __int128)1)<<64) + MT[j] - b_val);
                                borrow = 1;
                            } else {
                                target_key_calc[j] = MT[j] - b_val;
                                borrow = 0;
                            }
                        }
                        
                        std::cout << "\n-------------------------------------------------\n";
                        std::cout << "FOUND PRIVATE KEY: " << formatHex256(target_key_calc) << "\n";
                        std::cout << "-------------------------------------------------\n";
                        
                        cudaFreeHost(h_start_scalars);
                        cudaFreeHost(h_dist);
                        cudaFreeHost(h_scalars_Sp);
                        cudaFree(d_S); cudaFree(d_Kx); cudaFree(d_Ky); cudaFree(d_Kdist); cudaFree(d_Ktype);
                        cudaFree(d_scalars_Sp); cudaFree(d_Gx_Sp); cudaFree(d_Gy_Sp);
                        cudaFree(d_dp_buffer); cudaFree(d_dp_count);
                        return 0;
                    }
                } else {
                    DPInfo info;
                    info.type = host_dps[i].type;
                    for(int j=0; j<4;j++) info.distance[j] = host_dps[i].distance[j];
                    CPU_DP_MAP[key] = info;
                }
            }
            delete[] host_dps;
        }

        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        std::cout << "\rSpeed: " << std::fixed << std::setprecision(1) << (total_jumps/elapsed)/1e6 << " Mjumps/s | DPs: " << CPU_DP_MAP.size() << std::flush;
        
        if (found) break;
    }

    return 0;
}
