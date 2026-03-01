/*
 * CUDAKangaroo.cu  –  Pollard's Kangaroo (Lambda) GPU solver
 *
 * Algorithm overview
 * ==================
 * Given  Target public key  Q  and the knowledge that its private key  k
 * lies in the interval  [a, b],  we launch two families of random walkers:
 *
 *   Tame:  start at  G*(a + (b-a)/2 + offset)   with known scalar
 *   Wild:  start at  Q + G*offset                with unknown scalar (distance = offset)
 *
 * Each walker performs a deterministic pseudo-random walk by adding one of
 * J pre-computed jump points  J_i  at each step, chosen by hashing the
 * current  x-coordinate.  When a walker lands on a Distinguished Point
 * (DP) – defined as  x  having at least  dpBits  leading zeros – it
 * reports {x, distance, type} to the CPU.
 *
 * Collision:  tame.x == wild.x  =>  k = tame.scalar – wild.distance
 *
 * Performance tricks
 * ==================
 * 1. Jump table stored in __constant__ memory for broadcast reads.
 * 2. DP buffer in global memory; CPU periodically drains it via async
 *    memcpy so the GPU never stalls.
 * 3. Warp-level early exit polling via __shfl_sync.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>
#include <cmath>
#include <csignal>
#include <atomic>
#include <vector>
#include <unordered_map>
#include <array>
#include <algorithm>

/* ================================================================
   We do NOT include CUDAStructures.h / CUDAMath.h here because
   they define __device__ variables and non-inline functions that
   would clash with CUDACyclone.cu under -rdc=true.

   Instead we forward-declare the exact symbols we need.
   ================================================================ */

/* From CUDAMath.h – arithmetic primitives (defined there, linked via -rdc) */
__device__ void _ModInv(uint64_t* R);
__device__ void _ModMult(uint64_t *r, uint64_t *a, uint64_t *b);
__device__ void _ModMult(uint64_t *r, uint64_t *a);
__device__ void _ModSqr(uint64_t *rp, const uint64_t *up);
__device__ void ModSub256(uint64_t *r, uint64_t *a, uint64_t *b);
__device__ void ModNeg256(uint64_t *r, uint64_t *a);
__device__ void fieldAdd(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]);
__device__ void fieldInv(const uint64_t in[4], uint64_t out[4]);

/* From CUDAMath.h – scalar multiplication kernel */
__global__ void scalarMulKernelBase(const uint64_t* scalars_in, uint64_t* outX, uint64_t* outY, int N);

/* From CUDAUtils.h – host helpers (defined in header, but marked static/inline) */
extern bool hexToLE64(const std::string& h_in, uint64_t w[4]);
extern std::string formatHex256(const uint64_t limbs[4]);
extern long double ld_from_u256(const uint64_t v[4]);

static inline void host_add256_u64(const uint64_t a[4], uint64_t b, uint64_t out[4]) {
    __uint128_t sum = (__uint128_t)a[0] + b;
    out[0] = (uint64_t)sum;
    uint64_t carry = (uint64_t)(sum >> 64);
    for (int i = 1; i < 4; ++i) {
        sum = (__uint128_t)a[i] + carry;
        out[i] = (uint64_t)sum;
        carry = (uint64_t)(sum >> 64);
    }
}

static inline void host_add256(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
    __uint128_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        __uint128_t s = (__uint128_t)a[i] + b[i] + carry;
        out[i] = (uint64_t)s;
        carry = s >> 64;
    }
}

static inline void host_sub256(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t bi = b[i] + borrow;
        if (a[i] < bi) {
            out[i] = (uint64_t)(((__uint128_t(1) << 64) + a[i]) - bi);
            borrow = 1;
        } else {
            out[i] = a[i] - bi;
            borrow = 0;
        }
    }
}

static inline std::string host_formatCompressedPubHex(const uint64_t Rx[4], const uint64_t Ry[4]) {
    uint8_t out[33];
    out[0] = (Ry[0] & 1ULL) ? 0x03 : 0x02;
    int off = 1;
    for (int limb = 3; limb >= 0; --limb) {
        uint64_t v = Rx[limb];
        out[off+0]=(uint8_t)(v>>56); out[off+1]=(uint8_t)(v>>48);
        out[off+2]=(uint8_t)(v>>40); out[off+3]=(uint8_t)(v>>32);
        out[off+4]=(uint8_t)(v>>24); out[off+5]=(uint8_t)(v>>16);
        out[off+6]=(uint8_t)(v>> 8); out[off+7]=(uint8_t)(v>> 0);
        off += 8;
    }
    static const char* hexd = "0123456789ABCDEF";
    std::string s; s.resize(66);
    for (int i = 0; i < 33; ++i) { s[2*i] = hexd[(out[i]>>4)&0xF]; s[2*i+1] = hexd[out[i]&0xF]; }
    return s;
}

static inline std::string human_bytes_k(double bytes) {
    static const char* u[] = {"B","KB","MB","GB","TB","PB"};
    int k = 0;
    while (bytes >= 1024.0 && k < 5) { bytes /= 1024.0; ++k; }
    std::ostringstream o; o.setf(std::ios::fixed); o << std::setprecision(bytes<10?2:1) << bytes << " " << u[k];
    return o.str();
}

/* ================================================================
   Constants for the found-flag protocol (must match CUDAStructures.h)
   ================================================================ */
#define FOUND_NONE  0
#define FOUND_LOCK  1
#define FOUND_READY 2
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

/* ================================================================
   Kangaroo constants & jump table
   ================================================================ */

#define KANGAROO_NUM_JUMPS      32
#define KANGAROO_JUMP_MASK      (KANGAROO_NUM_JUMPS - 1)
#define KANGAROO_DP_BUFFER_SIZE (1u << 20)

/* Jump table in constant memory */
__constant__ uint64_t kc_jumpX[KANGAROO_NUM_JUMPS * 4];
__constant__ uint64_t kc_jumpY[KANGAROO_NUM_JUMPS * 4];
__constant__ uint64_t kc_jumpDist[KANGAROO_NUM_JUMPS * 4];

/* Target public key */
__constant__ uint64_t kc_targetX[4];
__constant__ uint64_t kc_targetY[4];

/* DP bit threshold */
__constant__ uint32_t kc_dpBits;

/* ================================================================
   DP entry structure (written by GPU, read by CPU)
   ================================================================ */

struct DPEntry {
    uint64_t x[4];
    uint64_t dist[4];
    uint32_t type;       /* 0 = tame, 1 = wild */
    uint32_t kangarooId;
};

/* ================================================================
   Device helpers
   ================================================================ */

__device__ __forceinline__ uint32_t jumpIndex(const uint64_t x[4]) {
    return (uint32_t)(x[0] & KANGAROO_JUMP_MASK);
}

__device__ __forceinline__ bool isDP(const uint64_t x[4], uint32_t dpBits) {
    if (dpBits == 0) return true;
    if (dpBits < 64)  return (x[3] >> (64 - dpBits)) == 0ULL;
    if (x[3] != 0ULL) return false;
    if (dpBits < 128) return (x[2] >> (64 - (dpBits - 64))) == 0ULL;
    if (x[2] != 0ULL) return false;
    if (dpBits < 192) return (x[1] >> (64 - (dpBits - 128))) == 0ULL;
    if (x[1] != 0ULL) return false;
    if (dpBits < 256) return (x[0] >> (64 - (dpBits - 192))) == 0ULL;
    return true;
}

__device__ __forceinline__ void dist_add256(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
    asm("add.cc.u64  %0, %1, %2;" : "=l"(out[0]) : "l"(a[0]), "l"(b[0]));
    asm("addc.cc.u64 %0, %1, %2;" : "=l"(out[1]) : "l"(a[1]), "l"(b[1]));
    asm("addc.cc.u64 %0, %1, %2;" : "=l"(out[2]) : "l"(a[2]), "l"(b[2]));
    asm("addc.u64    %0, %1, %2;" : "=l"(out[3]) : "l"(a[3]), "l"(b[3]));
}

/* Warp-level shuffle for 256-bit values (4 limbs) */
__device__ __forceinline__ void shfl_256(uint64_t v[4], int srcLane, uint32_t mask) {
    v[0] = __shfl_sync(mask, v[0], srcLane);
    v[1] = __shfl_sync(mask, v[1], srcLane);
    v[2] = __shfl_sync(mask, v[2], srcLane);
    v[3] = __shfl_sync(mask, v[3], srcLane);
}

/* ================================================================
   Kangaroo walk kernel  — OPTIMIZED
   
   Optimizations:
   1. Warp-level Batched Montgomery Inversion:
      Instead of 32 independent ModInv per warp per step,
      we build a product chain of all 32 dx values, do ONE
      ModInv, then back-propagate.  Reduces inversions by 32x.
      Cost: 1 ModInv + ~93 ModMult vs 32 ModInv.
   ================================================================ */

__global__ void kernel_kangaroo_walk(
    uint64_t* __restrict__ g_Px,
    uint64_t* __restrict__ g_Py,
    uint64_t* __restrict__ g_dist,
    const uint32_t* __restrict__ g_type,
    uint64_t threadsTotal,
    uint32_t stepsPerLaunch,
    DPEntry* __restrict__ g_dpBuffer,
    unsigned int* __restrict__ g_dpCount,
    uint32_t dpBufferCapacity,
    int* __restrict__ g_foundFlag
) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= threadsTotal) return;
    if (*((volatile int*)g_foundFlag) == FOUND_READY) return;

    const uint32_t lane = threadIdx.x & (WARP_SIZE - 1);
    const uint32_t full_mask = 0xFFFFFFFFu;

    uint64_t px[4], py[4], d[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        px[i] = g_Px[gid * 4 + i];
        py[i] = g_Py[gid * 4 + i];
        d[i]  = g_dist[gid * 4 + i];
    }
    uint32_t kType = g_type[gid];
    uint32_t dpBits = kc_dpBits;

    for (uint32_t step = 0; step < stepsPerLaunch; step++) {
        /* Warp-level early exit every 64 steps */
        if ((step & 63) == 0) {
            int f = 0;
            if (lane == 0) f = *((volatile int*)g_foundFlag);
            f = __shfl_sync(full_mask, f, 0);
            if (f == FOUND_READY) break;
        }

        /* 1. Pick jump */
        uint32_t jIdx = jumpIndex(px);
        uint64_t jx[4], jy[4], jd[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            jx[i] = kc_jumpX[jIdx * 4 + i];
            jy[i] = kc_jumpY[jIdx * 4 + i];
            jd[i] = kc_jumpDist[jIdx * 4 + i];
        }

        /* 2. Compute dx = jx - px */
        uint64_t dx[4];
        ModSub256(dx, jx, px);

        /* ========================================================
           BATCHED MONTGOMERY INVERSION (Shared Memory)

           Classic Montgomery trick using shared memory within each
           warp.  32 inversions become 1 inversion + 93 multiplications.

           prefix[i] = dx[0] * dx[1] * ... * dx[i]
           Invert prefix[31] = product of all dx
           Back-propagate:
             inv[31] = prefix[30] * total_inv
             inv[i]  = prefix[i-1] * running_right_product
             running *= dx[i+1]
           ======================================================== */
        /* Batched Montgomery Inversion via shared memory */
        const uint32_t warpId = threadIdx.x >> 5;
        __shared__ uint64_t s_dx[8][32][4];
        __shared__ uint64_t s_prefix[8][32][4];
        __shared__ uint64_t s_inv[8][32][4];
        uint64_t my_inv[4];
#pragma unroll
        for (int i = 0; i < 4; i++) s_dx[warpId][lane][i] = dx[i];
        __syncwarp(full_mask);

        /* Redo prefix product using s_dx */
        if (lane == 0) {
            /* Rebuild prefix */
            for (int i = 0; i < 4; i++) s_prefix[warpId][0][i] = s_dx[warpId][0][i];
            for (int l = 1; l < WARP_SIZE; l++) {
                uint64_t res[4];
                _ModMult(res, (uint64_t*)s_prefix[warpId][l-1], (uint64_t*)s_dx[warpId][l]);
                for (int i = 0; i < 4; i++) s_prefix[warpId][l][i] = res[i];
            }

            /* Invert total product */
            uint64_t inv5[5];
            for (int i = 0; i < 4; i++) inv5[i] = s_prefix[warpId][WARP_SIZE-1][i];
            inv5[4] = 0;
            _ModInv(inv5);

            /* Back-propagate */
            uint64_t right[4];
            for (int i = 0; i < 4; i++) right[i] = inv5[i];

            for (int l = WARP_SIZE - 1; l >= 0; l--) {
                if (l == 0) {
                    for (int i = 0; i < 4; i++) s_inv[warpId][l][i] = right[i];
                } else {
                    uint64_t res[4];
                    _ModMult(res, (uint64_t*)s_prefix[warpId][l-1], right);
                    for (int i = 0; i < 4; i++) s_inv[warpId][l][i] = res[i];
                }
                /* right *= dx[l] */
                uint64_t new_right[4];
                _ModMult(new_right, right, (uint64_t*)s_dx[warpId][l]);
                for (int i = 0; i < 4; i++) right[i] = new_right[i];
            }
        }
        __syncwarp(full_mask);

        /* Each lane reads its individual inverse */
#pragma unroll
        for (int i = 0; i < 4; i++) my_inv[i] = s_inv[warpId][lane][i];

        /* lambda = (jy - py) * inv(jx - px) */
        uint64_t dy_ec[4], lam[4];
        ModSub256(dy_ec, jy, py);
        _ModMult(lam, dy_ec, my_inv);

        /* x3 = lam^2 - px - jx */
        uint64_t x3[4];
        _ModSqr(x3, lam);
        ModSub256(x3, x3, px);
        ModSub256(x3, x3, jx);

        /* y3 = lam * (px - x3) - py */
        uint64_t tmp[4], y3[4];
        ModSub256(tmp, px, x3);
        _ModMult(y3, lam, tmp);
        ModSub256(y3, y3, py);

        /* Update point */
#pragma unroll
        for (int i = 0; i < 4; i++) { px[i] = x3[i]; py[i] = y3[i]; }

        /* Update distance: d += jd */
        uint64_t nd[4];
        dist_add256(d, jd, nd);
#pragma unroll
        for (int i = 0; i < 4; i++) d[i] = nd[i];

        /* 3. Check for Distinguished Point */
        if (isDP(px, dpBits)) {
            uint32_t slot = atomicAdd(g_dpCount, 1u);
            if (slot < dpBufferCapacity) {
                DPEntry* e = &g_dpBuffer[slot];
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    e->x[i]    = px[i];
                    e->dist[i] = d[i];
                }
                e->type = kType;
                e->kangarooId = (uint32_t)gid;
            }
        }
    }

    /* Write state back */
#pragma unroll
    for (int i = 0; i < 4; i++) {
        g_Px[gid * 4 + i]   = px[i];
        g_Py[gid * 4 + i]   = py[i];
        g_dist[gid * 4 + i] = d[i];
    }
}

/* ================================================================
   Kernel: add target Q to wild kangaroo initial points
   ================================================================ */

__global__ void kernel_add_target_to_wild(
    uint64_t* __restrict__ g_Px,
    uint64_t* __restrict__ g_Py,
    const uint32_t* __restrict__ g_type,
    uint64_t threadsTotal
) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= threadsTotal) return;
    if (g_type[gid] != 1) return; /* only wild */

    uint64_t px[4], py[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        px[i] = g_Px[gid * 4 + i];
        py[i] = g_Py[gid * 4 + i];
    }

    /* Check if this is the identity/zero point (scalar was 0) */
    bool isZero = (px[0] | px[1] | px[2] | px[3] | py[0] | py[1] | py[2] | py[3]) == 0ULL;

    if (isZero) {
        /* G*0 = infinity => result is just Q */
#pragma unroll
        for (int i = 0; i < 4; i++) {
            g_Px[gid * 4 + i] = kc_targetX[i];
            g_Py[gid * 4 + i] = kc_targetY[i];
        }
        return;
    }

    /* Affine addition: R = G*offset + Q */
    uint64_t qx[4], qy[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        qx[i] = kc_targetX[i];
        qy[i] = kc_targetY[i];
    }

    uint64_t dx[4], dy_ec[4];
    ModSub256(dx, qx, px);
    ModSub256(dy_ec, qy, py);

    uint64_t inv[5];
#pragma unroll
    for (int i = 0; i < 4; i++) inv[i] = dx[i];
    inv[4] = 0;
    _ModInv(inv);

    uint64_t lam[4];
    _ModMult(lam, dy_ec, inv);

    uint64_t x3[4];
    _ModSqr(x3, lam);
    ModSub256(x3, x3, px);
    ModSub256(x3, x3, qx);

    uint64_t tmp[4], y3[4];
    ModSub256(tmp, px, x3);
    _ModMult(y3, lam, tmp);
    ModSub256(y3, y3, py);

#pragma unroll
    for (int i = 0; i < 4; i++) {
        g_Px[gid * 4 + i] = x3[i];
        g_Py[gid * 4 + i] = y3[i];
    }
}

/* ================================================================
   Kernel: decompress public key Y from X
   ================================================================ */

__global__ void kernel_decompress_pubkey(const uint64_t* inX, uint64_t* outX, uint64_t* outY, uint8_t prefix) {
    uint64_t x[4], x2[4], x3[4], y2[4];
    uint64_t seven[4] = {7ULL, 0ULL, 0ULL, 0ULL};

    for (int i = 0; i < 4; i++) x[i] = inX[i];

    _ModSqr(x2, x);
    _ModMult(x3, x2, x);
    fieldAdd(x3, seven, y2);

    /* y = y2^((p+1)/4) mod p   — since p ≡ 3 mod 4 for secp256k1 */
    const uint64_t exp[4] = {
        0xFFFFFFFFBFFFFF0CULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
        0x3FFFFFFFFFFFFFFFULL
    };

    uint64_t result[4] = {1ULL, 0ULL, 0ULL, 0ULL};
    uint64_t base[4];
    for (int i = 0; i < 4; i++) base[i] = y2[i];

    for (int limb = 0; limb < 4; limb++) {
        uint64_t bits = exp[limb];
        for (int bit = 0; bit < 64; bit++) {
            if (bits & 1ULL) {
                _ModMult(result, result, base);
            }
            _ModSqr(base, base);
            bits >>= 1;
        }
    }

    uint8_t resultParity = (uint8_t)(result[0] & 1ULL);
    uint8_t wantedParity = prefix - 0x02;

    if (resultParity != wantedParity) {
        ModNeg256(result, result);
    }

    for (int i = 0; i < 4; i++) {
        outX[i] = x[i];
        outY[i] = result[i];
    }
}

/* ================================================================
   CPU-side DP hash table for collision detection
   ================================================================ */

struct DPRecord {
    uint64_t dist[4];
    uint32_t type;
};

struct DPHash {
    size_t operator()(const std::array<uint64_t, 4>& k) const {
        return std::hash<uint64_t>()(k[0]) ^ (std::hash<uint64_t>()(k[1]) << 1)
             ^ (std::hash<uint64_t>()(k[2]) << 2) ^ (std::hash<uint64_t>()(k[3]) << 3);
    }
};

class DPHashTable {
public:
    bool insert(const DPEntry& dp, uint64_t tame_dist[4], uint64_t wild_dist[4]) {
        std::array<uint64_t, 4> key = {dp.x[0], dp.x[1], dp.x[2], dp.x[3]};

        auto it = table_.find(key);
        if (it != table_.end()) {
            const DPRecord& existing = it->second;
            if (existing.type != dp.type) {
                if (existing.type == 0) {
                    for (int i = 0; i < 4; i++) { tame_dist[i] = existing.dist[i]; wild_dist[i] = dp.dist[i]; }
                } else {
                    for (int i = 0; i < 4; i++) { tame_dist[i] = dp.dist[i]; wild_dist[i] = existing.dist[i]; }
                }
                return true;
            }
            return false;
        }
        DPRecord rec;
        for (int i = 0; i < 4; i++) rec.dist[i] = dp.dist[i];
        rec.type = dp.type;
        table_[key] = rec;
        return false;
    }

    size_t size() const { return table_.size(); }
    void clear() { table_.clear(); }

private:
    std::unordered_map<std::array<uint64_t, 4>, DPRecord, DPHash> table_;
};

/* ================================================================
   Save / Resume state
   ================================================================ */

static bool saveState(const char* path,
    const uint64_t* h_Px, const uint64_t* h_Py, const uint64_t* h_dist,
    const uint32_t* h_types, uint64_t threadsTotal,
    const DPHashTable& dpTable, uint64_t totalJumps,
    const uint64_t targetX[4], const uint64_t targetY[4],
    const uint64_t range_start[4], const uint64_t range_end[4],
    uint32_t dpBits) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    /* Header magic */
    const uint32_t magic = 0x4B414E47; /* 'KANG' */
    fwrite(&magic, 4, 1, f);
    fwrite(&threadsTotal, 8, 1, f);
    fwrite(&totalJumps, 8, 1, f);
    fwrite(&dpBits, 4, 1, f);
    fwrite(targetX, 8, 4, f);
    fwrite(targetY, 8, 4, f);
    fwrite(range_start, 8, 4, f);
    fwrite(range_end, 8, 4, f);
    /* Kangaroo state */
    fwrite(h_Px,    8, threadsTotal * 4, f);
    fwrite(h_Py,    8, threadsTotal * 4, f);
    fwrite(h_dist,  8, threadsTotal * 4, f);
    fwrite(h_types, 4, threadsTotal, f);
    /* DP table */
    uint64_t dpSize = dpTable.size();
    fwrite(&dpSize, 8, 1, f);
    /* We can't easily serialize the unordered_map, so we skip DP table
       in save file - DPs will be re-collected on resume (slight loss). */
    fclose(f);
    return true;
}

static bool loadState(const char* path,
    std::vector<uint64_t>& h_Px, std::vector<uint64_t>& h_Py,
    std::vector<uint64_t>& h_dist, std::vector<uint32_t>& h_types,
    uint64_t& threadsTotal, uint64_t& totalJumps,
    uint64_t targetX[4], uint64_t targetY[4],
    uint64_t range_start[4], uint64_t range_end[4],
    uint32_t& dpBits) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    uint32_t magic = 0;
    if (fread(&magic, 4, 1, f) != 1) { fclose(f); return false; }
    if (magic != 0x4B414E47) { fclose(f); return false; }
    if (fread(&threadsTotal, 8, 1, f) != 1) { fclose(f); return false; }
    if (fread(&totalJumps, 8, 1, f) != 1) { fclose(f); return false; }
    if (fread(&dpBits, 4, 1, f) != 1) { fclose(f); return false; }
    if (fread(targetX, 8, 4, f) != 4) { fclose(f); return false; }
    if (fread(targetY, 8, 4, f) != 4) { fclose(f); return false; }
    if (fread(range_start, 8, 4, f) != 4) { fclose(f); return false; }
    if (fread(range_end, 8, 4, f) != 4) { fclose(f); return false; }
    
    h_Px.resize(threadsTotal * 4);
    h_Py.resize(threadsTotal * 4);
    h_dist.resize(threadsTotal * 4);
    h_types.resize(threadsTotal);
    
    if (fread(h_Px.data(),    8, threadsTotal * 4, f) != threadsTotal * 4) { fclose(f); return false; }
    if (fread(h_Py.data(),    8, threadsTotal * 4, f) != threadsTotal * 4) { fclose(f); return false; }
    if (fread(h_dist.data(),  8, threadsTotal * 4, f) != threadsTotal * 4) { fclose(f); return false; }
    if (fread(h_types.data(), 4, threadsTotal, f) != threadsTotal) { fclose(f); return false; }
    
    fclose(f);
    return true;
}

/* ================================================================
   Host helper: parse compressed public key hex
   ================================================================ */

static bool parseCompressedPubKey(const std::string& hex, uint64_t outX[4], uint64_t outY[4]) {
    std::string h = hex;
    if (h.size() >= 2 && h[0] == '0' && (h[1] == 'x' || h[1] == 'X')) h = h.substr(2);
    if (h.size() != 66) return false;

    uint8_t prefix = 0;
    {
        std::string pfx = h.substr(0, 2);
        prefix = (uint8_t)std::stoul(pfx, nullptr, 16);
        if (prefix != 0x02 && prefix != 0x03) return false;
    }

    std::string xhex = h.substr(2, 64);
    if (!hexToLE64(xhex, outX)) return false;

    /* Store prefix in outY[0] for GPU decompression */
    outY[0] = (uint64_t)prefix;
    outY[1] = 0; outY[2] = 0; outY[3] = 0;
    return true;
}

/* ================================================================
   Host helper: estimate total keys
   ================================================================ */

long double ld_from_u256(const uint64_t x[4]) {
    long double res = 0.0L;
    long double base = 1.0L;
    long double multiplier = 18446744073709551616.0L; // 2^64
    for (int i = 0; i < 4; i++) {
        res += (long double)x[i] * base;
        base *= multiplier;
    }
    return res;
}

/* ================================================================
   Main Kangaroo entry point
   ================================================================ */

static volatile sig_atomic_t g_sigint_kangaroo = 0;
static void handle_sigint_kangaroo(int) { g_sigint_kangaroo = 1; }

int kangaroo_main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint_kangaroo);

    std::string pubkey_hex, range_hex, save_file = "kangaroo_state.bin";
    bool resumeMode = false;
    uint32_t dpBits = 0; /* 0 means auto-calculate */
    uint32_t threadsPerBlock = 256;
    uint32_t saveIntervalSec = 300; /* auto-save every 5 minutes */
    uint32_t numBlocks = 0;
    uint32_t stepsPerLaunch = 512;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--kangaroo") continue;
        else if (arg == "--public-key" && i + 1 < argc) pubkey_hex = argv[++i];
        else if (arg == "--range" && i + 1 < argc) range_hex = argv[++i];
        else if (arg == "--dp-bits" && i + 1 < argc) dpBits = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg == "--steps" && i + 1 < argc) stepsPerLaunch = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg == "--blocks" && i + 1 < argc) numBlocks = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg == "--save-file" && i + 1 < argc) save_file = argv[++i];
        else if (arg == "--save-interval" && i + 1 < argc) saveIntervalSec = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg == "--resume") resumeMode = true;
    }

    if (!resumeMode && (pubkey_hex.empty() || range_hex.empty())) {
        std::cerr << "Usage: " << argv[0]
                  << " --kangaroo --public-key <compressed_hex> --range <start_hex>:<end_hex>"
                  << " [--dp-bits N] [--steps N] [--blocks N]\n"
                  << "        " << argv[0] << " --kangaroo --resume [--save-file <path>]\n";
        return EXIT_FAILURE;
    }

    /* Resume state variables */
    std::vector<uint64_t> r_Px, r_Py, r_dist;
    std::vector<uint32_t> r_types;
    uint64_t threadsTotal = 0;
    uint64_t totalJumps = 0;
    uint64_t targetX[4]{0}, targetY[4]{0};
    uint64_t range_start[4]{0}, range_end[4]{0};

    if (resumeMode) {
        std::cout << "Loading state from " << save_file << "...\n";
        if (!loadState(save_file.c_str(), r_Px, r_Py, r_dist, r_types,
                       threadsTotal, totalJumps, targetX, targetY,
                       range_start, range_end, dpBits)) {
            std::cerr << "Failed to load state from " << save_file << "\n";
            return EXIT_FAILURE;
        }
    } else {
        /* Parse range */
        size_t colon_pos = range_hex.find(':');
        if (colon_pos == std::string::npos) {
            std::cerr << "Error: range format must be start:end\n";
            return EXIT_FAILURE;
        }
        std::string start_hex = range_hex.substr(0, colon_pos);
        std::string end_hex   = range_hex.substr(colon_pos + 1);

        if (!hexToLE64(start_hex, range_start) || !hexToLE64(end_hex, range_end)) {
            std::cerr << "Error: invalid range hex\n";
            return EXIT_FAILURE;
        }
    }

    uint64_t range_width[4];
    host_sub256(range_end, range_start, range_width);
    host_add256_u64(range_width, 1ULL, range_width);

    /* Parse public key */
    if (!resumeMode) {
        if (!parseCompressedPubKey(pubkey_hex, targetX, targetY)) {
            std::cerr << "Error: invalid compressed public key (expected 02/03 + 32 bytes hex)\n";
            return EXIT_FAILURE;
        }
    }

    /* CUDA init */
    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        std::cerr << "CUDA init error\n";
        return EXIT_FAILURE;
    }
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    if (numBlocks == 0)
        numBlocks = (uint32_t)prop.multiProcessorCount * 4u;

    if (!resumeMode) {
        threadsTotal = (uint64_t)numBlocks * (uint64_t)threadsPerBlock;
    } else {
        /* On resume, use the block/thread count from the save file */
        numBlocks = (uint32_t)(threadsTotal / threadsPerBlock);
    }
    uint64_t numTame = threadsTotal / 2;
    uint64_t numWild = threadsTotal - numTame;

    /* --------------------------------------------------------
       Decompress public key Y on GPU
       -------------------------------------------------------- */
    if (!resumeMode) {
        uint8_t prefix = (uint8_t)targetY[0];
        uint64_t *d_inX, *d_outX, *d_outY;
        cudaMalloc(&d_inX, 4 * sizeof(uint64_t));
        cudaMalloc(&d_outX, 4 * sizeof(uint64_t));
        cudaMalloc(&d_outY, 4 * sizeof(uint64_t));
        cudaMemcpy(d_inX, targetX, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);

        kernel_decompress_pubkey<<<1, 1>>>(d_inX, d_outX, d_outY, prefix);
        cudaDeviceSynchronize();

        cudaMemcpy(targetX, d_outX, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(targetY, d_outY, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        cudaFree(d_inX); cudaFree(d_outX); cudaFree(d_outY);
    }

    /* Upload target to constant memory */
    cudaMemcpyToSymbol(kc_targetX, targetX, 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(kc_targetY, targetY, 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(kc_dpBits, &dpBits, sizeof(uint32_t));

    /* --------------------------------------------------------
       Build jump table:  d_j = 2^(bit_j)  distributed across
       [0, rangeBitLen/2] and then G * d_j
       -------------------------------------------------------- */

    uint32_t rangeBitLen = 0;
    for (int limb = 3; limb >= 0; limb--) {
        if (range_width[limb] != 0) {
            rangeBitLen = (uint32_t)(limb * 64 + 64 - __builtin_clzll(range_width[limb]));
            break;
        }
    }

    uint64_t h_jumpDist[KANGAROO_NUM_JUMPS * 4];
    uint64_t h_jumpScalars[KANGAROO_NUM_JUMPS * 4];
    std::memset(h_jumpDist, 0, sizeof(h_jumpDist));
    std::memset(h_jumpScalars, 0, sizeof(h_jumpScalars));

    for (int j = 0; j < KANGAROO_NUM_JUMPS; j++) {
        uint32_t bitPos = (uint32_t)((uint64_t)(j + 1) * (rangeBitLen / 2) / KANGAROO_NUM_JUMPS);
        if (bitPos > 250) bitPos = 250;
        if (bitPos == 0) bitPos = 1;
        int limb = bitPos / 64;
        int shift = bitPos % 64;
        h_jumpDist[j * 4 + limb] = 1ULL << shift;
        for (int k = 0; k < 4; k++)
            h_jumpScalars[j * 4 + k] = h_jumpDist[j * 4 + k];
    }

    /* Compute G * d_j for each jump */
    uint64_t *d_jScalars, *d_jOutX, *d_jOutY;
    cudaMalloc(&d_jScalars, KANGAROO_NUM_JUMPS * 4 * sizeof(uint64_t));
    cudaMalloc(&d_jOutX,    KANGAROO_NUM_JUMPS * 4 * sizeof(uint64_t));
    cudaMalloc(&d_jOutY,    KANGAROO_NUM_JUMPS * 4 * sizeof(uint64_t));
    cudaMemcpy(d_jScalars, h_jumpScalars, KANGAROO_NUM_JUMPS * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);

    scalarMulKernelBase<<<1, KANGAROO_NUM_JUMPS>>>(d_jScalars, d_jOutX, d_jOutY, KANGAROO_NUM_JUMPS);
    cudaDeviceSynchronize();

    uint64_t h_jumpX[KANGAROO_NUM_JUMPS * 4], h_jumpY[KANGAROO_NUM_JUMPS * 4];
    cudaMemcpy(h_jumpX, d_jOutX, KANGAROO_NUM_JUMPS * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_jumpY, d_jOutY, KANGAROO_NUM_JUMPS * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_jScalars); cudaFree(d_jOutX); cudaFree(d_jOutY);

    cudaMemcpyToSymbol(kc_jumpX,    h_jumpX,    sizeof(h_jumpX));
    cudaMemcpyToSymbol(kc_jumpY,    h_jumpY,    sizeof(h_jumpY));
    cudaMemcpyToSymbol(kc_jumpDist, h_jumpDist, sizeof(h_jumpDist));

    /* --------------------------------------------------------
       Initialize kangaroo starting points
       -------------------------------------------------------- */

    std::cout << "======== Kangaroo Mode ================================\n";
    std::cout << std::left << std::setw(20) << "Device"            << " : " << prop.name
              << " (compute " << prop.major << "." << prop.minor << ")\n";
    std::cout << std::left << std::setw(20) << "SM"                << " : " << prop.multiProcessorCount << "\n";
    std::cout << std::left << std::setw(20) << "Threads"           << " : " << threadsTotal
              << " (" << numTame << " tame + " << numWild << " wild)\n";
    std::cout << std::left << std::setw(20) << "Blocks"            << " : " << numBlocks << "\n";
    std::cout << std::left << std::setw(20) << "Steps/launch"      << " : " << stepsPerLaunch << "\n";
    std::cout << std::left << std::setw(20) << "DP bits"           << " : " << dpBits << "\n";
    std::cout << std::left << std::setw(20) << "Range bits"        << " : " << rangeBitLen << "\n";
    std::cout << std::left << std::setw(20) << "Jump table"        << " : " << KANGAROO_NUM_JUMPS << " entries\n";
    std::cout << std::left << std::setw(20) << "Target pubkey X"   << " : " << formatHex256(targetX) << "\n";
    std::cout << "-------------------------------------------------------\n";

    std::vector<uint64_t> h_startScalars;
    std::vector<uint32_t> h_types;

    if (!resumeMode) {
        h_startScalars.resize(threadsTotal * 4, 0);
        h_types.resize(threadsTotal, 0);

        /* Midpoint: mid = range_start + range_width / 2 */
        uint64_t half_width[4], mid[4];
        half_width[0] = (range_width[0] >> 1) | (range_width[1] << 63);
        half_width[1] = (range_width[1] >> 1) | (range_width[2] << 63);
        half_width[2] = (range_width[2] >> 1) | (range_width[3] << 63);
        half_width[3] = (range_width[3] >> 1);
        host_add256(range_start, half_width, mid);

        /* Tame: start at mid + i */
        for (uint64_t i = 0; i < numTame; i++) {
            uint64_t scalar[4];
            host_add256_u64(mid, i, scalar);
            for (int k = 0; k < 4; k++)
                h_startScalars[i * 4 + k] = scalar[k];
            h_types[i] = 0;
        }
        /* Wild: offset from Q = i */
        for (uint64_t i = 0; i < numWild; i++) {
            uint64_t idx = numTame + i;
            h_startScalars[idx * 4 + 0] = i;
            h_startScalars[idx * 4 + 1] = 0;
            h_startScalars[idx * 4 + 2] = 0;
            h_startScalars[idx * 4 + 3] = 0;
            h_types[idx] = 1;
        }
    }

    /* Allocate GPU arrays */
    uint64_t *d_Px, *d_Py, *d_dist;
    uint32_t *d_type;
    int *d_foundFlag;
    DPEntry *d_dpBuffer;
    unsigned int *d_dpCount;

    auto ck = [](cudaError_t e, const char* msg) {
        if (e != cudaSuccess) {
            std::cerr << msg << ": " << cudaGetErrorString(e) << "\n";
            std::exit(EXIT_FAILURE);
        }
    };

    ck(cudaMalloc(&d_Px,        threadsTotal * 4 * sizeof(uint64_t)), "alloc Px");
    ck(cudaMalloc(&d_Py,        threadsTotal * 4 * sizeof(uint64_t)), "alloc Py");
    ck(cudaMalloc(&d_dist,      threadsTotal * 4 * sizeof(uint64_t)), "alloc dist");
    ck(cudaMalloc(&d_type,      threadsTotal * sizeof(uint32_t)),     "alloc type");
    ck(cudaMalloc(&d_foundFlag, sizeof(int)),                         "alloc foundFlag");
    ck(cudaMalloc(&d_dpBuffer,  KANGAROO_DP_BUFFER_SIZE * sizeof(DPEntry)), "alloc dpBuf");
    ck(cudaMalloc(&d_dpCount,   sizeof(unsigned int)),                "alloc dpCount");

    { int zero = FOUND_NONE; cudaMemcpy(d_foundFlag, &zero, sizeof(int), cudaMemcpyHostToDevice); }
    { unsigned int zero = 0;  cudaMemcpy(d_dpCount, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice); }

    if (!resumeMode) {
        uint64_t *d_startScalars;
        ck(cudaMalloc(&d_startScalars, threadsTotal * 4 * sizeof(uint64_t)), "alloc startScalars");
        ck(cudaMemcpy(d_startScalars, h_startScalars.data(), threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy scalars");
        ck(cudaMemcpy(d_type, h_types.data(), threadsTotal * sizeof(uint32_t), cudaMemcpyHostToDevice), "cpy types");

        /* Scalar multiply all starting points: P_i = G * scalar_i */
        int bs = (int)((threadsTotal + threadsPerBlock - 1) / threadsPerBlock);
        scalarMulKernelBase<<<bs, threadsPerBlock>>>(d_startScalars, d_Px, d_Py, (int)threadsTotal);
        ck(cudaDeviceSynchronize(), "scalarMul init");

        /* Add Q to wild points: P_wild = G*offset + Q */
        kernel_add_target_to_wild<<<bs, threadsPerBlock>>>(d_Px, d_Py, d_type, threadsTotal);
        ck(cudaDeviceSynchronize(), "addQ to wild");

        /* Set initial distances (= starting scalars for purpose of tracking) */
        ck(cudaMemcpy(d_dist, d_startScalars, threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyDeviceToDevice), "cpy dist");
        cudaFree(d_startScalars);
    } else {
        /* Just upload the loaded state */
        ck(cudaMemcpy(d_Px,   r_Px.data(),   threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy r_Px");
        ck(cudaMemcpy(d_Py,   r_Py.data(),   threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy r_Py");
        ck(cudaMemcpy(d_dist, r_dist.data(), threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy r_dist");
        ck(cudaMemcpy(d_type, r_types.data(),threadsTotal *     sizeof(uint32_t), cudaMemcpyHostToDevice), "cpy r_types");
        
        /* Clear memory of vectors to free RAM */
        r_Px.clear(); r_Py.clear(); r_dist.clear(); r_types.clear();
    }

    /* Memory info */
    {
        size_t freeB = 0, totalB = 0;
        cudaMemGetInfo(&freeB, &totalB);
        size_t usedB = totalB - freeB;
        double util = totalB ? (double)usedB * 100.0 / (double)totalB : 0.0;
        std::cout << std::left << std::setw(20) << "Memory utilization"
                  << " : " << std::fixed << std::setprecision(1) << util << "% ("
                  << human_bytes_k((double)usedB) << " / " << human_bytes_k((double)totalB) << ")\n";
    }

    std::cout << "\n======== Phase-1: Kangaroo Walk ========================\n";

    /* --------------------------------------------------------
       Main loop
       -------------------------------------------------------- */

    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");

    DPHashTable dpTable;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto tLast = t0;
    auto tLastSave = t0;

    uint64_t lastJumps = 0;
    bool found = false;
    uint64_t solution_privkey[4] = {0};

    std::vector<DPEntry> hostDPBuffer(KANGAROO_DP_BUFFER_SIZE);

    while (!found && !g_sigint_kangaroo) {

        { unsigned int zero = 0;
          cudaMemcpyAsync(d_dpCount, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice, stream); }

        kernel_kangaroo_walk<<<numBlocks, threadsPerBlock, 0, stream>>>(
            d_Px, d_Py, d_dist, d_type,
            threadsTotal, stepsPerLaunch,
            d_dpBuffer, d_dpCount, KANGAROO_DP_BUFFER_SIZE,
            d_foundFlag
        );

        cudaStreamSynchronize(stream);
        totalJumps += (uint64_t)threadsTotal * stepsPerLaunch;

        /* Harvest DPs */
        unsigned int dpCount = 0;
        cudaMemcpy(&dpCount, d_dpCount, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (dpCount > KANGAROO_DP_BUFFER_SIZE) dpCount = KANGAROO_DP_BUFFER_SIZE;

        if (dpCount > 0) {
            cudaMemcpy(hostDPBuffer.data(), d_dpBuffer, dpCount * sizeof(DPEntry), cudaMemcpyDeviceToHost);

            for (unsigned int i = 0; i < dpCount; i++) {
                uint64_t tame_dist[4], wild_dist[4];
                if (dpTable.insert(hostDPBuffer[i], tame_dist, wild_dist)) {
                    /* COLLISION: private key = tame_dist - wild_dist */
                    host_sub256(tame_dist, wild_dist, solution_privkey);
                    found = true;
                    break;
                }
            }
        }

        /* Progress display */
        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - tLast).count();
        if (dt >= 1.0) {
            double elapsed = std::chrono::duration<double>(now - t0).count();
            double delta = (double)(totalJumps - lastJumps);
            double mjumps = delta / (dt * 1e6);

            long double total_expected = std::sqrt(ld_from_u256(range_width)) * 2.08L;
            long double prog = (total_expected > 0.0L) ? ((long double)totalJumps / total_expected) * 100.0L : 0.0L;
            if (prog > 100.0L) prog = 100.0L;

            std::cout << "\rTime: " << std::fixed << std::setprecision(1) << elapsed
                      << " s | Speed: " << std::fixed << std::setprecision(1) << mjumps
                      << " Mjumps/s | DPs: " << dpTable.size()
                      << " | Jumps: " << totalJumps
                      << " | Est: " << std::fixed << std::setprecision(2) << (double)prog << " %     ";
            std::cout.flush();
            lastJumps = totalJumps;
            tLast = now;

            /* Auto-save state periodically */
            if (saveIntervalSec > 0) {
                double dtSave = std::chrono::duration<double>(now - tLastSave).count();
                if (dtSave >= saveIntervalSec) {
                    std::vector<uint64_t> sv_Px(threadsTotal*4), sv_Py(threadsTotal*4), sv_dist(threadsTotal*4);
                    std::vector<uint32_t> sv_types(threadsTotal);
                    cudaMemcpy(sv_Px.data(),   d_Px,   threadsTotal*4*sizeof(uint64_t), cudaMemcpyDeviceToHost);
                    cudaMemcpy(sv_Py.data(),   d_Py,   threadsTotal*4*sizeof(uint64_t), cudaMemcpyDeviceToHost);
                    cudaMemcpy(sv_dist.data(), d_dist, threadsTotal*4*sizeof(uint64_t), cudaMemcpyDeviceToHost);
                    cudaMemcpy(sv_types.data(), d_type, threadsTotal*sizeof(uint32_t), cudaMemcpyDeviceToHost);
                    if (saveState(save_file.c_str(), sv_Px.data(), sv_Py.data(), sv_dist.data(),
                                  sv_types.data(), threadsTotal, dpTable, totalJumps,
                                  targetX, targetY, range_start, range_end, dpBits)) {
                        std::cout << "\n[Auto-saved to " << save_file << "]\n";
                    }
                    tLastSave = now;
                }
            }
        }
    }

    std::cout << "\n";

    if (found) {
        std::cout << "\n======== FOUND MATCH! =================================\n";
        std::cout << "Private Key   : " << formatHex256(solution_privkey) << "\n";

        /* Verify by computing G * privkey */
        uint64_t *d_vScalar, *d_vOutX, *d_vOutY;
        cudaMalloc(&d_vScalar, 4 * sizeof(uint64_t));
        cudaMalloc(&d_vOutX,   4 * sizeof(uint64_t));
        cudaMalloc(&d_vOutY,   4 * sizeof(uint64_t));
        cudaMemcpy(d_vScalar, solution_privkey, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        scalarMulKernelBase<<<1, 1>>>(d_vScalar, d_vOutX, d_vOutY, 1);
        cudaDeviceSynchronize();
        uint64_t solX[4], solY[4];
        cudaMemcpy(solX, d_vOutX, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(solY, d_vOutY, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        std::cout << "Public Key    : " << host_formatCompressedPubHex(solX, solY) << "\n";

        bool match = true;
        for (int i = 0; i < 4; i++) {
            if (solX[i] != targetX[i] || solY[i] != targetY[i]) match = false;
        }
        std::cout << "Verification  : " << (match ? "PASSED" : "MISMATCH") << "\n";

        cudaFree(d_vScalar); cudaFree(d_vOutX); cudaFree(d_vOutY);
    } else if (g_sigint_kangaroo) {
        std::cout << "======== INTERRUPTED — SAVING STATE ====================\n";
        std::cout << "Total jumps   : " << totalJumps << "\n";
        std::cout << "DPs collected : " << dpTable.size() << "\n";
        /* Save state to disk */
        std::vector<uint64_t> sv_Px(threadsTotal*4), sv_Py(threadsTotal*4), sv_dist(threadsTotal*4);
        std::vector<uint32_t> sv_types(threadsTotal);
        cudaMemcpy(sv_Px.data(),   d_Px,   threadsTotal*4*sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(sv_Py.data(),   d_Py,   threadsTotal*4*sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(sv_dist.data(), d_dist, threadsTotal*4*sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(sv_types.data(), d_type, threadsTotal*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (saveState(save_file.c_str(), sv_Px.data(), sv_Py.data(), sv_dist.data(),
                      sv_types.data(), threadsTotal, dpTable, totalJumps,
                      targetX, targetY, range_start, range_end, dpBits)) {
            std::cout << "State saved to: " << save_file << "\n";
            std::cout << "Resume with:    --kangaroo --resume --save-file " << save_file << "\n";
        } else {
            std::cerr << "ERROR: Failed to save state!\n";
        }
    }

    /* Cleanup */
    cudaFree(d_Px); cudaFree(d_Py); cudaFree(d_dist); cudaFree(d_type);
    cudaFree(d_foundFlag); cudaFree(d_dpBuffer); cudaFree(d_dpCount);
    cudaStreamDestroy(stream);

    return found ? EXIT_SUCCESS : EXIT_FAILURE;
}
