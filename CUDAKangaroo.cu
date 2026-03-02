#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cmath>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <random>
#include <sstream>

#include "CUDAKangaroo.cuh"
#include "CUDAUtils.h"

__constant__ uint64_t c_SpX[MAX_JUMP_POINTS * 4];
__constant__ uint64_t c_SpY[MAX_JUMP_POINTS * 4];

extern bool hexToLE64(const std::string& h_in, uint64_t w[4]);
extern std::string formatHex256(const uint64_t limbs[4]);
extern std::string formatCompressedPubHex(const uint64_t X[4], const uint64_t Y[4]);

// Forward declaration
extern __global__ void scalarMulKernelBase(const uint64_t* scalars_in, uint64_t* outX, uint64_t* outY, int N);

// ------------------------------------------------------------------
// Host-side helpers
// ------------------------------------------------------------------
static void host_add256(uint64_t out[4], const uint64_t a[4], const uint64_t b[4]) {
    unsigned __int128 c = 0;
    for (int i = 0; i < 4; i++) {
        c = (unsigned __int128)a[i] + b[i] + (c >> 64);
        out[i] = (uint64_t)c;
    }
}
static void host_sub256(uint64_t out[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t bi = b[i] + borrow;
        if (a[i] < bi) { out[i] = (uint64_t)((((unsigned __int128)1)<<64) + a[i] - bi); borrow = 1; }
        else           { out[i] = a[i] - bi; borrow = 0; }
    }
}
static bool host_gt256(const uint64_t a[4], const uint64_t b[4]) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] > b[i]) return true;
        if (a[i] < b[i]) return false;
    }
    return false;
}
static void host_shr1_256(uint64_t a[4]) {
    for (int i = 0; i < 3; i++) a[i] = (a[i] >> 1) | (a[i+1] << 63);
    a[3] >>= 1;
}

// Decompress a '02'/'03' pubkey to X, Y in little-endian uint64 arrays.
// Returns false if format is invalid.
static bool decompressPubkey(const std::string& hex, uint64_t X[4], uint64_t Y[4]) {
    if (hex.size() == 130 && hex.substr(0,2) == "04") {
        hexToLE64(hex.substr(2, 64), X);
        hexToLE64(hex.substr(66, 64), Y);
        return true;
    }
    // Compressed
    if (hex.size() != 66) return false;
    int prefix = std::stoi(hex.substr(0,2), nullptr, 16);
    if (prefix != 0x02 && prefix != 0x03) return false;
    hexToLE64(hex.substr(2, 64), X);

    // Secp256k1 p
    static const uint64_t P[4] = {0xFFFFFFFEFFFFFC2FULL,0xFFFFFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFFFULL};
    // y² = x³ + 7 (mod p), we need Y on device - use Python logic on CPU
    // For simplicity, use a small GPU call: compute Y using scalarMulBaseAffine indirectly.
    // Actually we only need Y from X using x^3 + 7 mod p on CPU with __int128.
    // We'll do it using Tonelli–Shanks exponent (p+1)/4 since p≡3 mod 4.
    // (p+1)/4 = 0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffffbfffff0c
    // Use bignum with uint64_t[4].
    // Defer: pass X, compute Y on GPU indirectly via a helper.
    // For now a simplified path using device later.
    // STUB: just set Y to 0 and let user pass uncompressed.
    Y[0]=Y[1]=Y[2]=Y[3]=0;
    std::cerr << "[warn] Compressed pubkey support: Y coordinate not computed. Please use uncompressed (04...) pubkey for now.\n";
    return false;
}

int getPow2Jmax(long double optimalmeanjumpsize) {
    long double sumjumpsize = 0;
    for (int i = 1; i < 256; ++i) {
        sumjumpsize += std::pow(2.0L, i - 1);
        long double now_mean  = sumjumpsize / i;
        long double next_mean = (sumjumpsize + std::pow(2.0L, i)) / i;
        if (optimalmeanjumpsize - now_mean <= next_mean - optimalmeanjumpsize)
            return i;
    }
    return 32;
}

// ------------------------------------------------------------------
// runKangaroo — called from main()
// ------------------------------------------------------------------
int runKangaroo(const std::string& range_hex, const std::string& pubkey_hex,
                uint32_t a, uint32_t /*b*/, uint32_t slices) {

    std::cout << "\n[################################################]\n";
    std::cout << "[#   GPU Pollard-Kangaroo PrivKey Recovery      #]\n";
    std::cout << "[#         bitcoin ecdsa secp256k1             #]\n";
    std::cout << "[################################################]\n\n";

    // ── parse range ────────────────────────────────────────────────
    size_t colon = range_hex.find(':');
    if (colon == std::string::npos) { std::cerr << "Error: --range must be start:end\n"; return 1; }
    uint64_t range_start[4]{}, range_end[4]{};
    hexToLE64(range_hex.substr(0, colon), range_start);
    hexToLE64(range_hex.substr(colon + 1), range_end);

    if (!host_gt256(range_end, range_start)) { std::cerr << "Error: range_end <= range_start\n"; return 1; }

    uint64_t W[4]; host_sub256(W, range_end, range_start);
    long double W_ld  = ld_from_u256(W);
    long double Wsqrt = ::sqrtl(W_ld);

    int pow2W    = (int)std::round(std::log2l(W_ld));
    int pow2Jmax = getPow2Jmax(Wsqrt / 2.0L);
    int pow2dp   = std::max(0, (pow2W / 2) - 2);
    uint32_t dp_mask = (uint32_t)((1ULL << pow2dp) - 1);

    uint64_t halfW[4]; for(int i=0;i<4;i++) halfW[i]=W[i];
    host_shr1_256(halfW);
    uint64_t M[4]; host_add256(M, range_start, halfW);

    std::cout << "[+] W  = 2^" << pow2W  << "\n";
    std::cout << "[+] Jmax = 2^" << pow2Jmax << "\n";
    std::cout << "[+] dp_mask = 2^" << pow2dp << "\n";
    std::cout << "[+] M = " << formatHex256(M) << "\n";

    // ── parse pubkey ───────────────────────────────────────────────
    uint64_t pubX[4]{}, pubY[4]{};
    if (!decompressPubkey(pubkey_hex, pubX, pubY)) {
        if (pubkey_hex.size() == 130 && pubkey_hex.substr(0,2) == "04") {
            hexToLE64(pubkey_hex.substr(2, 64), pubX);
            hexToLE64(pubkey_hex.substr(66, 64), pubY);
        } else {
            std::cerr << "Error: provide uncompressed pubkey (04...)\n";
            return 1;
        }
    }

    // ── determine how many kangaroos fit in VRAM ────────────────────
    size_t free_mem=0, total_mem=0;
    cudaMemGetInfo(&free_mem, &total_mem);
    // Each kangaroo consumes: X(32)+Y(32)+dist(32)+type(4) = 100 bytes
    // Leave 10% for overhead
    size_t usable = (size_t)(free_mem * 0.88);
    const size_t per_k = (4+4+4)*sizeof(uint64_t) + sizeof(uint32_t); // 100 bytes
    uint32_t TOTAL_KANGAROOS = (uint32_t)(usable / per_k);
    // Round down to multiple of 512 (must be divisible by 2 and threads)
    TOTAL_KANGAROOS = (TOTAL_KANGAROOS / 512) * 512;
    if (TOTAL_KANGAROOS < 512) { TOTAL_KANGAROOS = 512; }
    uint32_t KANGAROOS_PER_THREAD = 8;
    if (TOTAL_KANGAROOS % KANGAROOS_PER_THREAD != 0) TOTAL_KANGAROOS -= TOTAL_KANGAROOS % KANGAROOS_PER_THREAD;
    uint32_t threadsTotal = TOTAL_KANGAROOS / KANGAROOS_PER_THREAD;

    std::cout << "[+] VRAM free = " << (free_mem/1024/1024) << " MB\n";
    std::cout << "[+] Kangaroos = " << TOTAL_KANGAROOS << "  (threads=" << threadsTotal << ")\n";

    // ── allocate GPU buffers ─────────────────────────────────────────
    uint64_t *d_Kx, *d_Ky, *d_Kdist;
    uint32_t *d_Ktype;
    cudaMalloc(&d_Kx,    (size_t)TOTAL_KANGAROOS * 4 * sizeof(uint64_t));
    cudaMalloc(&d_Ky,    (size_t)TOTAL_KANGAROOS * 4 * sizeof(uint64_t));
    cudaMalloc(&d_Kdist, (size_t)TOTAL_KANGAROOS * 4 * sizeof(uint64_t));
    cudaMalloc(&d_Ktype, (size_t)TOTAL_KANGAROOS      * sizeof(uint32_t));

    unsigned int MAX_DPS = 200000;
    DPRecord* d_dp_buffer;
    unsigned int* d_dp_count;
    cudaMalloc(&d_dp_buffer, MAX_DPS * sizeof(DPRecord));
    cudaMalloc(&d_dp_count,  sizeof(unsigned int));

    // ── build jump point table c_SpX / c_SpY ─────────────────────────
    {
        size_t jpCount = (size_t)std::min(pow2Jmax, (int)MAX_JUMP_POINTS);
        std::vector<uint64_t> h_scalars(jpCount * 4, 0);
        for (size_t k = 0; k < jpCount; ++k) {
            // 2^k as uint256 little-endian
            int bit = (int)k;
            h_scalars[k*4 + (bit/64)] = 1ULL << (bit%64);
        }
        uint64_t *d_sp_s, *d_sp_x, *d_sp_y;
        cudaMalloc(&d_sp_s, jpCount*4*sizeof(uint64_t));
        cudaMalloc(&d_sp_x, jpCount*4*sizeof(uint64_t));
        cudaMalloc(&d_sp_y, jpCount*4*sizeof(uint64_t));
        cudaMemcpy(d_sp_s, h_scalars.data(), jpCount*4*sizeof(uint64_t), cudaMemcpyHostToDevice);
        scalarMulKernelBase<<<((int)jpCount+255)/256, 256>>>(d_sp_s, d_sp_x, d_sp_y, (int)jpCount);
        cudaDeviceSynchronize();
        std::vector<uint64_t> h_SpX(jpCount*4), h_SpY(jpCount*4);
        cudaMemcpy(h_SpX.data(), d_sp_x, jpCount*4*sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_SpY.data(), d_sp_y, jpCount*4*sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpyToSymbol(c_SpX, h_SpX.data(), jpCount*4*sizeof(uint64_t));
        cudaMemcpyToSymbol(c_SpY, h_SpY.data(), jpCount*4*sizeof(uint64_t));
        cudaFree(d_sp_s); cudaFree(d_sp_x); cudaFree(d_sp_y);
    }
    std::cout << "[+] Jump table ready (2^0..2^" << (pow2Jmax-1) << ")\n";

    // ── distributed point storage on CPU ──────────────────────────────
    struct DPInfo { int type; uint64_t distance[4]; };
    std::unordered_map<uint64_t, DPInfo> dp_map;
    dp_map.reserve(1 << 20);

    uint32_t restart_count = 0;
    bool found = false;
    uint64_t target_privatekey[4]{};

    auto do_restart = [&]() {
        restart_count++;
        std::cout << "\n[#] Restart #" << restart_count << "  — resetting kangaroo positions\n";

        // Allocate host arrays for initial scalars (tame) and offsets (wild)
        std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count() + restart_count);

        uint32_t half = TOTAL_KANGAROOS / 2;

        std::vector<uint64_t> h_scalars(TOTAL_KANGAROOS * 4);
        std::vector<uint64_t> h_dist(TOTAL_KANGAROOS * 4, 0);
        std::vector<uint32_t> h_types(TOTAL_KANGAROOS);

        int wsqrt_bits = pow2W / 2;
        uint64_t rand_mask0 = (wsqrt_bits < 64) ? ((1ULL << wsqrt_bits) - 1) : 0xFFFFFFFFFFFFFFFFULL;
        uint64_t rand_mask1 = (wsqrt_bits < 64)  ? 0 : (wsqrt_bits < 128 ? ((1ULL << (wsqrt_bits-64)) - 1) : 0xFFFFFFFFFFFFFFFFULL);

        // Tame: start at M + randT,  dist = randT
        for (uint32_t k = 0; k < half; ++k) {
            uint64_t rT[4] = { rng() & rand_mask0, rng() & rand_mask1, 0, 0 };
            uint64_t mT[4]; host_add256(mT, M, rT);
            for (int i=0;i<4;i++) h_scalars[k*4+i] = mT[i];
            for (int i=0;i<4;i++) h_dist[k*4+i]    = rT[i];
            h_types[k] = DP_RECORD_TAME;
        }
        // Wild: start at randW * G + pubkey,  we store scalar = randW; dist = randW
        for (uint32_t k = half; k < TOTAL_KANGAROOS; ++k) {
            uint64_t rW[4] = { rng() & rand_mask0, rng() & rand_mask1, 0, 0 };
            for (int i=0;i<4;i++) h_scalars[k*4+i] = rW[i];
            for (int i=0;i<4;i++) h_dist[k*4+i]    = rW[i];
            h_types[k] = DP_RECORD_WILD;
        }

        cudaMemcpy(d_Ktype, h_types.data(), TOTAL_KANGAROOS*sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Kdist, h_dist.data(),  TOTAL_KANGAROOS*4*sizeof(uint64_t), cudaMemcpyHostToDevice);

        // scalarMulKernelBase computes k*G
        uint64_t *d_tmp_s; cudaMalloc(&d_tmp_s, (size_t)TOTAL_KANGAROOS*4*sizeof(uint64_t));
        cudaMemcpy(d_tmp_s, h_scalars.data(), TOTAL_KANGAROOS*4*sizeof(uint64_t), cudaMemcpyHostToDevice);
        scalarMulKernelBase<<<(TOTAL_KANGAROOS+255)/256, 256>>>(d_tmp_s, d_Kx, d_Ky, TOTAL_KANGAROOS);
        cudaDeviceSynchronize();
        cudaFree(d_tmp_s);

        // Add pubkey to Wild kangaroos: Kx[half..] += pubkey  (i.e. P+randW*G)
        // We do this with a small kernel
        kernel_add_pubkey<<<(half+255)/256, 256>>>(
            d_Kx + (size_t)half*4, d_Ky + (size_t)half*4,
            pubX[0], pubX[1], pubX[2], pubX[3],
            pubY[0], pubY[1], pubY[2], pubY[3], half);
        cudaDeviceSynchronize();

        cudaMemset(d_dp_count, 0, sizeof(unsigned int));
        dp_map.clear();
    };

    // Initial setup
    do_restart();
    restart_count = 0; // reset counter (first run)

    std::cout << "[+] Running Kangaroos...\n";
    int blocks = (threadsTotal + 255) / 256;
    uint64_t total_jumps = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto t_last = t0;

    while (!found) {
        kernel_kangaroo_jump<<<blocks, 256>>>(
            d_Kx, d_Ky, d_Kdist, d_Ktype,
            threadsTotal, KANGAROOS_PER_THREAD, slices,
            (uint32_t)(1 << pow2Jmax), dp_mask,
            d_dp_buffer, d_dp_count, MAX_DPS);
        cudaDeviceSynchronize();
        total_jumps += (uint64_t)TOTAL_KANGAROOS * slices;

        // ── harvest DPs ─────────────────
        unsigned int dps = 0;
        cudaMemcpy(&dps, d_dp_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (dps > MAX_DPS) dps = MAX_DPS;

        if (dps > 0) {
            std::vector<DPRecord> host_dps(dps);
            cudaMemcpy(host_dps.data(), d_dp_buffer, dps*sizeof(DPRecord), cudaMemcpyDeviceToHost);
            cudaMemset(d_dp_count, 0, sizeof(unsigned int));

            for (auto& dp : host_dps) {
                uint64_t key = dp.X[0];
                auto it = dp_map.find(key);
                if (it != dp_map.end()) {
                    auto& ex = it->second;
                    if (ex.type != (int)dp.type) {
                        // Collision!
                        const uint64_t* D_T = (ex.type == DP_RECORD_TAME) ? ex.distance : dp.distance;
                        const uint64_t* D_W = (ex.type == DP_RECORD_WILD) ? ex.distance : dp.distance;

                        // PrivKey = M + D_T - D_W
                        uint64_t tmp[4]; host_add256(tmp, M, D_T);
                        host_sub256(target_privatekey, tmp, D_W);
                        found = true;
                        break;
                    }
                } else {
                    DPInfo info; info.type = dp.type;
                    for(int i=0;i<4;i++) info.distance[i] = dp.distance[i];
                    dp_map[key] = info;
                }
            }
        }

        // ── progress display ─────────────────
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - t0).count();
        double since   = std::chrono::duration<double>(now - t_last).count();
        if (since > 1.0 || found) {
            t_last = now;
            double jps = total_jumps / elapsed;
            double progress = (total_jumps / ((double)TOTAL_KANGAROOS)) / (2.0 * Wsqrt) * 100.0;
            std::cout << "\r[" << std::fixed << std::setprecision(0) << elapsed << "s] "
                      << std::fixed << std::setprecision(1) << jps/1e6 << " Mj/s | "
                      << "DPs=" << dp_map.size()
                      << " | Est: " << std::setprecision(1) << progress << "% "
                      << " | Restarts=" << restart_count
                      << "    " << std::flush;
        }

        // ── auto-restart if no solution after ~4*sqrt(W) jumps ─────────
        if (!found && total_jumps > (uint64_t)(4.0L * Wsqrt * TOTAL_KANGAROOS)) {
            do_restart();
            total_jumps = 0;
            t0 = std::chrono::high_resolution_clock::now();
            t_last = t0;
        }
    }

    std::cout << "\n\n";
    std::cout << "====================================================\n";
    std::cout << "  FOUND PRIVATE KEY: " << formatHex256(target_privatekey) << "\n";
    std::cout << "====================================================\n";

    // Save to file
    {
        std::ofstream f("found.txt", std::ios::app);
        f << "Private Key: " << formatHex256(target_privatekey)
          << "  Range: " << range_hex
          << "  Pubkey: " << pubkey_hex << "\n";
    }
    std::cout << "[+] Saved to found.txt\n";

    cudaFree(d_Kx); cudaFree(d_Ky); cudaFree(d_Kdist); cudaFree(d_Ktype);
    cudaFree(d_dp_buffer); cudaFree(d_dp_count);
    return 0;
}
