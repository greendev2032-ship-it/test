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

// ── secp256k1 prime: p = 2^256 - 2^32 - 977 ─────────────────────────
static const uint64_t CPU_P[4] = {
    0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
};
static void cpu_modsub256(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t borrow=0;
    for(int i=0;i<4;i++){
        uint64_t bi=b[i]+borrow;
        if(a[i]<bi){r[i]=(uint64_t)((((unsigned __int128)1)<<64)+a[i]-bi);borrow=1;}
        else{r[i]=a[i]-bi;borrow=0;}
    }
    if(borrow){ unsigned __int128 c=0; for(int i=0;i<4;i++){c=(unsigned __int128)r[i]+CPU_P[i]+(c>>64);r[i]=(uint64_t)c;} }
}
static void cpu_modadd256(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    unsigned __int128 c=0;
    for(int i=0;i<4;i++){c=(unsigned __int128)a[i]+b[i]+(c>>64);r[i]=(uint64_t)c;}
    bool ge=(c>>64)>0;
    if(!ge){for(int i=3;i>=0;i--){if(r[i]>CPU_P[i]){ge=true;break;}if(r[i]<CPU_P[i])break;}}
    if(ge){uint64_t bw=0;for(int i=0;i<4;i++){uint64_t bi=CPU_P[i]+bw;if(r[i]<bi){r[i]=(uint64_t)((((unsigned __int128)1)<<64)+r[i]-bi);bw=1;}else{r[i]-=bi;bw=0;}}}
}
static void cpu_reduce512(uint64_t full[8], uint64_t out[4]) {
    static const uint64_t K = 0x1000003D1ULL;
    unsigned __int128 acc=0; unsigned __int128 carry=0;
    uint64_t hi[5]={};
    for(int i=0;i<4;i++){acc=(unsigned __int128)full[i+4]*K+(acc>>64);hi[i]=(uint64_t)acc;}
    hi[4]=(uint64_t)(acc>>64);
    acc=0;
    for(int i=0;i<4;i++){acc=(unsigned __int128)full[i]+hi[i]+(acc>>64);out[i]=(uint64_t)acc;}
    uint64_t c=(uint64_t)(acc>>64)+hi[4];
    if(c){acc=(unsigned __int128)c*K;uint64_t add=(uint64_t)acc,cy2=(uint64_t)(acc>>64),prev=0;
        for(int i=0;i<4;i++){acc=(unsigned __int128)out[i]+(i==0?add:0)+(i==0?cy2:prev)+(acc>>64);out[i]=(uint64_t)acc;prev=0;}}
    bool ge=false;
    for(int i=3;i>=0;i--){if(out[i]>CPU_P[i]){ge=true;break;}if(out[i]<CPU_P[i])break;}
    if(ge){uint64_t bw=0;for(int i=0;i<4;i++){uint64_t bi=CPU_P[i]+bw;if(out[i]<bi){out[i]=(uint64_t)((((unsigned __int128)1)<<64)+out[i]-bi);bw=1;}else{out[i]-=bi;bw=0;}}}
}
static void cpu_modmul256(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t full[8]={};
    unsigned __int128 carry;
    for(int i=0;i<4;i++){carry=0;for(int j=0;j<4;j++){unsigned __int128 p=(unsigned __int128)a[i]*b[j]+full[i+j]+carry;full[i+j]=(uint64_t)p;carry=p>>64;}full[i+4]+=carry;}
    cpu_reduce512(full,r);
}
static void cpu_modsqr256(const uint64_t a[4], uint64_t r[4]){ cpu_modmul256(a,a,r); }
static void cpu_modexp256(const uint64_t base[4], const uint64_t exp[4], uint64_t r[4]) {
    uint64_t res[4]={1,0,0,0}, b[4]; for(int i=0;i<4;i++) b[i]=base[i];
    for(int i=0;i<256;i++){if((exp[i/64]>>(i%64))&1){cpu_modmul256(res,b,res);}cpu_modsqr256(b,b);}
    for(int i=0;i<4;i++) r[i]=res[i];
}
// Decompress a '02'/'03' or '04' pubkey to X, Y in little-endian uint64 arrays.
static bool decompressPubkey(const std::string& hex, uint64_t X[4], uint64_t Y[4]) {
    if (hex.size() == 130) {
        hexToLE64(hex.substr(2, 64), X);
        hexToLE64(hex.substr(66, 64), Y);
        return true;
    }
    if (hex.size() != 66) return false;
    int prefix = std::stoi(hex.substr(0,2), nullptr, 16);
    if (prefix != 0x02 && prefix != 0x03) return false;
    int y_parity = prefix & 1; // 02=even=0, 03=odd=1
    hexToLE64(hex.substr(2, 64), X);

    // y = (x^3 + 7)^((p+1)/4) mod p
    uint64_t x2[4], x3[4], rhs[4], seven[4]={7,0,0,0};
    cpu_modsqr256(X, x2);
    cpu_modmul256(X, x2, x3);
    cpu_modadd256(x3, seven, rhs);

    // (p+1)/4 in little-endian
    uint64_t exp[4] = {
        0xFFFFFFFFBFFFFF0CULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
        0x3FFFFFFFFFFFFFFFULL
    };
    cpu_modexp256(rhs, exp, Y);

    // adjust parity
    if ((int)(Y[0] & 1) != y_parity) cpu_modsub256(CPU_P, Y, Y);
    return true;
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

    uint64_t halfW[4]; for(int i=0;i<4;i++) halfW[i]=W[i];
    host_shr1_256(halfW);
    uint64_t M[4]; host_add256(M, range_start, halfW);

    std::cout << "[+] W  = 2^" << pow2W  << "\n";
    std::cout << "[+] Jmax = 2^" << pow2Jmax << "\n";
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

    // ── determine how many kangaroos fit / are needed ────────────────
    size_t free_mem=0, total_mem=0;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t usable = (size_t)(free_mem * 0.70);
    const size_t per_k = (4+4+4)*sizeof(uint64_t) + sizeof(uint32_t); // 100 bytes
    uint32_t TOTAL_KANGAROOS = (uint32_t)(usable / per_k);

    // Round down to multiple of 512
    TOTAL_KANGAROOS = (TOTAL_KANGAROOS / 512) * 512;
    if (TOTAL_KANGAROOS < 512) TOTAL_KANGAROOS = 512;

    const uint32_t KANGAROOS_PER_THREAD = 8;
    if (TOTAL_KANGAROOS % KANGAROOS_PER_THREAD != 0) TOTAL_KANGAROOS -= TOTAL_KANGAROOS % KANGAROOS_PER_THREAD;
    uint32_t threadsTotal = TOTAL_KANGAROOS / KANGAROOS_PER_THREAD;

    std::cout << "[+] VRAM free     = " << (free_mem/1024/1024) << " MB\n";
    std::cout << "[+] Kangaroos     = " << TOTAL_KANGAROOS << "  (threads=" << threadsTotal << ")\n";

    // ── Calculate Distingished Point Mask ───────────────────────────
    unsigned int MAX_DPS = 200000;
    // To prevent the CPU hashmap from exploding and causing OOM freezes,
    // we need to strictly control the DP generation rate.
    // If the GPU does ~2 Billion jumps/s, and we want max 50,000 DPs/s,
    // dp_mask should be 2B / 50k = 40,000 -> ~2^15.
    int pow2dp = 12; // safe default
    long double expected_jumps_total = 4.0L * Wsqrt;
    long double optimal_dp_mod = expected_jumps_total / (MAX_DPS * 0.5L);
    
    if (TOTAL_KANGAROOS > 1000000) pow2dp = 15; // force high mask for big K
    else if (optimal_dp_mod > 1.0L) pow2dp = (int)std::round(std::log2l(optimal_dp_mod));

    if (pow2dp > 20) pow2dp = 20;
    if (pow2dp < 0) pow2dp = 0;
    uint32_t dp_mask = (uint32_t)((1ULL << pow2dp) - 1);

    std::cout << "[+] dp_mask = 2^" << pow2dp << " (1 DP per " << (1ULL<<pow2dp) << " jumps)\n";

    // ── allocate GPU buffers ─────────────────────────────────────────
    uint64_t *d_Kx, *d_Ky, *d_Kdist;
    uint32_t *d_Ktype;
    cudaMalloc(&d_Kx,    (size_t)TOTAL_KANGAROOS * 4 * sizeof(uint64_t));
    cudaMalloc(&d_Ky,    (size_t)TOTAL_KANGAROOS * 4 * sizeof(uint64_t));
    cudaMalloc(&d_Kdist, (size_t)TOTAL_KANGAROOS * 4 * sizeof(uint64_t));
    cudaMalloc(&d_Ktype, (size_t)TOTAL_KANGAROOS      * sizeof(uint32_t));

    // Verify allocations succeeded
    cudaError_t cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        std::cerr << "[ERROR] GPU buffer allocation failed: " << cudaGetErrorString(cerr)
                  << " (tried " << (per_k*(size_t)TOTAL_KANGAROOS/1024/1024) << " MB)\n";
        return 1;
    }

    DPRecord* d_dp_buffer;
    unsigned int* d_dp_count;
    cudaMalloc(&d_dp_buffer, MAX_DPS * sizeof(DPRecord));
    cudaMalloc(&d_dp_count,  sizeof(unsigned int));

    // ── build jump point table c_SpX / c_SpY ─────────────────────────
    {
        size_t jpCount = (size_t)std::min((size_t)pow2Jmax, (size_t)MAX_JUMP_POINTS);
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

    uint64_t target_privatekey[4]{};
    bool found = false;
    uint32_t restart_count = 0;

    // Fast GPU initialization kernel instead of slow CPU loop
    auto do_init = [&]() {
        restart_count++;
        uint32_t half = TOTAL_KANGAROOS / 2;
        int w_bits = pow2W;
        if (w_bits > 128) w_bits = 128;
        uint64_t mask0 = (w_bits >= 64) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << w_bits) - 1);
        uint64_t mask1 = (w_bits <= 64) ? 0 : (w_bits >= 128 ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << (w_bits - 64)) - 1));

        std::cout << "\n[#] Initializing " << TOTAL_KANGAROOS << " Kangaroos on GPU...\n";

        // Instead of writing a whole new kernel file, we can do a simple lambda-like 
        // kernel or just reuse d_Kx/d_Ky/d_Ktype with a tiny raw CUDA string... 
        // Wait, nvcc needs proper kernels. Let's just create host arrays but use OpenMP or 
        // very fast PRNG to fill them in less than 1 second.
        
        std::vector<uint64_t> h_scalars(TOTAL_KANGAROOS * 4);
        std::vector<uint64_t> h_dist(TOTAL_KANGAROOS * 4, 0);
        std::vector<uint32_t> h_types(TOTAL_KANGAROOS);

        // Fast LCG instead of mt19937_64 which is very slow for 136 million calls
        uint64_t seed = std::chrono::steady_clock::now().time_since_epoch().count();
        
        #pragma omp parallel for
        for (int k = 0; k < (int)TOTAL_KANGAROOS; ++k) {
            // Simple fast Xorshift/LCG per thread
            uint64_t rstate = seed + k * 19937ULL;
            auto fast_rand = [&]() -> uint64_t {
                rstate ^= rstate >> 12; rstate ^= rstate << 25; rstate ^= rstate >> 27;
                return rstate * 0x2545F4914F6CDD1DULL;
            };

            uint64_t r[4] = { fast_rand() & mask0, fast_rand() & mask1, 0, 0 };
            
            if (k < (int)half) {
                // Tame
                uint64_t mT[4]; 
                unsigned __int128 c = 0;
                for (int i=0; i<4; i++) { c = (unsigned __int128)M[i] + r[i] + (c>>64); mT[i] = (uint64_t)c; }
                
                h_types[k] = DP_RECORD_TAME;
                for (int i=0;i<4;i++) { h_scalars[k*4+i] = mT[i]; h_dist[k*4+i] = r[i]; }
            } else {
                // Wild
                h_types[k] = DP_RECORD_WILD;
                for (int i=0;i<4;i++) { h_scalars[k*4+i] = r[i]; h_dist[k*4+i] = r[i]; }
            }
        }

        cudaMemcpy(d_Ktype, h_types.data(), TOTAL_KANGAROOS*sizeof(uint32_t), cudaMemcpyHostToDevice);

        // Use d_Kdist as temporary buffer
        cudaMemcpy(d_Kdist, h_scalars.data(), TOTAL_KANGAROOS*4*sizeof(uint64_t), cudaMemcpyHostToDevice);
        scalarMulKernelBase<<<(TOTAL_KANGAROOS+255)/256, 256>>>(d_Kdist, d_Kx, d_Ky, TOTAL_KANGAROOS);
        cudaDeviceSynchronize();
        cudaMemcpy(d_Kdist, h_dist.data(), TOTAL_KANGAROOS*4*sizeof(uint64_t), cudaMemcpyHostToDevice);

        kernel_add_pubkey<<<(half+255)/256, 256>>>(
            d_Kx + (size_t)half*4, d_Ky + (size_t)half*4,
            pubX[0], pubX[1], pubX[2], pubX[3],
            pubY[0], pubY[1], pubY[2], pubY[3], half);
        cudaDeviceSynchronize();

        cudaMemset(d_dp_count, 0, sizeof(unsigned int));
        dp_map.clear();
    };

    // Initial setup
    do_init();

    std::cout << "[+] Running Kangaroos...\n";
    int blocks = (threadsTotal + 255) / 256;
    uint64_t total_jumps = 0;
    uint64_t session_jumps = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto t_last = t0;

    // Calculate restart threshold (jumps PER kangaroo).
    // Ensure they jump at least 4 * dp_mask to allow trails to hit a DP.
    double req_jumps_per_K = (4.0 * Wsqrt) / (double)TOTAL_KANGAROOS;
    double min_jumps_for_dp = 4.0 * (1ULL << pow2dp);
    double restart_threshold_per_K = std::max(req_jumps_per_K, min_jumps_for_dp);
    uint64_t session_jumps_limit = (uint64_t)(restart_threshold_per_K * TOTAL_KANGAROOS);

    while (!found) {
        kernel_kangaroo_jump<<<blocks, 256>>>(
            d_Kx, d_Ky, d_Kdist, d_Ktype,
            threadsTotal, KANGAROOS_PER_THREAD, slices,
            (uint32_t)pow2Jmax, dp_mask,
            d_dp_buffer, d_dp_count, MAX_DPS);
        cudaDeviceSynchronize();
        uint64_t batch_jumps = (uint64_t)TOTAL_KANGAROOS * slices;
        total_jumps += batch_jumps;
        session_jumps += batch_jumps;

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
            double progress = ((double)session_jumps / (double)session_jumps_limit) * 100.0;
            std::cout << "\r[" << std::fixed << std::setprecision(0) << elapsed << "s] "
                      << std::fixed << std::setprecision(1) << jps/1e6 << " Mj/s | "
                      << "DPs: " << dp_map.size()
                      << " | Est: " << std::setprecision(2) << progress << "% "
                      << " | Res: " << (restart_count - 1)
                      << "    " << std::flush;
        }

        // Restart loop to prevent DP map from ballooning infinitely and to reset dead kangaroos
        if (!found && session_jumps >= session_jumps_limit) {
            do_init();
            session_jumps = 0;
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
