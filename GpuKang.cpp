// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"

#include "GpuKang.h"

// Remove these declarations that have default parameters
// cudaError_t cuSetGpuParams(TKparams Kparams, u64* _jmp2_table);
// void CallGpuKernelGen(TKparams Kparams, cudaStream_t stream = 0);
// void CallGpuKernelABC(TKparams Kparams, cudaStream_t stream = 0);

// Keep only the extern "C" declarations
extern "C" void CallGpuKernelABC(TKparams Kparams, cudaStream_t stream);
extern "C" void CallGpuKernelGen(TKparams Kparams, cudaStream_t stream);
extern "C" cudaError_t cuSetGpuParams(TKparams Kparams, u64* _jmp2_table);
void AddPointsToList(u32* data, int cnt, u64 ops_cnt);
extern bool gGenMode; //tames generation mode

// Add CUDA stream declarations
cudaStream_t computeStream;
cudaStream_t memoryStream;

// Calculate GPU score - RTX 5090 optimized calculation
static u32 get_gpu_score(cudaDeviceProp* props)
{
	u32 w = props->multiProcessorCount * 12;
    
    // Enhanced support for RTX 5090
    if (props->major >= 9) {
        printf("RTX 5090 or newer GPU architecture detected (compute capability %d.%d)\n", 
               props->major, props->minor);
        // For RTX 5090, use improved scoring formula
        w = (u32)(props->multiProcessorCount * 14);
    }
    
	return w;
}

// Check if we have an RTX 5090 or newer
static bool isRTX5090OrNewer(cudaDeviceProp* props) {
    return (props->major >= 9 || 
           (props->major == 8 && props->minor >= 9) || 
           (strstr(props->name, "RTX 50") != NULL));
}

int RCGpuKang::CalcKangCnt()
{
    // Determine optimal parameters based on GPU type
    if (IsOldGpu) {
        // Parameters for older GPUs remain unchanged
        Kparams.BlockCnt = mpCnt;
        Kparams.BlockSize = 512;
        Kparams.GroupCnt = 64;
    } else {
        // Optimized parameters for RTX 5090
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, CudaIndex);
        
        if (deviceProp.major >= 9) {
            printf("Optimizing kernel parameters for RTX 5090...\n");
            
            // For RTX 5090, we need to be more conservative with memory usage
            // and optimize for the architecture
            Kparams.BlockCnt = mpCnt > 80 ? 80 : mpCnt; // Limit SM usage
            Kparams.BlockSize = 256; // Better for L1 cache utilization 
            
            // For high SM count like RTX 5090 (170 SMs)
            printf("Using high-density configuration for %d SMs\n", mpCnt);
            Kparams.GroupCnt = 16;  // Significantly reduced from original 64/32
            
            // Calculate estimated DPs per kangaroo for display purposes
            float estDPsPerKang = powf(2.0f, (float)(64 - Kparams.DP));
            printf("Estimated DPs per kangaroo: %.3f.\n", estDPsPerKang);
        } else {
            // For RTX 3000/4000 series
            Kparams.BlockCnt = mpCnt;
            Kparams.BlockSize = 256;
            Kparams.GroupCnt = 32;
        }
    }
    
    // Calculate total kangaroo count with hard ceiling
    int totalKangCnt = Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
    const int MAX_SAFE_KANG_CNT = 1000000; // Hard limit to prevent segfault
    
    if (totalKangCnt > MAX_SAFE_KANG_CNT) {
        printf("Limiting BlockCnt to prevent excessive memory usage\n");
        totalKangCnt = MAX_SAFE_KANG_CNT;
        
        // Recalculate parameter to fit within limit
        Kparams.BlockCnt = MAX_SAFE_KANG_CNT / (Kparams.BlockSize * Kparams.GroupCnt);
        printf("Adjusted BlockCnt to %d\n", Kparams.BlockCnt);
    }
    
    return Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
}

//executes in main thread
bool RCGpuKang::Prepare(EcPoint _PntToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3)
{
    PntToSolve = _PntToSolve;
    Range = _Range;
    DP = _DP;
    EcJumps1 = _EcJumps1;
    EcJumps2 = _EcJumps2;
    EcJumps3 = _EcJumps3;
    StopFlag = false;
    Failed = false;
    u64 total_mem = 0;
    memset(dbg, 0, sizeof(dbg));
    memset(SpeedStats, 0, sizeof(SpeedStats));
    cur_stats_ind = 0;

    cudaError_t err;
    err = cudaSetDevice(CudaIndex);
    if (err != cudaSuccess) {
        printf("GPU %d, cudaSetDevice failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }
    
    // Create CUDA streams for better concurrency
    err = cudaStreamCreate(&computeStream);
    if (err != cudaSuccess) {
        printf("GPU %d, create compute stream failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }
    
    err = cudaStreamCreate(&memoryStream);
    if (err != cudaSuccess) {
        printf("GPU %d, create memory stream failed: %s\n", CudaIndex, cudaGetErrorString(err));
        cudaStreamDestroy(computeStream);
        return false;
    }

    // Set stream priorities with error checking
    int priority_high, priority_low;
    err = cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    if (err != cudaSuccess) {
        printf("Warning: Could not get stream priority range: %s\n", cudaGetErrorString(err));
        // Continue anyway with default priorities
    } else {
        cudaStream_t temp_stream;
        err = cudaStreamCreateWithPriority(&temp_stream, cudaStreamNonBlocking, priority_high);
        if (err == cudaSuccess) {
            cudaStreamDestroy(computeStream);
            computeStream = temp_stream;
        } else {
            printf("Warning: Failed to set compute stream priority: %s\n", cudaGetErrorString(err));
        }
    }

    // For RTX 5090, optimize memory settings
    cudaDeviceProp deviceProp;
    err = cudaGetDeviceProperties(&deviceProp, CudaIndex);
    if (err != cudaSuccess) {
        printf("GPU %d, failed to get device properties: %s\n", CudaIndex, cudaGetErrorString(err));
        cudaStreamDestroy(computeStream);
        cudaStreamDestroy(memoryStream);
        return false;
    }
    
    // Determine safe parameters for RTX 5090
    if (deviceProp.major >= 9) {
        printf("Setting conservative parameters for RTX 5090 to avoid memory issues...\n");
        
        // For RTX 5090, limit block count to prevent excessive memory usage
        if (mpCnt > 100) {
            Kparams.BlockCnt = 60; // Much more conservative limit
            printf("Limiting BlockCnt to %d (from %d SMs) to prevent excessive memory usage\n", 
                   Kparams.BlockCnt, mpCnt);
        } else {
            Kparams.BlockCnt = mpCnt;
        }
        
        // Use balanced values for BlockSize and GroupCnt
        Kparams.BlockSize = 256;
        Kparams.GroupCnt = 12; // Further reduced for safety
    } else {
        // For non-RTX 5090 GPUs, use standard parameters
        Kparams.BlockCnt = mpCnt;
        Kparams.BlockSize = IsOldGpu ? 512 : 256;
        Kparams.GroupCnt = IsOldGpu ? 64 : 32;
    }
    
    // Calculate safe kangaroo count
    KangCnt = Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
    
    // Safety check - hard limit on kangaroo count
    const u64 MAX_SAFE_KANG_CNT = 250000; // Significantly reduced limit to avoid segfault
    if (KangCnt > MAX_SAFE_KANG_CNT) {
        printf("Warning: Reducing kangaroo count from %d to %llu to prevent memory issues\n", 
               KangCnt, MAX_SAFE_KANG_CNT);
        KangCnt = MAX_SAFE_KANG_CNT;
        
        // Recalculate parameters to match the reduced KangCnt
        Kparams.GroupCnt = static_cast<u32>(KangCnt / (Kparams.BlockSize * Kparams.BlockCnt));
        if (Kparams.GroupCnt < 8) {
            Kparams.GroupCnt = 8;
            Kparams.BlockCnt = static_cast<u32>(KangCnt / (Kparams.BlockSize * Kparams.GroupCnt));
        }
    }
    
    // Set kernel parameters
    Kparams.KangCnt = KangCnt;
    Kparams.DP = DP;
    
    // For RTX 5090, set strict shared memory sizes with good safety margins
    size_t maxSharedMemPerBlock = deviceProp.sharedMemPerBlock;
    
    // Calculate shared memory sizes with very conservative safety margins
    Kparams.KernelA_LDS_Size = (unsigned int)(maxSharedMemPerBlock * 0.6);
    Kparams.KernelB_LDS_Size = (unsigned int)(maxSharedMemPerBlock * 0.5);
    Kparams.KernelC_LDS_Size = (unsigned int)(maxSharedMemPerBlock * 0.5);
    Kparams.IsGenMode = gGenMode;

    // Allocate memory with safe sizes for RTX 5090
    // Use a significantly reduced MAX_DP_CNT value for RTX 5090 to limit memory usage
    int effectiveMaxDpCnt = MAX_DP_CNT;
    if (!IsOldGpu && deviceProp.major >= 9) {
        effectiveMaxDpCnt = 16384; // Use a smaller value for RTX 5090
    }
    
    //allocate gpu mem
    u64 size;
    if (!IsOldGpu)
    {
        //L2    
        int L2size = Kparams.KangCnt * (3 * 32);
        total_mem += L2size;
        
        err = cudaMalloc((void**)&Kparams.L2, L2size);
        if (err != cudaSuccess)
        {
            printf("GPU %d, Allocate L2 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
            return false;
        }
    }
    
    // Allocate memory with safe sizes for RTX 5090
    size = effectiveMaxDpCnt * GPU_DP_SIZE + 16;
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.DPs_out, size);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate GpuOut memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    // For RTX 5090, we use a slightly different memory allocation size for kangaroos
    size = KangCnt * 96;
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.Kangs, size);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate pKangs memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    total_mem += JMP_CNT * 96;
    err = cudaMalloc((void**)&Kparams.Jumps1, JMP_CNT * 96);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate Jumps1 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    total_mem += JMP_CNT * 96;
    err = cudaMalloc((void**)&Kparams.Jumps2, JMP_CNT * 96);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate Jumps2 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    total_mem += JMP_CNT * 96;
    err = cudaMalloc((void**)&Kparams.Jumps3, JMP_CNT * 96);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate Jumps3 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    // Use a significantly reduced STEP_CNT value for RTX 5090 to limit memory usage
    int effectiveStepCnt = STEP_CNT;
    if (!IsOldGpu && deviceProp.major >= 9) {
        effectiveStepCnt = 128; // Drastically reduced step count for RTX 5090
    }

    // Allocate JumpsList with the effective STEP_CNT
    size = 2 * (u64)KangCnt * effectiveStepCnt;
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.JumpsList, size);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate JumpsList memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    // For RTX 5090, we need to limit the DP table size
    int effectiveDpTableMaxCnt = DPTABLE_MAX_CNT;
    if (!IsOldGpu && deviceProp.major >= 9) {
        effectiveDpTableMaxCnt = 8; // Significantly reduced from default
    }
    
    size = (u64)KangCnt * (16 * effectiveDpTableMaxCnt + sizeof(u32)); 
    // Add a safety check for excessively large allocations
    if (size > 2ULL * 1024 * 1024 * 1024) { // If over 2GB
        printf("Warning: DP table size is very large (%llu GB), reducing to prevent segfault\n", 
               size / (1024 * 1024 * 1024));
        size = 2ULL * 1024 * 1024 * 1024; // Limit to 2GB
    }
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.DPTable, size);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate DPTable memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    size = mpCnt * Kparams.BlockSize * sizeof(u64);
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.L1S2, size);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate L1S2 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    // For RTX 5090, reduce MD_LEN for better memory management
    int effectiveMdLen = MD_LEN;
    if (!IsOldGpu && deviceProp.major >= 9) {
        effectiveMdLen = 8; // Reduced from default
    }

    size = (u64)KangCnt * effectiveMdLen * (2 * 32);
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.LastPnts, size);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate LastPnts memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    size = (u64)KangCnt * effectiveMdLen * sizeof(u64);
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.LoopTable, size);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate LoopTable memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    total_mem += 1024;
    err = cudaMalloc((void**)&Kparams.dbg_buf, 1024);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate dbg_buf memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    size = sizeof(u32) * KangCnt + 8;
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.LoopedKangs, size);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate LoopedKangs memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    // Allocate host memory with error checking
    DPs_out = (u32*)malloc(effectiveMaxDpCnt * GPU_DP_SIZE);
    if (DPs_out == NULL) {
        printf("Failed to allocate host memory for DPs_out\n");
        return false;
    }

//jmp1
    u64* buf = (u64*)malloc(JMP_CNT * 96);
    if (buf == NULL) {
        printf("Failed to allocate memory for jumps buffer\n");
        return false;
    }
    
    for (int i = 0; i < JMP_CNT; i++)
    {
        memcpy(buf + i * 12, EcJumps1[i].p.x.data, 32);
        memcpy(buf + i * 12 + 4, EcJumps1[i].p.y.data, 32);
        memcpy(buf + i * 12 + 8, EcJumps1[i].dist.data, 32);
    }
    err = cudaMemcpy(Kparams.Jumps1, buf, JMP_CNT * 96, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("GPU %d, cudaMemcpy Jumps1 failed: %s\n", CudaIndex, cudaGetErrorString(err));
        free(buf);
        return false;
    }
    free(buf);
//jmp2
    buf = (u64*)malloc(JMP_CNT * 96);
    if (buf == NULL) {
        printf("Failed to allocate memory for jumps buffer\n");
        return false;
    }
    
    u64* jmp2_table = (u64*)malloc(JMP_CNT * 64);
    if (jmp2_table == NULL) {
        printf("Failed to allocate memory for jmp2_table\n");
        free(buf);
        return false;
    }
    
    for (int i = 0; i < JMP_CNT; i++)
    {
        memcpy(buf + i * 12, EcJumps2[i].p.x.data, 32);
        memcpy(jmp2_table + i * 8, EcJumps2[i].p.x.data, 32);
        memcpy(buf + i * 12 + 4, EcJumps2[i].p.y.data, 32);
        memcpy(jmp2_table + i * 8 + 4, EcJumps2[i].p.y.data, 32);
        memcpy(buf + i * 12 + 8, EcJumps2[i].dist.data, 32);
    }
    err = cudaMemcpy(Kparams.Jumps2, buf, JMP_CNT * 96, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("GPU %d, cudaMemcpy Jumps2 failed: %s\n", CudaIndex, cudaGetErrorString(err));
        free(buf);
        free(jmp2_table);
        return false;
    }
    free(buf);

    err = cuSetGpuParams(Kparams, jmp2_table);
    if (err != cudaSuccess)
    {
        free(jmp2_table);
        printf("GPU %d, cuSetGpuParams failed: %s!\r\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }
    free(jmp2_table);
//jmp3
    buf = (u64*)malloc(JMP_CNT * 96);
    if (buf == NULL) {
        printf("Failed to allocate memory for jumps buffer\n");
        return false;
    }
    
    for (int i = 0; i < JMP_CNT; i++)
    {
        memcpy(buf + i * 12, EcJumps3[i].p.x.data, 32);
        memcpy(buf + i * 12 + 4, EcJumps3[i].p.y.data, 32);
        memcpy(buf + i * 12 + 8, EcJumps3[i].dist.data, 32);
    }
    err = cudaMemcpy(Kparams.Jumps3, buf, JMP_CNT * 96, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("GPU %d, cudaMemcpy Jumps3 failed: %s\n", CudaIndex, cudaGetErrorString(err));
        free(buf);
        return false;
    }
    free(buf);

    printf("GPU %d: allocated %llu MB, %d kangaroos. OldGpuMode: %s\n", CudaIndex, total_mem / (1024 * 1024), KangCnt, IsOldGpu ? "Yes" : "No");
    return true;
}

void RCGpuKang::Release()
{
    free(RndPnts);
    free(DPs_out);
    cudaFree(Kparams.LoopedKangs);
    cudaFree(Kparams.dbg_buf);
    cudaFree(Kparams.LoopTable);
    cudaFree(Kparams.LastPnts);
    cudaFree(Kparams.L1S2);
    cudaFree(Kparams.DPTable);
    cudaFree(Kparams.JumpsList);
    cudaFree(Kparams.Jumps3);
    cudaFree(Kparams.Jumps2);
    cudaFree(Kparams.Jumps1);
    cudaFree(Kparams.Kangs);
    cudaFree(Kparams.DPs_out);
    if (!IsOldGpu)
        cudaFree(Kparams.L2);
    
    // Destroy streams when done
    cudaStreamDestroy(computeStream);
    cudaStreamDestroy(memoryStream);
}

void RCGpuKang::Stop()
{
    StopFlag = true;
}

void RCGpuKang::GenerateRndDistances()
{
    for (int i = 0; i < KangCnt; i++)
    {
        EcInt d;
        if (i < KangCnt / 3)
            d.RndBits(Range - 4); //TAME kangs
        else
        {
            d.RndBits(Range - 1);
            d.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
        }
        memcpy(RndPnts[i].priv, d.data, 24);
    }
}

bool RCGpuKang::Start()
{
    if (Failed)
        return false;

    cudaError_t err;
    err = cudaSetDevice(CudaIndex);
    if (err != cudaSuccess)
        return false;

    HalfRange.Set(1);
    HalfRange.ShiftLeft(Range - 1);
    PntHalfRange = ec.MultiplyG(HalfRange);
    NegPntHalfRange = PntHalfRange;
    NegPntHalfRange.y.NegModP();

    PntA = ec.AddPoints(PntToSolve, NegPntHalfRange);
    PntB = PntA;
    PntB.y.NegModP();

    RndPnts = (TPointPriv*)malloc(KangCnt * 96);
    if (RndPnts == NULL) {
        printf("Failed to allocate memory for RndPnts\n");
        return false;
    }
    
    GenerateRndDistances();

    //calc on GPU - it's faster
    u8 buf_PntA[64], buf_PntB[64];
    PntA.SaveToBuffer64(buf_PntA);
    PntB.SaveToBuffer64(buf_PntB);
    for (int i = 0; i < KangCnt; i++)
    {
        if (i < KangCnt / 3)
            memset(RndPnts[i].x, 0, 64);
        else
            if (i < 2 * KangCnt / 3)
                memcpy(RndPnts[i].x, buf_PntA, 64);
            else
                memcpy(RndPnts[i].x, buf_PntB, 64);
    }
    
    // Copy to GPU with error checking
    err = cudaMemcpy(Kparams.Kangs, RndPnts, KangCnt * 96, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("GPU %d, cudaMemcpy failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }
    
    // Launch with error checking
    cudaError_t kernelErr;
    CallGpuKernelGen(Kparams, computeStream);
    kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        printf("GPU %d, CallGpuKernelGen failed: %s\n", CudaIndex, cudaGetErrorString(kernelErr));
        return false;
    }
    
    // Wait for generation to complete
    err = cudaStreamSynchronize(computeStream);
    if (err != cudaSuccess) {
        printf("GPU %d, stream synchronize failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    // Initialize other buffers with error checking
    err = cudaMemset(Kparams.L1S2, 0, mpCnt * Kparams.BlockSize * sizeof(u64));
    if (err != cudaSuccess) {
        printf("GPU %d, cudaMemset L1S2 failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }
    
    err = cudaMemset(Kparams.dbg_buf, 0, 1024);
    if (err != cudaSuccess) {
        printf("GPU %d, cudaMemset dbg_buf failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }
    
    err = cudaMemset(Kparams.LoopTable, 0, KangCnt * MD_LEN * sizeof(u64));
    if (err != cudaSuccess) {
        printf("GPU %d, cudaMemset LoopTable failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }
    
    return true;
}

#ifdef DEBUG_MODE
int RCGpuKang::Dbg_CheckKangs()
{
    int kang_size = mpCnt * Kparams.BlockSize * Kparams.GroupCnt * 96;
    u64* kangs = (u64*)malloc(kang_size);
    cudaError_t err = cudaMemcpy(kangs, Kparams.Kangs, kang_size, cudaMemcpyDeviceToHost);
    int res = 0;
    for (int i = 0; i < KangCnt; i++)
    {
        EcPoint Pnt, p;
        Pnt.LoadFromBuffer64((u8*)&kangs[i * 12 + 0]);
        EcInt dist;
        dist.Set(0);
        memcpy(dist.data, &kangs[i * 12 + 8], 24);
        bool neg = false;
        if (dist.data[2] >> 63)
        {
            neg = true;
            memset(((u8*)dist.data) + 24, 0xFF, 16);
            dist.Neg();
        }
        p = ec.MultiplyG_Fast(dist);
        if (neg)
            p.y.NegModP();
        if (i < KangCnt / 3)
            p = p;
        else
            if (i < 2 * KangCnt / 3)
                p = ec.AddPoints(PntA, p);
            else
                p = ec.AddPoints(PntB, p);
        if (!p.IsEqual(Pnt))
            res++;
    }
    free(kangs);
    return res;
}
#endif

extern u32 gTotalErrors;

// Executes in separate thread
void RCGpuKang::Execute()
{
    cudaError_t err = cudaSetDevice(CudaIndex);
    if (err != cudaSuccess) {
        printf("GPU %d, cudaSetDevice failed: %s\n", CudaIndex, cudaGetErrorString(err));
        gTotalErrors++;
        return;
    }

    if (!Start())
    {
        printf("GPU %d, start failed\n", CudaIndex);
        gTotalErrors++;
        return;
    }
    
#ifdef DEBUG_MODE
    u64 iter = 1;
#endif
    
    while (!StopFlag)
    {
        u64 t1 = GetTickCount64();
        
        // Use CUDA streams with error checking
        err = cudaMemsetAsync(Kparams.DPs_out, 0, 4, memoryStream);
        if (err != cudaSuccess) {
            printf("GPU %d, cudaMemsetAsync DPs_out failed: %s\n", CudaIndex, cudaGetErrorString(err));
            gTotalErrors++;
            break;
        }
        
        err = cudaMemsetAsync(Kparams.DPTable, 0, KangCnt * sizeof(u32), memoryStream);
        if (err != cudaSuccess) {
            printf("GPU %d, cudaMemsetAsync DPTable failed: %s\n", CudaIndex, cudaGetErrorString(err));
            gTotalErrors++;
            break;
        }
        
        err = cudaMemsetAsync(Kparams.LoopedKangs, 0, 8, memoryStream);
        if (err != cudaSuccess) {
            printf("GPU %d, cudaMemsetAsync LoopedKangs failed: %s\n", CudaIndex, cudaGetErrorString(err));
            gTotalErrors++;
            break;
        }
        
        err = cudaStreamSynchronize(memoryStream);
        if (err != cudaSuccess) {
            printf("GPU %d, cudaStreamSynchronize memory stream failed: %s\n", CudaIndex, cudaGetErrorString(err));
            gTotalErrors++;
            break;
        }
        
        // Call main kernel with error checking
        CallGpuKernelABC(Kparams, computeStream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("GPU %d, CallGpuKernelABC launch failed: %s\n", CudaIndex, cudaGetErrorString(err));
            gTotalErrors++;
            break;
        }
        
        // Wait for computation to finish
        err = cudaStreamSynchronize(computeStream);
        if (err != cudaSuccess) {
            printf("GPU %d, cudaStreamSynchronize compute stream failed: %s\n", CudaIndex, cudaGetErrorString(err));
            gTotalErrors++;
            break;
        }
        
        // Copy results with error checking
        int cnt;
        err = cudaMemcpyAsync(&cnt, Kparams.DPs_out, 4, cudaMemcpyDeviceToHost, memoryStream);
        if (err != cudaSuccess) {
            printf("GPU %d, cudaMemcpyAsync cnt failed: %s\n", CudaIndex, cudaGetErrorString(err));
            gTotalErrors++;
            break;
        }
        
        err = cudaStreamSynchronize(memoryStream);
        if (err != cudaSuccess) {
            printf("GPU %d, cudaStreamSynchronize memory stream failed: %s\n", CudaIndex, cudaGetErrorString(err));
            gTotalErrors++;
            break;
        }
        
        if (cnt >= MAX_DP_CNT)
        {
            cnt = MAX_DP_CNT;
            printf("GPU %d, gpu DP buffer overflow, some points lost, increase DP value!\n", CudaIndex);
        }
        u64 pnt_cnt = (u64)KangCnt * STEP_CNT;

        if (cnt)
        {
            err = cudaMemcpyAsync(DPs_out, Kparams.DPs_out + 4, cnt * GPU_DP_SIZE, cudaMemcpyDeviceToHost, memoryStream);
            if (err != cudaSuccess) {
                printf("GPU %d, cudaMemcpyAsync DPs_out failed: %s\n", CudaIndex, cudaGetErrorString(err));
                gTotalErrors++;
                break;
            }
            
            err = cudaStreamSynchronize(memoryStream);
            if (err != cudaSuccess) {
                printf("GPU %d, cudaStreamSynchronize memory stream failed: %s\n", CudaIndex, cudaGetErrorString(err));
                gTotalErrors++;
                break;
            }
            
            // Add points to result list
            AddPointsToList(DPs_out, cnt, (u64)KangCnt * STEP_CNT);
        }

        // Debug info
        err = cudaMemcpyAsync(dbg, Kparams.dbg_buf, 1024, cudaMemcpyDeviceToHost, memoryStream);
        if (err != cudaSuccess) {
            printf("GPU %d, cudaMemcpyAsync dbg failed: %s\n", CudaIndex, cudaGetErrorString(err));
            // Not critical, continue
        }

        u32 lcnt;
        err = cudaMemcpyAsync(&lcnt, Kparams.LoopedKangs, 4, cudaMemcpyDeviceToHost, memoryStream);
        if (err != cudaSuccess) {
            printf("GPU %d, cudaMemcpyAsync lcnt failed: %s\n", CudaIndex, cudaGetErrorString(err));
            // Not critical, continue
        }
        
        err = cudaStreamSynchronize(memoryStream);
        if (err != cudaSuccess) {
            printf("GPU %d, cudaStreamSynchronize memory stream failed: %s\n", CudaIndex, cudaGetErrorString(err));
            // Not critical, continue
        }
        
        // Calculate and display speed statistics
        u64 t2 = GetTickCount64();
        u64 tm = t2 - t1;
        if (!tm)
            tm = 1;
        int cur_speed = (int)(pnt_cnt / (tm * 1000));
        
        SpeedStats[cur_stats_ind] = cur_speed;
        cur_stats_ind = (cur_stats_ind + 1) % STATS_WND_SIZE;

#ifdef DEBUG_MODE
        if ((iter % 300) == 0)
        {
            int corr_cnt = Dbg_CheckKangs();
            if (corr_cnt)
            {
                printf("DBG: GPU %d, KANGS CORRUPTED: %d\n", CudaIndex, corr_cnt);
                gTotalErrors++;
            }
            else
                printf("DBG: GPU %d, ALL KANGS OK!\n", CudaIndex);
        }
        iter++;
#endif
    }

    Release();
}

int RCGpuKang::GetStatsSpeed()
{
    int res = SpeedStats[0];
    for (int i = 1; i < STATS_WND_SIZE; i++)
        res += SpeedStats[i];
    return res / STATS_WND_SIZE;
}