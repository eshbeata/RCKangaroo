#!/bin/bash

# Simple RTX 5090 compatibility script

# Set colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}= RCKangaroo RTX 5090 Direct Fix =${NC}"
echo -e "${GREEN}========================================${NC}"
echo

# Create a backup of GpuKang.cpp
echo -e "${YELLOW}Creating backup of GpuKang.cpp...${NC}"
cp GpuKang.cpp GpuKang.cpp.backup

# Directly modify GpuKang.cpp to limit resources
echo -e "${BLUE}Modifying GpuKang.cpp to add RTX 5090 compatibility...${NC}"
cat > GpuKang.cpp << 'EOF'
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
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, CudaIndex);
    
    // RTX 5090 SPECIFIC FIX: For RTX 5090, limit block count drastically to prevent segfault
    if (deviceProp.major >= 9) {
        printf("RTX 5090 COMPATIBILITY MODE: Using ultra-conservative settings\n");
        
        // Ultra-conservative settings for RTX 5090 to prevent segmentation faults
        Kparams.BlockCnt = 16;  // Drastically reduced block count
        Kparams.BlockSize = 128; // Smaller block size
        Kparams.GroupCnt = 8;    // Fewer groups
        
        printf("RTX 5090 COMPATIBILITY MODE: Using very limited parameters: BlockCnt=%d, BlockSize=%d, GroupCnt=%d\n",
               Kparams.BlockCnt, Kparams.BlockSize, Kparams.GroupCnt);
    } else if (IsOldGpu) {
        // Parameters for older GPUs remain unchanged
        Kparams.BlockCnt = mpCnt;
        Kparams.BlockSize = 512;
        Kparams.GroupCnt = 64;
    } else {
        // For RTX 3000/4000 series
        Kparams.BlockCnt = mpCnt;
        Kparams.BlockSize = 256;
        Kparams.GroupCnt = 32;
    }
    
    // Calculate total kangaroo count with hard ceiling for RTX 5090
    int totalKangCnt = Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
    const int MAX_SAFE_KANG_CNT = 40000; // Hard limit to prevent segfault on RTX 5090
    
    if (deviceProp.major >= 9 && totalKangCnt > MAX_SAFE_KANG_CNT) {
        printf("RTX 5090 COMPATIBILITY MODE: Limiting total kangaroo count from %d to %d\n", 
               totalKangCnt, MAX_SAFE_KANG_CNT);
        
        totalKangCnt = MAX_SAFE_KANG_CNT;
        
        // Recalculate BlockCnt to fit within limit
        Kparams.BlockCnt = MAX_SAFE_KANG_CNT / (Kparams.BlockSize * Kparams.GroupCnt);
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

    // Get device properties to check for RTX 5090
    cudaDeviceProp deviceProp;
    err = cudaGetDeviceProperties(&deviceProp, CudaIndex);
    if (err != cudaSuccess) {
        printf("GPU %d, failed to get device properties: %s\n", CudaIndex, cudaGetErrorString(err));
        cudaStreamDestroy(computeStream);
        cudaStreamDestroy(memoryStream);
        return false;
    }
    
    // For RTX 5090, use extremely conservative settings to prevent segfaults
    if (deviceProp.major >= 9) {
        printf("RTX 5090 COMPATIBILITY MODE: Using ultra-conservative settings\n");
        
        // Extremely reduced values for RTX 5090
        KangCnt = CalcKangCnt(); // This will call our modified CalcKangCnt() with limits
        
        // Calculate extremely conservative shared memory sizes for RTX 5090
        size_t maxSharedMemPerBlock = deviceProp.sharedMemPerBlock;
        printf("Device max shared memory per block: %zu bytes\n", maxSharedMemPerBlock);
        
        // Use very conservative shared memory settings (25% of max)
        Kparams.KernelA_LDS_Size = (unsigned int)(maxSharedMemPerBlock * 0.25);
        Kparams.KernelB_LDS_Size = (unsigned int)(maxSharedMemPerBlock * 0.25);
        Kparams.KernelC_LDS_Size = (unsigned int)(maxSharedMemPerBlock * 0.25);
    } else {
        // Call original calculation for non-RTX 5090 GPUs
        KangCnt = CalcKangCnt();
        
        // Set shared memory sizes normally
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, CudaIndex);
        size_t maxSharedMemPerBlock = deviceProp.sharedMemPerBlock;
        Kparams.KernelA_LDS_Size = (unsigned int)(maxSharedMemPerBlock * 0.6);
        Kparams.KernelB_LDS_Size = (unsigned int)(maxSharedMemPerBlock * 0.5);
        Kparams.KernelC_LDS_Size = (unsigned int)(maxSharedMemPerBlock * 0.5);
    }
    
    // Set kernel parameters
    Kparams.KangCnt = KangCnt;
    Kparams.DP = DP;
    Kparams.IsGenMode = gGenMode;

    // For RTX 5090, limit step count and other parameters
    int effectiveMaxDpCnt = MAX_DP_CNT;
    int effectiveStepCnt = STEP_CNT;
    int effectiveDpTableMaxCnt = DPTABLE_MAX_CNT;
    int effectiveMdLen = MD_LEN;
    
    if (deviceProp.major >= 9) {
        // Ultra-reduced parameters for RTX 5090
        effectiveMaxDpCnt = 8192;      // Drastically reduced from MAX_DP_CNT
        effectiveStepCnt = 64;         // Drastically reduced from STEP_CNT
        effectiveDpTableMaxCnt = 4;    // Drastically reduced from DPTABLE_MAX_CNT
        effectiveMdLen = 8;            // Reduced from MD_LEN
        
        printf("RTX 5090 COMPATIBILITY MODE: Using reduced parameters:\n");
        printf("  - MaxDpCnt: %d\n", effectiveMaxDpCnt);
        printf("  - StepCnt: %d\n", effectiveStepCnt);
        printf("  - DpTableMaxCnt: %d\n", effectiveDpTableMaxCnt);
        printf("  - MdLen: %d\n", effectiveMdLen);
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
    
    // Allocate memory with safe sizes
    size = effectiveMaxDpCnt * GPU_DP_SIZE + 16;
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.DPs_out, size);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate GpuOut memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

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

    // Allocate JumpsList with the effective STEP_CNT
    size = 2 * (u64)KangCnt * effectiveStepCnt;
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.JumpsList, size);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate JumpsList memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    size = (u64)KangCnt * (16 * effectiveDpTableMaxCnt + sizeof(u32)); 
    if (deviceProp.major >= 9 && size > 1ULL * 1024 * 1024 * 1024) {
        printf("RTX 5090 COMPATIBILITY MODE: DP table size too large, reducing to 1GB\n");
        size = 1ULL * 1024 * 1024 * 1024; 
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
    
    // Add RTX 5090 specific error checking
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, CudaIndex);
    
    // Launch with error checking
    cudaError_t kernelErr;
    if (deviceProp.major >= 9) {
        printf("RTX 5090 COMPATIBILITY MODE: Launching kernel with extra safety checks\n");
        
        // For RTX 5090 we synchronize device before kernel launch
        cudaDeviceSynchronize();
    }
    
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

    // Get device properties to check for RTX 5090
    cudaDeviceProp deviceProp;
    err = cudaGetDeviceProperties(&deviceProp, CudaIndex);
    
    // For RTX 5090, use special slow and safe execution
    bool isRTX5090 = (deviceProp.major >= 9);
    
    if (isRTX5090) {
        printf("RTX 5090 COMPATIBILITY MODE: Using ultra-safe execution path\n");
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
        
        // For RTX 5090, synchronize between operations
        if (isRTX5090) {
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("RTX 5090 COMPATIBILITY MODE: Device sync failed: %s\n", cudaGetErrorString(err));
                gTotalErrors++;
                break;
            }
        } else {
            err = cudaStreamSynchronize(memoryStream);
            if (err != cudaSuccess) {
                printf("GPU %d, cudaStreamSynchronize memory stream failed: %s\n", CudaIndex, cudaGetErrorString(err));
                gTotalErrors++;
                break;
            }
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
        
        if (isRTX5090) {
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("RTX 5090 COMPATIBILITY MODE: Device sync failed: %s\n", cudaGetErrorString(err));
                gTotalErrors++;
                break;
            }
        } else {
            err = cudaStreamSynchronize(memoryStream);
            if (err != cudaSuccess) {
                printf("GPU %d, cudaStreamSynchronize memory stream failed: %s\n", CudaIndex, cudaGetErrorString(err));
                gTotalErrors++;
                break;
            }
        }
        
        // For RTX 5090, limit the maximum points to an even smaller number
        if (isRTX5090 && cnt > 1024) {
            printf("RTX 5090 COMPATIBILITY MODE: Limiting points from %d to 1024\n", cnt);
            cnt = 1024;
        } else if (cnt >= MAX_DP_CNT) {
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
            
            if (isRTX5090) {
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess) {
                    printf("RTX 5090 COMPATIBILITY MODE: Device sync failed: %s\n", cudaGetErrorString(err));
                    gTotalErrors++;
                    break;
                }
            } else {
                err = cudaStreamSynchronize(memoryStream);
                if (err != cudaSuccess) {
                    printf("GPU %d, cudaStreamSynchronize memory stream failed: %s\n", CudaIndex, cudaGetErrorString(err));
                    gTotalErrors++;
                    break;
                }
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
        
        if (isRTX5090) {
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("RTX 5090 COMPATIBILITY MODE: Device sync failed: %s\n", cudaGetErrorString(err));
                // Not critical, continue
            }
        } else {
            err = cudaStreamSynchronize(memoryStream);
            if (err != cudaSuccess) {
                printf("GPU %d, cudaStreamSynchronize memory stream failed: %s\n", CudaIndex, cudaGetErrorString(err));
                // Not critical, continue
            }
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

        // For RTX 5090, add a small delay between iterations to prevent resource exhaustion
        if (isRTX5090) {
            #ifdef _WIN32
            Sleep(50);  // 50 ms delay on Windows
            #else
            usleep(50000); // 50 ms delay on Linux/Unix
            #endif
        }
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
EOF

# Create a direct build script that will compile with our modified file
echo -e "${BLUE}Creating direct build script...${NC}"
cat > build_direct.sh << 'EOF'
#!/bin/bash

# Set colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}= RCKangaroo RTX 5090 Direct Build =${NC}"
echo -e "${GREEN}========================================${NC}"
echo

# Clean previous build
echo -e "${YELLOW}Cleaning previous build...${NC}"
make clean

# Build with direct modifications
echo -e "${BLUE}Building with direct RTX 5090 fixes...${NC}"
echo -e "${YELLOW}Compiling CPU code...${NC}"

# Add usleep function for Linux systems (if needed)
cat > usleep.h << 'EOH'
#ifndef USLEEP_H
#define USLEEP_H

#ifdef _WIN32
// Windows Sleep is in milliseconds
#include <windows.h>
#else
// Unix usleep is in microseconds
#include <unistd.h>
#endif

#endif
EOH

g++ -O3 -march=native -I/usr/local/cuda-12.1/include -c RCKangaroo.cpp -o RCKangaroo.o
g++ -O3 -march=native -I/usr/local/cuda-12.1/include -c GpuKang.cpp -o GpuKang.o
g++ -O3 -march=native -I/usr/local/cuda-12.1/include -c Ec.cpp -o Ec.o
g++ -O3 -march=native -I/usr/local/cuda-12.1/include -c utils.cpp -o utils.o

echo -e "${YELLOW}Compiling CUDA code...${NC}"
nvcc -O3 -gencode=arch=compute_90,code=sm_90 --use_fast_math --threads 0 --gpu-architecture=sm_90 -Xptxas=-v,-O3 -Xcompiler=-O3,-march=native --default-stream=per-thread --maxrregcount=32 -c RCGpuCore.cu -o RCGpuCore.o

if [ $? -ne 0 ]; then
    echo -e "${RED}CUDA compilation failed${NC}"
    exit 1
fi

echo -e "${YELLOW}Linking final executable...${NC}"
g++ -O3 -march=native -I/usr/local/cuda-12.1/include -o rckangaroo RCKangaroo.o GpuKang.o Ec.o utils.o RCGpuCore.o -L/usr/local/cuda-12.1/lib64 -lcudart -pthread

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed${NC}"
    exit 1
fi

echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${BLUE}Executable: ./rckangaroo${NC}"
echo
echo -e "${YELLOW}Recommended command line:${NC}"
echo -e "./rckangaroo -dp 16 -range 84 -start 1000000000000000000000 -pubkey 0329c4574a4fd8c810b7e42a4b398882b381bcd85e40c6883712912d167c83e73a"
echo

chmod +x rckangaroo

exit 0
EOF

echo -e "${GREEN}Direct fix created successfully!${NC}"
echo -e "${YELLOW}To build with direct fixes, run:${NC}"
echo -e "chmod +x build_direct.sh"
echo -e "./build_direct.sh"
echo
echo -e "${BLUE}This approach directly modifies GpuKang.cpp with RTX 5090 specific optimizations.${NC}"
echo -e "${BLUE}The file has been specialized to run safely on the RTX 5090 with minimal resource usage.${NC}"

chmod +x build_direct.sh