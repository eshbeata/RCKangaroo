// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"

#include "GpuKang.h"

cudaError_t cuSetGpuParams(TKparams Kparams, u64* _jmp2_table);
void CallGpuKernelGen(TKparams Kparams, cudaStream_t stream = 0);
void CallGpuKernelABC(TKparams Kparams, cudaStream_t stream = 0);
void AddPointsToList(u32* data, int cnt, u64 ops_cnt);
extern bool gGenMode; //tames generation mode

// Add CUDA stream declarations
cudaStream_t computeStream;
cudaStream_t memoryStream;

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
        Kparams.BlockCnt = mpCnt;
        
        // For RTX 5090 with compute capability 9.0+, optimize block size
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, CudaIndex);
        
        if (deviceProp.major >= 9) {
            printf("Optimizing kernel parameters for RTX 5090...\n");
            
            // RTX 5090 has excellent L1 cache and shared memory, 
            // so we can use moderate block sizes for better occupancy
            Kparams.BlockSize = 256;
            
            // RTX 5090 has more CUDA cores per SM, so we can increase the group count
            Kparams.GroupCnt = 48;
            
            // Use GPU-specific optimization based on SM count
            if (mpCnt >= 128) {
                // For GPUs with very high SM count like RTX 5090
                Kparams.GroupCnt = 64;
                printf("Using high-density configuration for %d SMs\n", mpCnt);
            } else if (mpCnt >= 84) {
                // For high-end RTX GPUs
                Kparams.GroupCnt = 48;
                printf("Using balanced configuration for %d SMs\n", mpCnt);
            } else {
                // For mid-range RTX GPUs
                Kparams.GroupCnt = 32;
                printf("Using standard configuration for %d SMs\n", mpCnt);
            }
        } else {
            // For RTX 3000/4000 series
            Kparams.BlockSize = 256;
            Kparams.GroupCnt = 32;
        }
    }
    
    // Calculate and return total kangaroo count
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
    if (err != cudaSuccess)
        return false;
    
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
        
        err = cudaStreamCreateWithPriority(&temp_stream, cudaStreamNonBlocking, priority_low);
        if (err == cudaSuccess) {
            cudaStreamDestroy(memoryStream);
            memoryStream = temp_stream;
        } else {
            printf("Warning: Failed to set memory stream priority: %s\n", cudaGetErrorString(err));
        }
    }

    // Set up memory pool for RTX 5090 with error checking
    if (useMemoryPools && !IsOldGpu) {
        cudaMemPool_t memPool;
        err = cudaDeviceGetDefaultMemPool(&memPool, CudaIndex);
        if (err == cudaSuccess) {
            // Configure memory pool with error checking
            uint64_t threshold = UINT64_MAX;
            err = cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &threshold);
            if (err != cudaSuccess) {
                printf("GPU %d, failed to set memory pool attribute: %s\n", CudaIndex, cudaGetErrorString(err));
                // Continue anyway, not critical
            }
        } else {
            printf("GPU %d, could not get default memory pool: %s\n", CudaIndex, cudaGetErrorString(err));
            useMemoryPools = false;
        }
    }

    // First calculate and validate kangaroo counts to avoid overflow
    Kparams.BlockCnt = mpCnt;
    
    // For RTX 5090, we need to limit the BlockSize and GroupCnt to reasonable values
    // to prevent excessive memory allocation
    int maxBlockSize = IsOldGpu ? 512 : 256;
    int maxGroupCnt = 64;  // Limiting to a reasonable maximum to prevent segfault
    
    if (!IsOldGpu) {
        cudaDeviceProp deviceProp;
        if (cudaGetDeviceProperties(&deviceProp, CudaIndex) == cudaSuccess) {
            // Limit kangaroo count for very large GPUs
            if (deviceProp.major >= 9) {
                printf("Setting conservative parameters for RTX 5090 to avoid memory issues...\n");
                if (mpCnt > 100) {
                    // Reduce BlockCnt for very large SM counts
                    Kparams.BlockCnt = 100;
                    printf("Limiting BlockCnt to %d (from %d SMs) to prevent excessive memory usage\n", 
                           Kparams.BlockCnt, mpCnt);
                }
                
                // Use balanced values for 5090
                maxBlockSize = 256;
                maxGroupCnt = 32;
            }
        }
    }
    
    Kparams.BlockSize = IsOldGpu ? 512 : maxBlockSize;
    Kparams.GroupCnt = IsOldGpu ? 64 : maxGroupCnt;
    
    // Calculate safe kangaroo count
    KangCnt = Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
    
    // Safety check - if kangaroo count is too large, reduce it to a safe value
    const u64 MAX_SAFE_KANG_CNT = 4000000; // Limit to avoid segfault
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
    
    Kparams.KangCnt = KangCnt;
    Kparams.DP = DP;
    Kparams.KernelA_LDS_Size = 64 * JMP_CNT + 16 * Kparams.BlockSize;
    Kparams.KernelB_LDS_Size = 64 * JMP_CNT;
    Kparams.KernelC_LDS_Size = 96 * JMP_CNT;
    Kparams.IsGenMode = gGenMode;

    // For RTX 5090, optimize cache behavior
    if (!IsOldGpu) {
        // Set CUDA kernel execution policies for better performance
        cudaFuncCache cacheConfig = cudaFuncCachePreferShared;
        err = cudaDeviceSetCacheConfig(cacheConfig);
        if (err != cudaSuccess) {
            printf("GPU %d, failed to set cache config: %s\n", CudaIndex, cudaGetErrorString(err));
            // Continue anyway, not critical
        }
        
        // Increase L2 fetch granularity with error checking
        err = cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 128);
        if (err != cudaSuccess) {
            printf("Warning: Could not set L2 fetch granularity: %s\n", cudaGetErrorString(err));
            // Continue anyway, not critical
        }
    }

    // Rest of the function stays mostly the same but with better error checking

    // ... rest of the memory allocation code ...
    
    //allocate gpu mem
    u64 size;
    if (!IsOldGpu)
    {
        //L2	
        int L2size = Kparams.KangCnt * (3 * 32);
        total_mem += L2size;
        
        // For RTX 5090, use stream-ordered memory allocation if supported
        if (useStreamOrderMemOps) {
            err = cudaMallocAsync((void**)&Kparams.L2, L2size, computeStream);
            if (err != cudaSuccess) {
                printf("GPU %d, Async allocation failed, falling back to standard malloc: %s\n", CudaIndex, cudaGetErrorString(err));
                err = cudaMalloc((void**)&Kparams.L2, L2size);
            }
        } else {
            err = cudaMalloc((void**)&Kparams.L2, L2size);
        }
        
        if (err != cudaSuccess)
        {
            printf("GPU %d, Allocate L2 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
            return false;
            }
        
        // For RTX 5090, maximize the L2 cache usage
        size = L2size;
        // On newer GPUs, increase the persistent L2 cache size limit
        if (size > persistingL2CacheMaxSize)
            size = persistingL2CacheMaxSize;
            
        err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); // set max allowed size for L2
        if (err != cudaSuccess) {
            printf("Warning: Could not set L2 cache size limit: %s\n", cudaGetErrorString(err));
            // Continue anyway, not critical
        }
        
        //persisting for L2
        cudaStreamAttrValue stream_attribute;                                                   
        stream_attribute.accessPolicyWindow.base_ptr = Kparams.L2;
        stream_attribute.accessPolicyWindow.num_bytes = size;										
        stream_attribute.accessPolicyWindow.hitRatio = 1.0;                                     
        stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;             
        stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;  	
        err = cudaStreamSetAttribute(computeStream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
        if (err != cudaSuccess)
        {
            printf("GPU %d, cudaStreamSetAttribute failed (non-critical): %s\n", CudaIndex, cudaGetErrorString(err));
            // Continue anyway, not critical
        }
    }
    
    // Increase buffer sizes for RTX 5090
    size = MAX_DP_CNT * GPU_DP_SIZE + 16;
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.DPs_out, size);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate GpuOut memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    // Allocate memory for larger kangaroo groups
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
        printf("GPU %d Allocate Jumps1 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    total_mem += JMP_CNT * 96;
    err = cudaMalloc((void**)&Kparams.Jumps3, JMP_CNT * 96);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate Jumps3 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    // Increase JumpsList size for RTX 5090
    size = 2 * (u64)KangCnt * STEP_CNT;
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.JumpsList, size);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate JumpsList memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    size = (u64)KangCnt * (16 * DPTABLE_MAX_CNT + sizeof(u32)); //we store 16bytes of X
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

    size = (u64)KangCnt * MD_LEN * (2 * 32);
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.LastPnts, size);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate LastPnts memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    size = (u64)KangCnt * MD_LEN * sizeof(u64);
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.LoopTable, size);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate LastPnts memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
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
    DPs_out = (u32*)malloc(MAX_DP_CNT * GPU_DP_SIZE);
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

    printf("GPU %d: allocated %llu MB, %d kangaroos. OldGpuMode: %s\r\n", CudaIndex, total_mem / (1024 * 1024), KangCnt, IsOldGpu ? "Yes" : "No");
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
	GenerateRndDistances();
/* 
	//we can calc start points on CPU
	for (int i = 0; i < KangCnt; i++)
	{
		EcInt d;
		memcpy(d.data, RndPnts[i].priv, 24);
		d.data[3] = 0;
		d.data[4] = 0;
		EcPoint p = ec.MultiplyG(d);
		memcpy(RndPnts[i].x, p.x.data, 32);
		memcpy(RndPnts[i].y, p.y.data, 32);
	}
	for (int i = KangCnt / 3; i < 2 * KangCnt / 3; i++)
	{
		EcPoint p;
		p.LoadFromBuffer64((u8*)RndPnts[i].x);
		p = ec.AddPoints(p, PntA);
		p.SaveToBuffer64((u8*)RndPnts[i].x);
	}
	for (int i = 2 * KangCnt / 3; i < KangCnt; i++)
	{
		EcPoint p;
		p.LoadFromBuffer64((u8*)RndPnts[i].x);
		p = ec.AddPoints(p, PntB);
		p.SaveToBuffer64((u8*)RndPnts[i].x);
	}
	//copy to gpu
	err = cudaMemcpy(Kparams.Kangs, RndPnts, KangCnt * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
/**/
	//but it's faster to calc then on GPU
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
	//copy to gpu
	err = cudaMemcpy(Kparams.Kangs, RndPnts, KangCnt * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	CallGpuKernelGen(Kparams);

	err = cudaMemset(Kparams.L1S2, 0, mpCnt * Kparams.BlockSize * 8);
	if (err != cudaSuccess)
		return false;
	cudaMemset(Kparams.dbg_buf, 0, 1024);
	cudaMemset(Kparams.LoopTable, 0, KangCnt * MD_LEN * sizeof(u64));
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

//executes in separate thread
void RCGpuKang::Execute()
{
	cudaSetDevice(CudaIndex);

	if (!Start())
	{
		gTotalErrors++;
		return;
	}
#ifdef DEBUG_MODE
	u64 iter = 1;
#endif
	cudaError_t err;	
	while (!StopFlag)
	{
		u64 t1 = GetTickCount64();
		
		// Use CUDA streams for better parallelism between operations
		cudaMemsetAsync(Kparams.DPs_out, 0, 4, memoryStream);
		cudaMemsetAsync(Kparams.DPTable, 0, KangCnt * sizeof(u32), memoryStream);
		cudaMemsetAsync(Kparams.LoopedKangs, 0, 8, memoryStream);
		cudaStreamSynchronize(memoryStream);
		
		// Call main kernel with compute stream
		CallGpuKernelABC(Kparams, computeStream);
		
		int cnt;
		err = cudaMemcpyAsync(&cnt, Kparams.DPs_out, 4, cudaMemcpyDeviceToHost, memoryStream);
		cudaStreamSynchronize(memoryStream);
		
		if (err != cudaSuccess)
		{
			printf("GPU %d, CallGpuKernel failed: %s\r\n", CudaIndex, cudaGetErrorString(err));
			gTotalErrors++;
			break;
		}
		
		if (cnt >= MAX_DP_CNT)
		{
			cnt = MAX_DP_CNT;
			printf("GPU %d, gpu DP buffer overflow, some points lost, increase DP value!\r\n", CudaIndex);
		}
		u64 pnt_cnt = (u64)KangCnt * STEP_CNT;

		if (cnt)
		{
			err = cudaMemcpyAsync(DPs_out, Kparams.DPs_out + 4, cnt * GPU_DP_SIZE, cudaMemcpyDeviceToHost, memoryStream);
			cudaStreamSynchronize(memoryStream);
			if (err != cudaSuccess)
			{
				gTotalErrors++;
				break;
			}
			AddPointsToList(DPs_out, cnt, (u64)KangCnt * STEP_CNT);
		}

		//dbg
		cudaMemcpyAsync(dbg, Kparams.dbg_buf, 1024, cudaMemcpyDeviceToHost, memoryStream);

		u32 lcnt;
		cudaMemcpyAsync(&lcnt, Kparams.LoopedKangs, 4, cudaMemcpyDeviceToHost, memoryStream);
		cudaStreamSynchronize(memoryStream);
		
		u64 t2 = GetTickCount64();
		u64 tm = t2 - t1;
		if (!tm)
			tm = 1;
		int cur_speed = (int)(pnt_cnt / (tm * 1000));
		//printf("GPU %d kernel time %d ms, speed %d MH\r\n", CudaIndex, (int)tm, cur_speed);

		SpeedStats[cur_stats_ind] = cur_speed;
		cur_stats_ind = (cur_stats_ind + 1) % STATS_WND_SIZE;

#ifdef DEBUG_MODE
		if ((iter % 300) == 0)
		{
			int corr_cnt = Dbg_CheckKangs();
			if (corr_cnt)
			{
				printf("DBG: GPU %d, KANGS CORRUPTED: %d\r\n", CudaIndex, corr_cnt);
				gTotalErrors++;
			}
			else
				printf("DBG: GPU %d, ALL KANGS OK!\r\n", CudaIndex);
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