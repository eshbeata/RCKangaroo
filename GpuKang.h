// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC

#pragma once

#include "defs.h"
#include "Ec.h"

#pragma pack(push, 1)
struct TPointPriv
{
	u64 x[8];
	u64 y[8];
	u64 priv[3];
};
#pragma pack(pop)

#define STATS_WND_SIZE 16

class RCGpuKang
{
public:
	Ec ec;
	RCGpuKang()
	{
		DPs_out = NULL;
		RndPnts = NULL;
	}
	bool Prepare(EcPoint PntToSolve, int Range, int DP, EcJMP* EcJumps1, EcJMP* EcJumps2, EcJMP* EcJumps3);
	void Execute();
	void Stop();
	void Release();
	bool Start();
	int GetStatsSpeed();

public:
	int CudaIndex;
	bool IsOldGpu;
	int mpCnt;	
	size_t persistingL2CacheMaxSize;

	// Memory optimization for RTX 5090
	// Use these flags to control memory optimization features
	bool useMemoryPools = true;  // Enable CUDA memory pools for faster allocations
	bool useManagedMemory = true; // Use unified memory for some buffers
	bool useStreamOrderMemOps = true; // Enable stream ordered memory operations
	
	TKparams Kparams;
	u32* DPs_out;
	TPointPriv* RndPnts;
	int KangCnt;
	EcPoint PntToSolve;
	EcPoint PntA;
	EcPoint PntB;
	EcInt HalfRange;
	EcPoint PntHalfRange;
	EcPoint NegPntHalfRange;
	EcPoint PntTame;
	int Range;
	int DP;
	EcJMP* EcJumps1;
	EcJMP* EcJumps2;
	EcJMP* EcJumps3;
	
	u32 dbg[MD_LEN + 2];

	int SpeedStats[STATS_WND_SIZE];
	int cur_stats_ind;

	bool Failed;
	bool StopFlag;

private:
	int CalcKangCnt();
	void GenerateRndDistances();
#ifdef DEBUG_MODE
	int Dbg_CheckKangs();
#endif
};
