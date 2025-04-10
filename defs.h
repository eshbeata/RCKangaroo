// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC

#pragma once 

#pragma warning(disable : 4996)

typedef unsigned long long u64;
typedef long long i64;
typedef unsigned int u32;
typedef int i32;
typedef unsigned short u16;
typedef short i16;
typedef unsigned char u8;
typedef char i8;

// Increased maximum GPU count for multi-GPU setups
#define MAX_GPU_CNT			64

//use different options for cards older than RTX 50xx
#ifdef __CUDA_ARCH__
	#if __CUDA_ARCH__ < 900
		#define OLD_GPU
	#endif
	#ifdef OLD_GPU
		#define BLOCK_SIZE			512
		//can be 8, 16, 24, 32, 40, 48, 56, 64
		#define PNT_GROUP_CNT		64	
	#else
		// Special handling for RTX 5090
		#ifdef RTX_5090_GPU
			// Optimized settings for RTX 5090 to prevent segmentation faults
			#define BLOCK_SIZE                       256
			#define PNT_GROUP_CNT                    12    // Reduced from 48 for RTX 5090
			#define STEP_CNT                         96    // Reduced from 1500 for RTX 5090
			#define MD_LEN                          12     // Reduced from 16 for RTX 5090
			#define JMP_CNT                         512
			#define DPTABLE_MAX_CNT                  8     // Significantly reduced from 32 for RTX 5090
			#define MAX_CNT_LIST                    65536  // Significantly reduced for RTX 5090
			#define MAX_DP_CNT                      16384  // Significantly reduced for RTX 5090
		#else
			// Optimized block size for RTX 5090
			#define BLOCK_SIZE			256
			// Increased group count for better SM utilization on RTX 5090
			#define PNT_GROUP_CNT		48
			#define STEP_CNT			1500
			#define MD_LEN				16
			#define JMP_CNT				1024
			#define DPTABLE_MAX_CNT		32
			#define MAX_CNT_LIST		(2048 * 1024)
			#define MAX_DP_CNT			(1024 * 1024)
		#endif
	#endif
#else //CPU, fake values
	#define BLOCK_SIZE			512
	#define PNT_GROUP_CNT		64
	#define STEP_CNT			1500
	#define MD_LEN				16
	#define JMP_CNT				1024
	#define DPTABLE_MAX_CNT		32
	#define MAX_CNT_LIST		(2048 * 1024)
	#define MAX_DP_CNT			(1024 * 1024)
#endif

#define JMP_MASK			(JMP_CNT-1)

// kang type
#define TAME				0  // Tame kangs
#define WILD1				1  // Wild kangs1 
#define WILD2				2  // Wild kangs2

#define GPU_DP_SIZE			48

#define DP_FLAG				0x8000
#define INV_FLAG			0x4000
#define JMP2_FLAG			0x2000

//#define DEBUG_MODE

//gpu kernel parameters
struct TKparams
{
	u64* Kangs;
	u32 KangCnt;
	u32 BlockCnt;
	u32 BlockSize;
	u32 GroupCnt;
	u64* L2;
	u64 DP;
	u32* DPs_out;
	u64* Jumps1; //x(32b), y(32b), d(32b)
	u64* Jumps2; //x(32b), y(32b), d(32b)
	u64* Jumps3; //x(32b), y(32b), d(32b)
	u64* JumpsList; //list of all performed jumps, grouped by warp(32) every 8 groups (from PNT_GROUP_CNT). Each jump is 2 bytes: 10bit jump index + flags: INV_FLAG, DP_FLAG, JMP2_FLAG
	u32* DPTable;
	u32* L1S2;
	u64* LastPnts;
	u64* LoopTable;
	u32* dbg_buf;
	u32* LoopedKangs;
	bool IsGenMode; //tames generation mode

	u32 KernelA_LDS_Size;
	u32 KernelB_LDS_Size;
	u32 KernelC_LDS_Size;	
};

