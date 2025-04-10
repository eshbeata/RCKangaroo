#ifndef RCGPUCORE_H
#define RCGPUCORE_H

#include "defs.h"

// Declare CUDA kernels
extern "C" __global__ void KernelA(const TKparams Kparams);
extern "C" __global__ void KernelB(const TKparams Kparams);
extern "C" __global__ void KernelC(const TKparams Kparams);
extern "C" __global__ void KernelGen(const TKparams Kparams);

// Declare GPU kernel-related functions
void CallGpuKernelABC(TKparams Kparams);
void CallGpuKernelGen(TKparams Kparams);
cudaError_t cuSetGpuParams(TKparams Kparams, u64* _jmp2_table);

#endif // RCGPUCORE_H