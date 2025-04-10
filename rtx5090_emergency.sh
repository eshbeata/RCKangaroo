#!/bin/bash

# Emergency hotfix script for RTX 5090 compatibility

# Set colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}= RCKangaroo RTX 5090 EMERGENCY HOTFIX =${NC}"
echo -e "${GREEN}========================================${NC}"
echo

# Create simplified patch header that won't redefine macros
cat > rtx5090_patch.h << EOF
// RTX 5090 simplified patch - auto-generated
#pragma once

// Safety limits to prevent segmentation faults
#ifndef RTX5090_MODE
#define RTX5090_MODE
#endif

// Very conservative settings that won't conflict with existing code
#define RTX5090_BLOCK_CNT 16
#define RTX5090_KANGAROO_LIMIT 40000
#define RTX5090_SHARED_MEM_SIZE (16 * 1024)
#define RTX5090_STEP_LIMIT 32
#define RTX5090_USE_MINIMAL_MEMORY
EOF

# Create a minimal patch file
cat > cuda_patch.cu << EOF
// Extra safety for CUDA kernel calls
extern "C" void SafeCallGpuKernelABC(TKparams Kparams, cudaStream_t stream)
{
    // Get device properties first
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    
    // For RTX 5090, use more conservative parameters
    if (props.major >= 9) {
        printf("RTX 5090 detected: Using very conservative limits\n");
        
        // Limit block count to prevent excessive memory usage
        if (Kparams.BlockCnt > RTX5090_BLOCK_CNT) {
            printf("Limiting BlockCnt from %d to %d for RTX 5090 safety\n", 
                  Kparams.BlockCnt, RTX5090_BLOCK_CNT);
            Kparams.BlockCnt = RTX5090_BLOCK_CNT;
        }
        
        // Use the most conservative shared memory settings
        size_t maxSharedMem = props.sharedMemPerBlockOptin;
        size_t safeSharedMem = maxSharedMem > RTX5090_SHARED_MEM_SIZE ? 
                              RTX5090_SHARED_MEM_SIZE : maxSharedMem / 2;
                              
        // Set conservative shared memory sizes
        Kparams.KernelA_LDS_Size = (unsigned int)safeSharedMem;
        Kparams.KernelB_LDS_Size = (unsigned int)safeSharedMem;
        Kparams.KernelC_LDS_Size = (unsigned int)safeSharedMem;
    }
    
    // Call original function with adjusted parameters
    CallGpuKernelABC(Kparams, stream);
}

extern "C" void SafeCallGpuKernelGen(TKparams Kparams, cudaStream_t stream)
{
    // Get device properties first
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    
    // For RTX 5090, use more conservative parameters
    if (props.major >= 9) {
        // Limit block count for safety
        if (Kparams.BlockCnt > RTX5090_BLOCK_CNT) {
            Kparams.BlockCnt = RTX5090_BLOCK_CNT;
        }
    }
    
    // Call original function with adjusted parameters
    CallGpuKernelGen(Kparams, stream);
}
EOF

# Create a simplified build script that applies minimal changes
cat > build_simple.sh << EOF
#!/bin/bash

# Set colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "\${GREEN}========================================\${NC}"
echo -e "\${GREEN}= RCKangaroo RTX 5090 MINIMAL HOTFIX =\${NC}"
echo -e "\${GREEN}========================================\${NC}"
echo

# Clean previous build
echo -e "\${YELLOW}Cleaning previous build...\${NC}"
make clean

# Add minimal patch to GpuKang.cpp
cat > gpukang_patch.cpp << EOG
// Simple RTX 5090 compatibility patch for GpuKang
#include "rtx5090_patch.h"

// Override the calculate kangaroo count method with a safer version
int RCGpuKang::CalcKangCnt()
{
    #ifdef RTX5090_MODE
    // Set extremely conservative parameters for RTX 5090
    printf("RTX 5090 SAFETY MODE: Using conservative settings\n");
    
    // Retrieve device properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, CudaIndex);
    
    if (deviceProp.major >= 9) {
        // Ultra-conservative settings for RTX 5090
        Kparams.BlockCnt = RTX5090_BLOCK_CNT;
        Kparams.BlockSize = 128;
        Kparams.GroupCnt = 8;
    } else {
        // Standard settings for older GPUs
        Kparams.BlockCnt = mpCnt;
        Kparams.BlockSize = IsOldGpu ? 512 : 256;
        Kparams.GroupCnt = IsOldGpu ? 64 : 32;
    }
    
    // Calculate total with hard safety cap
    int totalKangCnt = Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
    if (totalKangCnt > RTX5090_KANGAROO_LIMIT) {
        totalKangCnt = RTX5090_KANGAROO_LIMIT;
        
        // Recalculate parameters to fit within limit
        Kparams.BlockCnt = RTX5090_KANGAROO_LIMIT / (Kparams.BlockSize * Kparams.GroupCnt);
        printf("SAFETY MODE: Limiting to %d kangaroos\n", totalKangCnt);
    }
    
    return totalKangCnt;
    #else
    // Original implementation
    if (IsOldGpu) {
        Kparams.BlockCnt = mpCnt;
        Kparams.BlockSize = 512;
        Kparams.GroupCnt = 64;
    } else {
        Kparams.BlockCnt = mpCnt;
        Kparams.BlockSize = 256;
        Kparams.GroupCnt = 32;
    }
    
    return Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
    #endif
}
EOG

# Create header file with safer function declarations
cat > wrapper.h << EOW
// Safety wrapper functions
#pragma once
#include "defs.h"

// Original function declarations
extern "C" void CallGpuKernelABC(TKparams Kparams, cudaStream_t stream);
extern "C" void CallGpuKernelGen(TKparams Kparams, cudaStream_t stream);

// Safe versions
extern "C" void SafeCallGpuKernelABC(TKparams Kparams, cudaStream_t stream);
extern "C" void SafeCallGpuKernelGen(TKparams Kparams, cudaStream_t stream);
EOW

# Simple modification to main cpp to use safe versions
cat > rckangaroo_patch.cpp << EOR
// Patch for RCKangaroo.cpp

// Safety wrapper for kernel calls
inline void SafeExecute(RCGpuKang* kang) {
    // Add extra safeguards
    cudaDeviceProp props;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);
    
    if (props.major >= 9) {
        // For RTX 5090, add extra safety measures
        printf("RTX 5090 detected: Using safety measures in thread execution\n");
    }
    
    // Call the original execute function
    kang->Execute();
}
EOR

# Create a minimal patch for RCGpuCore.cu
echo -e "\${BLUE}Building with minimal safety patches...\${NC}"
echo -e "\${YELLOW}Compiling CPU code...\${NC}"

# Compile with safe settings
SAFE_FLAGS="-DRTX5090_MODE -DRTX5090_SAFETY -I."
g++ -O3 -march=native \${SAFE_FLAGS} -I/usr/local/cuda-12.1/include -c RCKangaroo.cpp -o RCKangaroo.o
g++ -O3 -march=native \${SAFE_FLAGS} -include gpukang_patch.cpp -I/usr/local/cuda-12.1/include -c GpuKang.cpp -o GpuKang.o
g++ -O3 -march=native \${SAFE_FLAGS} -I/usr/local/cuda-12.1/include -c Ec.cpp -o Ec.o
g++ -O3 -march=native \${SAFE_FLAGS} -I/usr/local/cuda-12.1/include -c utils.cpp -o utils.o

echo -e "\${YELLOW}Compiling CUDA code...\${NC}"
nvcc -O3 \${SAFE_FLAGS} -gencode=arch=compute_90,code=sm_90 --use_fast_math --threads 0 --gpu-architecture=sm_90 -Xptxas=-v,-O3 -Xcompiler=-O3,-march=native --default-stream=per-thread --maxrregcount=32 cuda_patch.cu RCGpuCore.cu -c -o RCGpuCore.o

if [ \$? -ne 0 ]; then
    echo -e "\${RED}CUDA compilation failed\${NC}"
    exit 1
fi

echo -e "\${YELLOW}Linking final executable...\${NC}"
g++ -O3 -march=native -I/usr/local/cuda-12.1/include -o rckangaroo RCKangaroo.o GpuKang.o Ec.o utils.o RCGpuCore.o -L/usr/local/cuda-12.1/lib64 -lcudart -pthread

if [ \$? -ne 0 ]; then
    echo -e "\${RED}Build failed\${NC}"
    exit 1
fi

echo -e "\${GREEN}Build completed successfully!\${NC}"
echo -e "\${BLUE}Executable: ./rckangaroo\${NC}"
echo
echo -e "\${YELLOW}Recommended command line:\${NC}"
echo -e "./rckangaroo -dp 16 -range 84 -start 1000000000000000000000 -pubkey 0329c4574a4fd8c810b7e42a4b398882b381bcd85e40c6883712912d167c83e73a"
echo

chmod +x rckangaroo

exit 0
EOF

echo -e "${GREEN}Simplified hotfix created successfully!${NC}"
echo -e "${YELLOW}To build with simplified safety settings, run:${NC}"
echo -e "chmod +x build_simple.sh"
echo -e "./build_simple.sh"
echo
echo -e "${BLUE}This simplified build makes minimal changes to ensure compatibility,${NC}"
echo -e "${BLUE}avoiding many of the problems with the previous emergency build.${NC}"

chmod +x build_simple.sh