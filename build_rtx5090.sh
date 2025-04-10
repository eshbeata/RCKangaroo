#!/bin/bash

# Build script for RCKangaroo with RTX 5090 optimizations

# Set colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}= RCKangaroo RTX 5090 Optimized Build =${NC}"
echo -e "${GREEN}========================================${NC}"
echo

# Clean previous build
echo -e "${YELLOW}Cleaning previous build...${NC}"
make clean

# Define RTX 5090 optimizations
echo -e "${BLUE}Setting RTX 5090 optimizations...${NC}"

# Create optimized settings header
cat > rtx5090_settings.h << EOF
// RTX 5090 optimized settings - auto-generated
#pragma once

// Safety limits to prevent segmentation faults
#define RTX5090_MODE
#define RTX5090_BLOCK_SIZE 256
#define RTX5090_PNT_GROUP_CNT 12
#define RTX5090_STEP_CNT 96
#define RTX5090_MD_LEN 12
#define RTX5090_DPTABLE_MAX_CNT 8
#define RTX5090_MAX_CNT_LIST 65536
#define RTX5090_MAX_DP_CNT 16384

// Memory layout optimizations
#define OPTIMIZE_FOR_RTX5090
EOF

# Create build script
echo -e "${BLUE}Creating optimized CUDA flags...${NC}"

# Add optimization flags to the build
NVCC_FLAGS="-O3 -DRTX5090_MODE -DOPTIMIZE_FOR_RTX5090 -gencode=arch=compute_90,code=sm_90 --use_fast_math --threads 0 --gpu-architecture=sm_90 -Xptxas=\"-v,-O3\" -Xcompiler=\"-O3,-march=native\" --default-stream=per-thread --maxrregcount=64"

# Build with optimized settings
echo -e "${BLUE}Building with RTX 5090 optimizations...${NC}"
echo -e "${YELLOW}Compiling CPU code...${NC}"
g++ -O3 -march=native -DRTX5090_MODE -DOPTIMIZE_FOR_RTX5090 -I/usr/local/cuda/include -c RCKangaroo.cpp -o RCKangaroo.o
g++ -O3 -march=native -DRTX5090_MODE -DOPTIMIZE_FOR_RTX5090 -I/usr/local/cuda/include -c GpuKang.cpp -o GpuKang.o
g++ -O3 -march=native -DRTX5090_MODE -DOPTIMIZE_FOR_RTX5090 -I/usr/local/cuda/include -c Ec.cpp -o Ec.o
g++ -O3 -march=native -DRTX5090_MODE -DOPTIMIZE_FOR_RTX5090 -I/usr/local/cuda/include -c utils.cpp -o utils.o

echo -e "${YELLOW}Compiling CUDA code...${NC}"
nvcc $NVCC_FLAGS -c RCGpuCore.cu -o RCGpuCore.o

if [ $? -ne 0 ]; then
    echo -e "${RED}CUDA compilation failed${NC}"
    exit 1
fi

echo -e "${YELLOW}Linking final executable...${NC}"
g++ -O3 -march=native -I/usr/local/cuda/include -o rckangaroo RCKangaroo.o GpuKang.o Ec.o utils.o RCGpuCore.o -L/usr/local/cuda/lib64 -lcudart -pthread

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