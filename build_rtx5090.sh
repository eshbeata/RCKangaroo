#!/bin/bash

# Set colorful output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}= RCKangaroo RTX 5090 Launcher =${NC}"
echo -e "${GREEN}================================${NC}"

echo -e "\n${BLUE}=== DETECTING CUDA INSTALLATION ===${NC}\n"

# Check for CUDA installation
if [ -d "/usr/local/cuda" ]; then
    CUDA_PATH="/usr/local/cuda"
    CUDA_VERSION=$(cat $CUDA_PATH/version.txt | grep -o "[0-9]\+\.[0-9]\+\.[0-9]\+" || echo "unknown")
    echo -e "${GREEN}✓ CUDA detected: Version $CUDA_VERSION at $CUDA_PATH${NC}"
    
    # Check CUDA version is sufficient (need 11.0+)
    MAJOR_VERSION=$(echo $CUDA_VERSION | cut -d. -f1)
    if [ "$MAJOR_VERSION" -ge "11" ]; then
        echo -e "${GREEN}✓ CUDA version is sufficient for RTX 5090${NC}"
    else
        echo -e "${RED}✗ CUDA version is too old. Need 11.0+ for RTX 5090${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ CUDA not found in /usr/local/cuda${NC}"
    echo -e "Please install CUDA 11.0 or later"
    exit 1
fi

echo -e "\n${BLUE}=== DETECTING NVIDIA GPUS ===${NC}\n"

# Check for NVIDIA GPU using nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)
    GPU_COUNT=$(echo "$GPU_INFO" | wc -l)
    
    echo -e "${GREEN}✓ Found $GPU_COUNT NVIDIA GPU(s):${NC}"
    echo "$GPU_INFO" | while read -r line; do
        NAME=$(echo $line | cut -d, -f1)
        DRIVER=$(echo $line | cut -d, -f2)
        MEMORY=$(echo $line | cut -d, -f3)
        echo "   - $NAME (Driver: $DRIVER, Memory: $MEMORY)"
    done
    
    # Check if RTX 5090 is detected
    if echo "$GPU_INFO" | grep -i "RTX" | grep -i "50" &> /dev/null; then
        echo -e "${GREEN}✓ RTX 5090 or compatible GPU detected!${NC}"
    else
        echo -e "${YELLOW}! No RTX 5090 detected, but will try to continue${NC}"
    fi
else
    echo -e "${RED}✗ nvidia-smi not found. Are NVIDIA drivers installed?${NC}"
    exit 1
fi

echo -e "\n${BLUE}=== SETTING UP ENVIRONMENT ===${NC}\n"

# Set environment variables
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
export CUDA_DEVICE_MAX_CONNECTIONS=32
export CUDA_VISIBLE_DEVICES=0 # Use first GPU
export RTX5090_MODE=1
export OPTIMIZE_FOR_RTX5090=1

echo -e "${GREEN}✓ Environment variables set${NC}"

echo -e "\n${BLUE}=== OPTIMIZING MAKEFILE ===${NC}\n"

# Add RTX 5090 specific flags to Makefile
if [ -f Makefile ]; then
    # Create backup
    cp Makefile Makefile.backup
    
    # Update NVCC flags for RTX 5090
    if grep -q "RTX5090" Makefile; then
        echo -e "${GREEN}✓ Makefile is already optimized for RTX 5090${NC}"
    else
        sed -i 's/\(NVCCFLAGS :=.*\)/\1 -DRTX5090_MODE -DOPTIMIZE_FOR_RTX5090 -Xptxas=-v,-O3 --default-stream=per-thread --maxrregcount=64 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90/' Makefile
        echo -e "${GREEN}✓ Makefile is optimized for RTX 5090${NC}"
    fi
else
    echo -e "${RED}✗ Makefile not found${NC}"
    exit 1
fi

echo -e "\n${BLUE}=== BUILDING RCKANGAROO ===${NC}\n"

# Clean any previous build
echo -e "${YELLOW}! Cleaning previous build...${NC}"
make clean

# Build with RTX 5090 optimizations
echo -e "${YELLOW}! Building RCKangaroo with optimized settings for RTX 5090...${NC}"
make -j$(nproc) CXXFLAGS="-O3 -march=native -DRTX5090_MODE -DOPTIMIZE_FOR_RTX5090"

# Check if build was successful
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ Build successful!${NC}"
    
    # Create a launcher script
    cat > run_rtx5090.sh << 'EOF'
#!/bin/bash
export RTX5090_MODE=1
export OPTIMIZE_FOR_RTX5090=1
export CUDA_VISIBLE_DEVICES=0
./rckangaroo "$@"
EOF
    chmod +x run_rtx5090.sh
    
    echo -e "\n${GREEN}✓ Created run_rtx5090.sh launcher script${NC}"
    echo -e "\n${YELLOW}! To run RCKangaroo, use: ./run_rtx5090.sh [options]${NC}"
else
    echo -e "\n${RED}✗ Build failed${NC}"
    # Restore Makefile
    mv Makefile.backup Makefile
    exit 1
fi