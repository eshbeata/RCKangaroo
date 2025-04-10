#!/bin/bash

# build_and_run.sh - Script to build and run RCKangaroo optimized for RTX 5090
# Make executable with: chmod +x build_and_run.sh

# Color codes for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print section headers
print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error messages
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to print warning messages
print_warning() {
    echo -e "${YELLOW}! $1${NC}"
}

# Function to detect CUDA version
detect_cuda() {
    print_header "DETECTING CUDA INSTALLATION"
    
    if command_exists nvcc; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        CUDA_PATH=$(which nvcc | rev | cut -d'/' -f3- | rev)
        print_success "CUDA detected: Version $CUDA_VERSION at $CUDA_PATH"
        
        # Check if CUDA version is sufficient for RTX 5090
        if [ "$(echo "$CUDA_VERSION >= 12.0" | bc)" -eq 1 ]; then
            print_success "CUDA version is sufficient for RTX 5090"
        else
            print_warning "CUDA version may be too old for RTX 5090. Consider upgrading to CUDA 12.0 or newer."
        fi
    else
        print_error "CUDA not found! Please install CUDA Toolkit 12.0 or newer."
        exit 1
    fi
}

# Function to detect and validate NVIDIA GPUs
detect_gpus() {
    print_header "DETECTING NVIDIA GPUS"
    
    if command_exists nvidia-smi; then
        # Get GPU info
        GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)
        GPU_COUNT=$(echo "$GPU_INFO" | wc -l)
        
        if [ $GPU_COUNT -gt 0 ]; then
            print_success "Found $GPU_COUNT NVIDIA GPU(s):"
            echo "$GPU_INFO" | awk -F', ' '{printf "   - %s (Driver: %s, Memory: %s)\n", $1, $2, $3}'
            
            # Check for RTX 5090
            if echo "$GPU_INFO" | grep -q "RTX 50"; then
                print_success "RTX 5090 or compatible GPU detected!"
                RTX5090_DETECTED=true
            else
                print_warning "No RTX 5090 detected. Performance optimization will be limited."
                RTX5090_DETECTED=false
            fi
        else
            print_error "No NVIDIA GPUs detected!"
            exit 1
        fi
    else
        print_error "nvidia-smi not found! Please install NVIDIA drivers."
        exit 1
    fi
}

# Function to set up environment variables
setup_environment() {
    print_header "SETTING UP ENVIRONMENT"
    
    # Set environment variables
    export CUDA_DEVICE_MAX_CONNECTIONS=32
    export CUDA_AUTO_BOOST=0
    
    # Check if we can use large pages
    if [ -d "/sys/kernel/mm/hugepages" ]; then
        if [ -f "/usr/lib/x86_64-linux-gnu/libhugetlbfs.so" ]; then
            print_success "Large page support enabled"
            export HUGETLB_MORECORE=yes
            export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhugetlbfs.so
        fi
    fi
    
    print_success "Environment variables set"
}

# Function to optimize the makefile for RTX 5090
optimize_makefile() {
    print_header "OPTIMIZING MAKEFILE"
    
    # Update CUDA path if different from default
    if [ "$CUDA_PATH" != "/usr/local/cuda" ]; then
        sed -i "s|CUDA_PATH ?= /usr/local/cuda|CUDA_PATH ?= $CUDA_PATH|g" Makefile
        print_success "Updated CUDA path in Makefile to $CUDA_PATH"
    fi
    
    print_success "Makefile is optimized for RTX 5090"
}

# Function to build RCKangaroo
build_rckangaroo() {
    print_header "BUILDING RCKANGAROO"
    
    # First clean any previous build
    print_warning "Cleaning previous build..."
    make clean
    
    # Build with optimized settings
    print_warning "Building RCKangaroo with optimized settings for RTX 5090..."
    make -j$(nproc)
    
    if [ $? -eq 0 ]; then
        print_success "Build completed successfully"
    else
        print_error "Build failed"
        exit 1
    fi
}

# Function to run system optimization if user has sudo rights
run_system_optimization() {
    print_header "SYSTEM OPTIMIZATION"
    
    if [ -x ./optimize_system.sh ]; then
        if command_exists sudo && sudo -n true 2>/dev/null; then
            print_warning "Running system optimization (requires sudo)..."
            sudo ./optimize_system.sh
        else
            print_warning "System optimization requires sudo. Run './optimize_system.sh' manually with sudo when ready."
        fi
    else
        print_warning "optimize_system.sh not found or not executable. Run 'chmod +x optimize_system.sh' first."
    fi
}

# Function to run RCKangaroo
run_rckangaroo() {
    print_header "RUNNING RCKANGAROO"
    
    if [ ! -f "./rckangaroo" ]; then
        print_error "rckangaroo executable not found. Build failed?"
        exit 1
    fi
    
    if [ $# -eq 0 ]; then
        print_warning "No parameters provided. Running in benchmark mode."
        ./rckangaroo > rckangaroo.log &
        RCPID=$!
        echo "RCKangaroo is running in background (PID: $RCPID). Output is being saved to rckangaroo.log"
    else
        print_warning "Running with provided parameters: $@"
        ./rckangaroo "$@" > rckangaroo.log &
        RCPID=$!
        echo "RCKangaroo is running in background (PID: $RCPID). Output is being saved to rckangaroo.log"
    fi
    
    # Start the monitoring script if available
    if [ -x ./monitor_and_tune.sh ]; then
        sleep 2 # Give RCKangaroo a moment to start
        print_warning "Starting performance monitoring..."
        ./monitor_and_tune.sh
    else
        print_warning "monitor_and_tune.sh not found or not executable. Run 'chmod +x monitor_and_tune.sh' first."
        print_warning "You can monitor output with: tail -f rckangaroo.log"
    fi
}

# Main script execution begins here
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}= RCKangaroo RTX 5090 Launcher =${NC}"
echo -e "${GREEN}================================${NC}"

# Parse command-line arguments
BUILD_ONLY=false
RUN_ONLY=false
SKIP_SYSTEM_OPT=false
RCKANGAROO_ARGS=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --build-only) BUILD_ONLY=true; shift ;;
        --run-only) RUN_ONLY=true; shift ;;
        --skip-system-opt) SKIP_SYSTEM_OPT=true; shift ;;
        --help) 
            echo "Usage: ./build_and_run.sh [options] [-- rckangaroo_args]"
            echo "Options:"
            echo "  --build-only         Only build, don't run"
            echo "  --run-only           Only run, don't build"
            echo "  --skip-system-opt    Skip system optimization"
            echo "  --help               Show this help message"
            echo "  -- rckangaroo_args   Pass remaining arguments to rckangaroo"
            exit 0
            ;;
        --) 
            shift
            RCKANGAROO_ARGS="$@"
            break
            ;;
        *) 
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Detect required tools and hardware
detect_cuda
detect_gpus
setup_environment

# Skip building if --run-only is specified
if [ "$RUN_ONLY" = false ]; then
    optimize_makefile
    build_rckangaroo
fi

# Skip running if --build-only is specified
if [ "$BUILD_ONLY" = false ]; then
    # Skip system optimization if --skip-system-opt is specified
    if [ "$SKIP_SYSTEM_OPT" = false ]; then
        run_system_optimization
    fi
    
    # Run with any provided arguments
    run_rckangaroo $RCKANGAROO_ARGS
fi

print_header "COMPLETED"
if [ "$BUILD_ONLY" = true ]; then
    print_success "Build completed. Run with: ./build_and_run.sh --run-only -- [arguments]"
elif [ "$RUN_ONLY" = true ]; then
    print_success "RCKangaroo is running. Check rckangaroo.log for output."
else
    print_success "Build and run completed. RCKangaroo is running."
fi