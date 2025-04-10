#!/bin/bash

# optimize_system.sh - Script to optimize Linux system for RCKangaroo on RTX 5090
# Make the script executable with: chmod +x optimize_system.sh
# Run with sudo: sudo ./optimize_system.sh

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo ./optimize_system.sh)"
    exit 1
fi

echo "Optimizing system for RCKangaroo on RTX 5090..."

# Detect RTX 5090 specifically
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -i "RTX 5090" > /dev/null; then
    echo "RTX 5090 detected - applying specialized optimizations"
    IS_RTX5090=1
else
    echo "Warning: RTX 5090 not detected. Some optimizations may not be optimal for your GPU."
    IS_RTX5090=0
fi

# Set GPU performance to maximum
echo "Setting NVIDIA GPU to maximum performance mode..."
nvidia-smi -pm 1
nvidia-smi --auto-boost-default=0
nvidia-smi -ac $(nvidia-smi --query-gpu=clocks.max.memory,clocks.max.graphics --format=csv,noheader | head -n 1 | sed 's/ //g')

# RTX 5090 specific optimizations
if [ "$IS_RTX5090" -eq 1 ]; then
    echo "Applying RTX 5090 specific tuning..."
    
    # Set optimal memory and compute clocks (values may need adjusting based on specific card model)
    nvidia-smi -i 0 --applications-clocks=2500,2100
    
    # Set power limit to maximum (adjust value based on your specific card)
    MAX_POWER=$(nvidia-smi --query-gpu=power.limit.default --format=csv,noheader | head -n 1 | grep -o '[0-9.]*')
    nvidia-smi -i 0 -pl $MAX_POWER
    
    # Optimize compute mode for exclusive process use
    nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
    
    # Optimize GPU persistence mode
    nvidia-smi -i 0 -pm 1
    
    # Enable resizable BAR if available
    if nvidia-smi --query-gpu=pcie.resizable_bar.max_size --format=csv,noheader | grep -v "N/A" > /dev/null; then
        echo "Enabling resizable BAR support for improved memory transfers"
        # Note: This is typically done in BIOS, but this command checks if it's enabled
        echo "Resizable BAR is available and should be enabled in BIOS for optimal performance"
    fi
fi

# Check and increase PCIe speed if possible
echo "Optimizing PCIe settings..."
if [ -x "$(command -v lspci)" ]; then
    GPU_BUS_ID=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -n 1 | cut -d ":" -f 2- | tr -d '[:space:]')
    PCIE_GEN=$(lspci -vv -s $GPU_BUS_ID | grep "LnkSta:" | grep -oP "Speed \K[0-9\.]+GT\/s")
    PCIE_WIDTH=$(lspci -vv -s $GPU_BUS_ID | grep "LnkSta:" | grep -oP "Width x\K[0-9]+")
    
    echo "Current PCIe: Gen $PCIE_GEN, Width x$PCIE_WIDTH"
    if [ "$IS_RTX5090" -eq 1 ]; then
        echo "RTX 5090 requires PCIe Gen 5 for optimal performance. Please ensure this is enabled in your BIOS."
    else
        echo "To maximize PCIe bandwidth, ensure your system BIOS has PCIe Gen 4 or better enabled"
    fi
fi

# Optimize CPU governor for performance
echo "Setting CPU governor to performance mode..."
if [ -d "/sys/devices/system/cpu/cpu0/cpufreq" ]; then
    for governor in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo "performance" > $governor
    done
    echo "CPU governors set to performance mode"
else
    echo "Could not set CPU governor (normal in some VMs or containers)"
fi

# Optimize memory settings
echo "Optimizing memory settings..."
echo 3 > /proc/sys/vm/drop_caches
echo 1 > /proc/sys/vm/compact_memory
echo 0 > /proc/sys/vm/swappiness
echo 100 > /proc/sys/vm/dirty_ratio
echo 50 > /proc/sys/vm/dirty_background_ratio
echo 1500 > /proc/sys/vm/dirty_writeback_centisecs

# Enhanced memory optimization specific to RTX 5090's large memory requirements
if [ "$IS_RTX5090" -eq 1 ]; then
    echo "Applying advanced memory optimizations for RTX 5090..."
    # Increase max map count for large memory allocations
    sysctl -w vm.max_map_count=16777216
    
    # Optimize transparent hugepages
    echo "always" > /sys/kernel/mm/transparent_hugepage/enabled
    echo "always" > /sys/kernel/mm/transparent_hugepage/defrag
    
    # Set higher memory bandwidth limits for RTX 5090
    if [ -f "/proc/sys/vm/min_free_kbytes" ]; then
        echo 1048576 > /proc/sys/vm/min_free_kbytes
    fi
fi

# Optimize network settings for possible better IPC in multi-GPU setups
echo "Optimizing network settings..."
sysctl -w net.core.rmem_max=26214400
sysctl -w net.core.wmem_max=26214400
sysctl -w net.ipv4.tcp_rmem="4096 87380 16777216"
sysctl -w net.ipv4.tcp_wmem="4096 65536 16777216"

# Enhanced network settings for NVLink if present (for multi-GPU RTX 5090 setups)
if nvidia-smi topo -m | grep -i "NV" > /dev/null; then
    echo "NVLink detected - optimizing for multi-GPU communication"
    # These settings help with multi-GPU communication
    sysctl -w net.core.netdev_max_backlog=250000
    sysctl -w net.core.somaxconn=4096
fi

# Disable NUMA balancing if present (can help with large memory allocations)
if [ -f "/proc/sys/kernel/numa_balancing" ]; then
    echo "Disabling NUMA balancing..."
    echo 0 > /proc/sys/kernel/numa_balancing
fi

# Set real-time priority for the RCKangaroo process if it's running
if pgrep rckangaroo > /dev/null; then
    echo "Setting real-time priority for RCKangaroo process..."
    PID=$(pgrep rckangaroo)
    chrt -f -p 99 $PID
    renice -n -20 -p $PID
fi

# Setup for large page support (can improve memory access performance)
echo "Setting up large page support..."
if [ -d "/sys/kernel/mm/hugepages" ]; then
    # Increased hugepages for RTX 5090's larger memory
    if [ "$IS_RTX5090" -eq 1 ]; then
        echo 512 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
        
        if sysctl vm.nr_hugepages; then
            sysctl -w vm.nr_hugepages=512
        fi
        
        # Try to enable 1GB hugepages if available (better for RTX 5090)
        if [ -d "/sys/kernel/mm/hugepages/hugepages-1048576kB" ]; then
            echo 8 > /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
            echo "1GB hugepages enabled - optimal for RTX 5090"
        fi
    else
        echo 128 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
        
        if sysctl vm.nr_hugepages; then
            sysctl -w vm.nr_hugepages=128
        fi
    fi
    
    if [ -d "/sys/devices/system/node" ]; then
        NODES=$(ls -d /sys/devices/system/node/node* | wc -l)
        PAGES_PER_NODE=0
        
        if [ "$IS_RTX5090" -eq 1 ]; then
            PAGES_PER_NODE=$((512 / NODES))
        else
            PAGES_PER_NODE=$((128 / NODES))
        fi
        
        for node in /sys/devices/system/node/node*; do
            if [ -f "$node/hugepages/hugepages-2048kB/nr_hugepages" ]; then
                echo $PAGES_PER_NODE > "$node/hugepages/hugepages-2048kB/nr_hugepages"
            fi
        done
    fi
    
    # Add instructions for using large pages
    echo "Large pages enabled. To use them with RCKangaroo, run with:"
    if [ "$IS_RTX5090" -eq 1 ]; then
        echo "CUDA_DEVICE_MAX_CONNECTIONS=64 CUDA_AUTO_BOOST=0 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhugetlbfs.so HUGETLB_MORECORE=yes ./rckangaroo [options]"
    else
        echo "CUDA_DEVICE_MAX_CONNECTIONS=32 CUDA_AUTO_BOOST=0 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhugetlbfs.so HUGETLB_MORECORE=yes ./rckangaroo [options]"
    fi
else
    echo "No hugepages support found on this system"
fi

# Create an optimized launch script specifically for RTX 5090
if [ "$IS_RTX5090" -eq 1 ]; then
    echo "Creating optimized launch script for RTX 5090..."
    cat > rtx5090_run.sh << 'EOL'
#!/bin/bash
# Optimized launch script for RCKangaroo on RTX 5090
# Set optimal environment variables for RTX 5090
export CUDA_DEVICE_MAX_CONNECTIONS=64
export CUDA_AUTO_BOOST=0
export CUDA_FORCE_PTX_JIT=1
export CUDA_CACHE_MAXSIZE=2147483647
export CUDA_CACHE_DISABLE=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

# Optional: Enable large pages if your system is set up for them
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhugetlbfs.so
# export HUGETLB_MORECORE=yes

# Launch RCKangaroo with optimal CUDA settings
./rckangaroo "$@"
EOL
    chmod +x rtx5090_run.sh
    echo "Created rtx5090_run.sh - use this script to launch RCKangaroo with optimal settings"
fi

# Set optimal file system caching for large file operations
echo "Optimizing file system for large operations..."
sysctl -w vm.vfs_cache_pressure=50

echo "System optimization complete!"
echo "For best results with RTX 5090, also consider:"
echo "1. Ensuring proper cooling - RTX 5090 runs hot and needs excellent airflow"
echo "2. Using a high-quality power supply (1000W+ gold/platinum certified recommended)"
echo "3. Installing the latest NVIDIA drivers (minimum version 535.129.03)"
echo "4. Closing all other GPU-intensive applications before running RCKangaroo"
if [ "$IS_RTX5090" -eq 1 ]; then
    echo "5. Using the provided rtx5090_run.sh script for optimal performance"
    echo "6. Consider enabling resizable BAR in your BIOS if available"
    echo "7. For multi-GPU setups, ensure NVLink is properly configured"
fi