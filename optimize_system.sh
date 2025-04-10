#!/bin/bash

# optimize_system.sh - Script to optimize Linux system for RCKangaroo on RTX 5090
# Make the script executable with: chmod +x optimize_system.sh
# Run with sudo: sudo ./optimize_system.sh

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo ./optimize_system.sh)"
    exit 1
fi

echo "Optimizing system for RCKangaroo on RTX 5090..."

# Set GPU performance to maximum
echo "Setting NVIDIA GPU to maximum performance mode..."
nvidia-smi -pm 1
nvidia-smi --auto-boost-default=0
nvidia-smi -ac $(nvidia-smi --query-gpu=clocks.max.memory,clocks.max.graphics --format=csv,noheader | head -n 1 | sed 's/ //g')

# Check and increase PCIe speed if possible
echo "Optimizing PCIe settings..."
if [ -x "$(command -v lspci)" ]; then
    GPU_BUS_ID=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -n 1 | cut -d ":" -f 2- | tr -d '[:space:]')
    PCIE_GEN=$(lspci -vv -s $GPU_BUS_ID | grep "LnkSta:" | grep -oP "Speed \K[0-9\.]+GT\/s")
    PCIE_WIDTH=$(lspci -vv -s $GPU_BUS_ID | grep "LnkSta:" | grep -oP "Width x\K[0-9]+")
    
    echo "Current PCIe: Gen $PCIE_GEN, Width x$PCIE_WIDTH"
    echo "To maximize PCIe bandwidth, ensure your system BIOS has PCIe Gen 4 or better enabled"
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

# Optimize network settings for possible better IPC in multi-GPU setups
echo "Optimizing network settings..."
sysctl -w net.core.rmem_max=26214400
sysctl -w net.core.wmem_max=26214400
sysctl -w net.ipv4.tcp_rmem="4096 87380 16777216"
sysctl -w net.ipv4.tcp_wmem="4096 65536 16777216"

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
    echo 128 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
    
    if sysctl vm.nr_hugepages; then
        sysctl -w vm.nr_hugepages=128
    fi
    
    if [ -d "/sys/devices/system/node" ]; then
        for node in /sys/devices/system/node/node*; do
            if [ -f "$node/hugepages/hugepages-2048kB/nr_hugepages" ]; then
                echo 32 > "$node/hugepages/hugepages-2048kB/nr_hugepages"
            fi
        done
    fi
    
    # Add instructions for using large pages
    echo "Large pages enabled. To use them with RCKangaroo, run with:"
    echo "CUDA_DEVICE_MAX_CONNECTIONS=32 CUDA_AUTO_BOOST=0 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhugetlbfs.so HUGETLB_MORECORE=yes ./rckangaroo [options]"
else
    echo "No hugepages support found on this system"
fi

echo "System optimization complete!"
echo "For best results, also consider:"
echo "1. Ensuring proper cooling for your RTX 5090"
echo "2. Using a high-quality power supply"
echo "3. Updating to the latest NVIDIA drivers"
echo "4. Closing other GPU-intensive applications before running RCKangaroo"