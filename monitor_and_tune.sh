#!/bin/bash

# monitor_and_tune.sh - Script to monitor and tune RCKangaroo performance on RTX 5090
# Make the script executable with: chmod +x monitor_and_tune.sh

# Default interval in seconds between updates
INTERVAL=5

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--interval) INTERVAL="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Function to check if RCKangaroo is running
check_rc_running() {
    if ! pgrep rckangaroo > /dev/null; then
        echo "RCKangaroo is not running. Please start it first."
        exit 1
    fi
}

# Function to get detailed GPU stats
get_gpu_stats() {
    echo "============ GPU STATS ============"
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory --format=csv,noheader
    
    # Get memory usage
    echo "============ GPU MEMORY ============"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep $(pgrep rckangaroo)
    
    # Get detailed process statistics
    echo "============ PROCESS STATS ============"
    ps -p $(pgrep rckangaroo) -o pid,ppid,cmd,%cpu,%mem,rss
}

# Function to check GPU thermal throttling
check_thermal_throttling() {
    THROTTLE_STATUS=$(nvidia-smi -q -d PERFORMANCE | grep "Throttling Reason" -A 10)
    if echo "$THROTTLE_STATUS" | grep -q "Active: Yes"; then
        echo "WARNING: GPU is being throttled!"
        echo "$THROTTLE_STATUS"
        echo "Consider improving cooling or reducing power limit."
    fi
}

# Function to optimize clocks if needed
optimize_clocks() {
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | sed 's/ %//g')
    MEM_UTIL=$(nvidia-smi --query-gpu=utilization.memory --format=csv,noheader | sed 's/ %//g')
    
    # If GPU utilization is below 90% but memory is high, we might be memory-bound
    if [ "$GPU_UTIL" -lt 90 ] && [ "$MEM_UTIL" -gt 80 ]; then
        echo "Memory-bound operation detected. Optimizing memory clocks..."
        CURRENT_MEM_CLOCK=$(nvidia-smi --query-gpu=clocks.current.memory --format=csv,noheader | sed 's/ MHz//g')
        MAX_MEM_CLOCK=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader | sed 's/ MHz//g')
        
        if [ "$CURRENT_MEM_CLOCK" -lt "$MAX_MEM_CLOCK" ]; then
            if [ -x "$(command -v nvidia-settings)" ] && [ -n "$DISPLAY" ]; then
                echo "Trying to increase memory clock..."
                nvidia-settings -a "[gpu:0]/GPUMemoryTransferRateOffset[3]=500"
            else
                echo "Cannot adjust memory clocks automatically - nvidia-settings not available or no display."
                echo "For maximum performance, consider manually setting memory overclocks."
            fi
        fi
    fi
}

# Function to analyze and display RCKangaroo output for performance insights
analyze_output() {
    if [ -f "rckangaroo.log" ]; then
        echo "============ PERFORMANCE ANALYSIS ============"
        SPEED=$(grep "Speed:" rckangaroo.log | tail -n 1)
        if [ -n "$SPEED" ]; then
            echo "Current performance: $SPEED"
            
            # Extract raw speed value for trend analysis
            RAW_SPEED=$(echo "$SPEED" | grep -oP "Speed: \K[0-9]+" || echo "0")
            echo "$RAW_SPEED" >> speed_history.tmp
            
            # Calculate trend if we have enough data points
            if [ $(wc -l < speed_history.tmp) -gt 5 ]; then
                RECENT_AVG=$(tail -n 5 speed_history.tmp | awk '{sum+=$1} END {print sum/5}')
                PREV_AVG=$(head -n 5 speed_history.tmp | awk '{sum+=$1} END {print sum/5}')
                
                if (( $(echo "$RECENT_AVG > $PREV_AVG" | bc -l) )); then
                    echo "Performance trend: Improving ⬆️"
                elif (( $(echo "$RECENT_AVG < $PREV_AVG * 0.95" | bc -l) )); then
                    echo "Performance trend: Decreasing ⬇️ - Check for thermal throttling"
                else
                    echo "Performance trend: Stable ➡️"
                fi
            fi
            
            # Keep the history file from growing too large
            if [ $(wc -l < speed_history.tmp) -gt 100 ]; then
                tail -n 50 speed_history.tmp > speed_history.new
                mv speed_history.new speed_history.tmp
            fi
        else
            echo "No performance data found in logs yet."
        fi
    else
        echo "No log file found. Consider running RCKangaroo with output redirection: ./rckangaroo [options] > rckangaroo.log"
    fi
}

# Main monitoring loop
clear
echo "Starting RCKangaroo performance monitor for RTX 5090..."
echo "Press Ctrl+C to stop monitoring."

# Create clean history file
> speed_history.tmp

# Initial check
check_rc_running

while true; do
    clear
    echo "RCKangaroo Performance Monitor - Refresh every ${INTERVAL}s"
    echo "Time: $(date)"
    
    get_gpu_stats
    check_thermal_throttling
    optimize_clocks
    analyze_output
    
    echo -e "\nPress Ctrl+C to exit."
    sleep $INTERVAL
done