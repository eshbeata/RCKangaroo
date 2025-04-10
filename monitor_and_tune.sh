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
    if ! pgrep -f rckangaroo > /dev/null; then
        echo "RCKangaroo is not running. Please start it first."
        exit 1
    fi
}

# Function to get detailed GPU stats
get_gpu_stats() {
    echo "============ GPU STATS ============"
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory --format=csv,noheader
    
    # Get memory usage - fixed to use pgrep correctly
    echo "============ GPU MEMORY ============"
    RC_PID=$(pgrep -f rckangaroo)
    if [ -n "$RC_PID" ]; then
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep "$RC_PID" || echo "No GPU memory usage found for RCKangaroo"
    else
        echo "RCKangaroo process not found"
    fi
    
    # Get detailed process statistics - fixed to use pgrep correctly
    echo "============ PROCESS STATS ============"
    RC_PID=$(pgrep -f rckangaroo)
    if [ -n "$RC_PID" ]; then
        ps -p $RC_PID -o pid,ppid,cmd,%cpu,%mem,rss || echo "Could not get process stats"
    else
        echo "RCKangaroo process not found"
    fi
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
    if [ -n "$GPU_UTIL" ] && [ -n "$MEM_UTIL" ]; then
        if [ "$GPU_UTIL" -lt 90 ] && [ "$MEM_UTIL" -gt 80 ]; then
            echo "Memory-bound operation detected. Optimizing memory clocks..."
            CURRENT_MEM_CLOCK=$(nvidia-smi --query-gpu=clocks.current.memory --format=csv,noheader | sed 's/ MHz//g')
            MAX_MEM_CLOCK=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader | sed 's/ MHz//g')
            
            if [ -n "$CURRENT_MEM_CLOCK" ] && [ -n "$MAX_MEM_CLOCK" ] && [ "$CURRENT_MEM_CLOCK" -lt "$MAX_MEM_CLOCK" ]; then
                if [ -x "$(command -v nvidia-settings)" ] && [ -n "$DISPLAY" ]; then
                    echo "Trying to increase memory clock..."
                    nvidia-settings -a "[gpu:0]/GPUMemoryTransferRateOffset[3]=500"
                else
                    echo "Cannot adjust memory clocks automatically - nvidia-settings not available or no display."
                    echo "For maximum performance, consider manually setting memory overclocks."
                fi
            fi
        fi
    else
        echo "Could not determine GPU or memory utilization"
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
                # Use AWK for calculations to avoid bc dependency
                RECENT_AVG=$(tail -n 5 speed_history.tmp | awk '{sum+=$1} END {print sum/5}')
                PREV_AVG=$(head -n 5 speed_history.tmp | awk '{sum+=$1} END {print sum/5}')
                
                if awk "BEGIN {exit !($RECENT_AVG > $PREV_AVG)}"; then
                    echo "Performance trend: Improving ⬆️"
                elif awk "BEGIN {exit !($RECENT_AVG < $PREV_AVG * 0.95)}"; then
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
            # Show the last few lines of the log to help diagnose issues
            echo "Last 5 lines of rckangaroo.log:"
            tail -n 5 rckangaroo.log
        fi
    else
        echo "No log file found. RCKangaroo should be running with output redirection: ./rckangaroo [options] > rckangaroo.log"
    fi
}

# Function to show system statistics
show_system_stats() {
    echo "============ SYSTEM STATS ============"
    echo "CPU Usage:"
    mpstat 1 1 | tail -n 1
    
    echo "Memory Usage:"
    free -h
    
    echo "Disk I/O:"
    iostat -xh 1 1 | grep -v "loop" | tail -n +6 | head -n 2
}

# Main monitoring loop
clear
echo "Starting RCKangaroo performance monitor for RTX 5090..."
echo "Press Ctrl+C to stop monitoring."

# Create clean history file
> speed_history.tmp

# Initial check
RC_PID=$(pgrep -f rckangaroo)
if [ -z "$RC_PID" ]; then
    echo "WARNING: RCKangaroo does not appear to be running."
    echo "Starting monitoring anyway, but some stats may be unavailable."
fi

while true; do
    clear
    echo "RCKangaroo Performance Monitor - Refresh every ${INTERVAL}s"
    echo "Time: $(date)"
    
    get_gpu_stats
    check_thermal_throttling
    optimize_clocks
    show_system_stats
    analyze_output
    
    echo -e "\nPress Ctrl+C to exit."
    sleep $INTERVAL
done