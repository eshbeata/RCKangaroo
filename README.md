# RCKangaroo

(c) 2024, RetiredCoder (RC)

RCKangaroo is free and open-source (GPLv3).
This software demonstrates efficient GPU implementation of SOTA Kangaroo method for solving ECDLP. 
It's part #3 of my research, you can find more details here: https://github.com/RetiredC

Discussion thread: https://bitcointalk.org/index.php?topic=5517607

## Features

- Lowest K=1.15, it means 1.8 times less required operations compared to classic method with K=2.1, also it means that you need 1.8 times less memory to store DPs.
- Fast, about 8GKeys/s on RTX 4090, 4GKeys/s on RTX 3090, and up to 14GKeys/s on RTX 5090.
- Keeps DP overhead as small as possible.
- Supports ranges up to 170 bits.
- Both Windows and Linux are supported.
- Optimized for the latest RTX 5090 GPUs.

## RTX 5090 Optimization Guide for Linux

### Prerequisites

- NVIDIA RTX 5090 or compatible GPU
- CUDA Toolkit 12.0 or newer
- GCC 9.0 or newer
- Linux with recent NVIDIA drivers (535.x or newer recommended)

### Compilation and Optimization

The project includes several scripts that make building, optimizing, and running RCKangaroo on RTX 5090 easy:

1. First, make the scripts executable:
   ```bash
   chmod +x build_and_run.sh optimize_system.sh monitor_and_tune.sh
   ```

2. For best performance, optimize your system (requires sudo):
   ```bash
   sudo ./optimize_system.sh
   ```
   This script:
   - Sets GPU to maximum performance mode
   - Optimizes CPU governor settings
   - Configures memory settings
   - Sets up large page support
   - Optimizes PCIe settings

3. Build and run with optimizations for RTX 5090:
   ```bash
   ./build_and_run.sh
   ```

For more control, you can use these options:
- `--build-only`: Only build, don't run
- `--run-only`: Only run, don't build
- `--skip-system-opt`: Skip system optimization steps
- `-- [args]`: Pass additional arguments to RCKangaroo

Example: Build and run with custom parameters:
```bash
./build_and_run.sh -- -dp 16 -range 84 -start 1000000000000000000000 -pubkey 0329c4574a4fd8c810b7e42a4b398882b381bcd85e40c6883712912d167c83e73a
```

### Performance Monitoring

While RCKangaroo is running, you can monitor its performance with:
```bash
./monitor_and_tune.sh
```

This tool shows:
- Real-time GPU statistics
- Memory usage
- Performance trends
- Thermal throttling detection
- Automatic clock optimization

## RTX 5090-Specific Optimizations

This version includes several optimizations specifically for RTX 5090 GPUs:

1. **CUDA Stream Optimization**: Separate compute and memory streams for better parallel processing
2. **Memory Pool Management**: Faster memory allocation with optimized memory pools
3. **Persistent L2 Cache**: Maximizes usage of RTX 5090's large L2 cache
4. **Thread Configuration**: Dynamic adjustment based on GPU Streaming Multiprocessor count
5. **Block and Group Size Optimization**: Tuned for optimal RTX 5090 occupancy
6. **Kernel Parameter Tuning**: Parameters specialized for RTX 5090 architecture
7. **Buffer Size Increases**: Larger buffers to take advantage of RTX 5090's VRAM
8. **Loop Detection Enhancement**: Improved detection to reduce kangaroo wastage

## Command Line Parameters

**-gpu**		which GPUs are used, for example, "035" means that GPUs #0, #3 and #5 are used. If not specified, all available GPUs are used. 

**-pubkey**		public key to solve, both compressed and uncompressed keys are supported. If not specified, software starts in benchmark mode and solves random keys. 

**-start**		start offset of the key, in hex. Mandatory if "-pubkey" option is specified. For example, for puzzle #85 start offset is "1000000000000000000000". 

**-range**		bit range of private the key. Mandatory if "-pubkey" option is specified. For example, for puzzle #85 bit range is "84" (84 bits). Must be in range 32...170. 

**-dp**		DP bits. Must be in range 14...60. Low DP bits values cause larger DB but reduces DP overhead and vice versa. 

**-max**		option to limit max number of operations. For example, value 5.5 limits number of operations to 5.5 * 1.15 * sqrt(range), software stops when the limit is reached. 

**-tames**		filename with tames. If file not found, software generates tames (option "-max" is required) and saves them to the file. If the file is found, software loads tames to speedup solving. 

## Example Commands

Sample command line for puzzle #85:
```
./rckangaroo -dp 16 -range 84 -start 1000000000000000000000 -pubkey 0329c4574a4fd8c810b7e42a4b398882b381bcd85e40c6883712912d167c83e73a
```

Sample command to generate tames:
```
./rckangaroo -dp 16 -range 76 -tames tames76.dat -max 10
```

## Performance Notes for RTX 5090

- For optimal performance on RTX 5090, use DP values between 16-20
- The RTX 5090 performs best with larger kangaroo group sizes
- When using multiple RTX 5090 GPUs, enable peer access for faster communication
- Monitor temperatures to avoid thermal throttling
- For maximum performance, consider running the system optimization script

## Some Notes

Fastest ECDLP solvers will always use SOTA/SOTA+ method, as it's 1.4/1.5 times faster and requires less memory for DPs compared to the best 3-way kangaroos with K=1.6. 
Even if you already have a faster implementation of kangaroo jumps, incorporating SOTA method will improve it further. 
While adding the necessary loop-handling code will cause you to lose about 5–15% of your current speed, the SOTA method itself will provide a 40% performance increase. 
Overall, this translates to roughly a 25% net improvement, which should not be ignored if your goal is to build a truly fast solver. 

## Changelog

v3.1:
- Optimized for RTX 5090 GPUs with specialized Linux support
- Added CUDA stream optimizations for parallel processing
- Implemented memory pool management for RTX 5090
- Added advanced system optimization for Linux
- Included performance monitoring tools
- Enhanced L2 cache utilization for RTX 5090
- Optimized kernel parameters based on SM count

v3.0:
- added "-tames" and "-max" options.
- fixed some bugs.

v2.0:
- added support for 30xx, 20xx and 1xxx cards.
- some minor changes.

v1.1:
- added ability to start software on 30xx cards.

v1.0:
- initial release.