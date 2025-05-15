# RCKangaroo Project Documentation

This document provides an overview of the source files and major components of the RCKangaroo repository.

---

## File: defs.h
Defines common constants, types and macros used across the project.

## File: Ec.h / Ec.cpp

- Class `Ec`: Implements the elliptic curve operations.
- Key methods:
  - `bool initCurve(params)`: Initialize curve parameters.
  - `Point add(const Point &a, const Point &b)`: Point addition.
  - `Point mul(const BigInt &k, const Point &p)`: Scalar multiplication.

## File: GpuKang.h / GpuKang.cpp

- Class `GpuKang`: Interfaces with GPU to accelerate Kangaroo algorithm.
- Key methods:
  - `void setupKernel()`: Prepare CUDA kernel.
  - `BigInt solve(const CurveParams &params)`: Run kangaroo on GPU.

## File: RCGpuCore.cu

- CUDA kernels for Tame and Wild kangaroo walks.
- Core functions:
  - `__global__ void tameWalk(...)`
  - `__global__ void wildWalk(...)`

## File: RCGpuUtils.h
Utility functions for GPU, including memory management and error checking.

## File: utils.h / utils.cpp

- General-purpose helpers:
  - Big integer conversions
  - Random number generation
  - Timing utilities

## File: RCKangaroo.cpp

- Entry point of the application.
- Parses arguments, loads curve, and chooses CPU or GPU solver.
- Example usage:
  ```bash
  ./RCKangaroo --curve p256 --mode gpu
  ```

## Makefile

Defines build targets for CPU and GPU versions:
- `make cpu`  
- `make gpu`  

## README.md

Contains project description, build instructions and licensing information.

---

*This documentation provides a high-level overview. For detailed API and implementation comments, please refer to Doxygen annotations in the source files.*

## Detailed API Reference

### File: Ec.h / Ec.cpp

Class `EcInt`:
- `EcInt()`: Default constructor, initializes to zero.
- `void Assign(EcInt& val)`: Copy assignment from another `EcInt`.
- `void Set(u64 val)`: Sets the integer to a 64-bit value.
- `void SetZero()`: Resets the integer to zero.
- `bool SetHexStr(const char* str)`: Parses a hexadecimal string and sets the value; returns true on success.
- `void GetHexStr(char* str)`: Outputs the value as a hexadecimal string into the provided buffer.
- `u16 GetU16(int index)`: Retrieves the 16-bit word at the given index.
- `bool Add(EcInt& val)`: Adds `val` to this integer; returns true if carry occurred.
- `bool Sub(EcInt& val)`: Subtracts `val`; returns true if borrow occurred.
- `void Neg()`: Two's-complement negation across full width.
- `void Neg256()`: Negates within 256-bit range.
- `void ShiftRight(int nbits)`: Logical right-shift by specified bits.
- `void ShiftLeft(int nbits)`: Left-shift by specified bits.
- `bool IsLessThanU(EcInt& val)`: Unsigned less-than comparison.
- `bool IsLessThanI(EcInt& val)`: Signed less-than comparison.
- `bool IsEqual(EcInt& val)`: Checks equality.
- `bool IsZero()`: Returns true if the integer is zero.
- `void Mul_u64(EcInt& val, u64 multiplier)`: Multiplies `val` by a 64-bit multiplier.
- `void Mul_i64(EcInt& val, i64 multiplier)`: Multiplies `val` by a signed 64-bit multiplier.
- `void AddModP(EcInt& val)`: Modular addition modulo prime P.
- `void SubModP(EcInt& val)`: Modular subtraction modulo P.
- `void NegModP()`: Modular negation modulo P.
- `void NegModN()`: Modular negation modulo group order N.
- `void MulModP(EcInt& val)`: Modular multiplication modulo P.
- `void InvModP()`: Modular multiplicative inverse modulo P.
- `void SqrtModP()`: Modular square root modulo P.
- `void RndBits(int nbits)`: Generates a random integer with specified bit-length.
- `void RndMax(EcInt& max)`: Generates a random integer in [0, max).

Class `EcPoint`:
- `bool IsEqual(EcPoint& pnt)`: Compares two points for equality.
- `void LoadFromBuffer64(u8* buffer)`: Loads point coordinates from a 64-byte buffer.
- `void SaveToBuffer64(u8* buffer)`: Saves point coordinates into a 64-byte buffer.
- `bool SetHexStr(const char* str)`: Parses compressed or uncompressed point from hex string.

Class `Ec`: Static elliptic-curve operations:
- `static EcPoint AddPoints(EcPoint& p1, EcPoint& p2)`: Adds two EC points.
- `static EcPoint DoublePoint(EcPoint& p)`: Doubles an EC point.
- `static EcPoint MultiplyG(EcInt& k)`: Multiplies the generator point by scalar k.
- `static EcInt CalcY(EcInt& x, bool is_even)`: Computes Y coordinate for given X and parity.
- `static bool IsValidPoint(EcPoint& p)`: Verifies point lies on the curve.

Global functions:
- `void InitEc()`: Initializes global curve parameters (P, N, G) from secp256k1 constants.
- `void DeInitEc()`: Frees any allocated curve resources.
- `void SetRndSeed(u64 seed)`: Seeds the pseudo-random generator.

### File: GpuKang.h / GpuKang.cpp

Struct `EcJMP`:
- `EcPoint p`: Jump target point.
- `EcInt dist`: Corresponding distance value.

Struct `TPointPriv`:
- `u64 x[4], y[4], priv[4]`: Raw 256-bit storage for point and private key.

Class `RCGpuKang`:
- `bool Prepare(EcPoint P, int Range, int DP, EcJMP* j1, EcJMP* j2, EcJMP* j3)`: Prepares kernels and memory parameterized by point P, search range, and jump tables.
- `void Execute()`: Starts GPU kernels for tame and wild walks.
- `void Stop()`: Sets internal flag to halt execution.
- `int CalcKangCnt()`: Calculates optimal number of kangaroos based on range and resources.
- `int GetStatsSpeed()`: Returns rolling average of iterations per second.
- Internal helpers:
  - `void GenerateRndDistances()`: Populates random jump distances.
  - `bool Start()`: Allocates GPU memory and launches kernels.
  - `void Release()`: Deallocates GPU resources.
  - `int Dbg_CheckKangs()`: [DEBUG_MODE] Validates kernel results against CPU fallback.

### File: utils.h / utils.cpp

- `bool parse_u8(const char* s, u8* res)`: Converts two-character hex string to byte.
- `u64 toU64(const EcInt& a)`: Extracts u64 from low 64 bits of `EcInt`.
- `u64 GetTimeNs()`: Returns high-resolution monotonic timestamp in nanoseconds.
- `u32 FastRand()`: Fast 32-bit pseudo-random number generator.

### File: RCGpuUtils.h

- `void cudaCheckError(const char* msg)`: Asserts and prints last CUDA error with message.
- `void* cudaMallocOrDie(size_t size)`: Allocates GPU memory or exits on failure.

### File: RCGpuCore.cu

- `__global__ void tameWalk(...)`: Kernel implementing tame kangaroo walks on device.
- `__global__ void wildWalk(...)`: Kernel implementing wild kangaroo walks.

### File: RCKangaroo.cpp

- `int main(int argc, char** argv)`: Entry point; parses CLI, initializes curve, and dispatches CPU/GPU solver.
- `void parseArgs(int argc, char** argv, ...)`: Processes command-line switches.
- `void runCpuSolver(...)`: Executes CPU-based kangaroo algorithm.
- `void runGpuSolver(...)`: Executes GPU-based kangaroo algorithm.

### File: defs.h

- Typedefs: `u8, u32, u64, i64`
- Common macros for alignment and visibility.
- Declarations for curve parameters, seeds, and random context.
