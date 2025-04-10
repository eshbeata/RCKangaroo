CC := g++
NVCC := nvcc
CUDA_PATH ?= /usr/local/cuda

# Flags for optimized compilation
CCFLAGS := -O3 -march=native -I$(CUDA_PATH)/include
# Enhanced CUDA flags targeting RTX 5090 (compute capability 9.0)
NVCCFLAGS := -O3 \
    -gencode=arch=compute_90,code=sm_90 \
    -gencode=arch=compute_89,code=sm_89 \
    -gencode=arch=compute_86,code=sm_86 \
    --use_fast_math \
    --threads 0 \
    --gpu-architecture=sm_90 \
    -Xptxas=-v,-O3 \
    -Xcompiler=-O3,-march=native \
    --default-stream per-thread \
    --relocatable-device-code=true \
    --maxrregcount=64 \
    --ptxas-options=-v,-O3

LDFLAGS := -L$(CUDA_PATH)/lib64 -lcudart -pthread

CPU_SRC := RCKangaroo.cpp GpuKang.cpp Ec.cpp utils.cpp
GPU_SRC := RCGpuCore.cu

CPP_OBJECTS := $(CPU_SRC:.cpp=.o)
CU_OBJECTS := $(GPU_SRC:.cu=.o)

TARGET := rckangaroo

all: $(TARGET)

$(TARGET): $(CPP_OBJECTS) $(CU_OBJECTS)
	$(CC) $(CCFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CCFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(CPP_OBJECTS) $(CU_OBJECTS) $(TARGET)
