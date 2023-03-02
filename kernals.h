#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

__global__ void child_launch();

__global__ void child_collect();
