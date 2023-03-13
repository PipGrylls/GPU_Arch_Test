#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>


__global__ void child_launch(int i, curandState_t localstate, int p_idx, int *child_out);


__global__ void init_gpurand(unsigned long long seed, int N, curandState *state);