#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

__global__ void child_launch();

__global__ void child_collect();


// For testing generation of random numbers on each thread
__global__ void populate_random(int length, float *rnd_array, curandState *state);