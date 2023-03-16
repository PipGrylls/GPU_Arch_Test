// -*- mode: C -*-

#include "kernals.h"
#include <stdio.h>


cudaError_t err;  // cudaError_t is a type defined in cuda.h

// Boilerplate error checking code borrowed from stackoverflow
void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}


__global__ void child_launch(int i, curandState_t localstate, int p_idx, int *child_out){
    // Launch
    // get id
    int t_idx = threadIdx.x;
    int b_idx = blockIdx.x*blockDim.x;
    int idx = t_idx+b_idx;
    //p_idx is the index of the spawning dmain
    
    // get a random number between 0 and 1
    float rnd_float = curand_uniform(&localstate);
    int rnd = int(rnd_float*10.); //convert this to an interger number of seconds 0 to 10


    // transfer id to an array in memory
    // Shared
    // Device
    child_out[p_idx+idx] = idx;
    child_out[p_idx+idx+1] = i;
    child_out[p_idx+idx+2] = rnd;
    
    // exit
}


// Kernel to initialise RNG on the GPU. Used the cuRAND device API with one
// RNG sequence per CUDA thread.
__global__ void init_gpurand(unsigned long long seed, int N, curandState *state){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx<N){

        unsigned long long seq = (unsigned long long)idx;
        
        // Seperate subsequence for each thread
        curand_init(seed, seq, 0ull, &state[idx]);
    
        // Different seed for each thread (faster but risky)
        //curand_init(seed+23498*idx, 0ull, 0ull, &state[idx]);
    }
}