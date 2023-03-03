#import "kernals.h"

__global__ void child_launch(int *i){
    // Launch
    // get id
    int t_idx = threadIdx.x
    int b_idx = blockIdx.x*blockDim.x;
    int idx = t_idx+b_idx

    // get a random number between 0 and 1
    float rnd_float = curand_uniform(&localstate)
    int rnd = int(rnd_float*10) //convert this to an interger number of seconds 0 to 10
    int child_out[3] = [idx, i, rnd];
    // transfer id to an array in memory:
       // Shared
       // Device
    // wait 
    // launch event()
    // exit
}

__global__ void child_collect(){
    // Launch
    // find event
    // collect child memory address
    // transfer result to known address
    // exit
}


// Kernel to initialise RNG on the GPU. Used the cuRAND device API with one
// RNG sequence per CUDA thread.
__global__ void init_gpurand(unsigned long long seed, int ngrids, curandState *state){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx<ngrids){

        unsigned long long seq = (unsigned long long)idx;
        
        // Seperate subsequence for each thread
        curand_init(seed, seq, 0ull, &state[idx]);
    
        // Different seed for each thread (faster but risky)
        //curand_init(seed+23498*idx, 0ull, 0ull, &state[idx]);
    }
}