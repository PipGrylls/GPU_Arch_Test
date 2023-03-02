#import "kernals.h"

__global__ void child_launch(int *i){
    // Launch
    // get id
    int t_idx = threadIdx.x
    int b_idx = blockIdx.x*blockDim.x;
    int idx = t_idx+b_idx

    // get a random number
    int rnd;

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


__global__ void populate_random(int length, float *rnd_array, curandStatePhilox4_32_10_t *state){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < length){
        // 4 random numbers
        float4 rnd = curand_uniform4(&state[idx]);

        // use one of these
        rnd_array[idx] = rnd.z;
    }      

    return;
}