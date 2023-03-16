// -*- mode: C -*-

#include "d_main.h"
#include "kernals.h"


__global__ void dev_main(int *N_child, curandState_t *d_state, int child_out[]){
    // get block rank and id
    // We are currently unintrested in using the 2nd dimension in threads and blocks
    int t_idx = threadIdx.x;
    int b_idx = blockIdx.x*blockDim.x;
    int idx = t_idx+b_idx;
    curandState *child_state;
    cudaMalloc((void**)&child_state, (*N_child)*sizeof(curandState));
    init_gpurand<<< *N_child, 1>>>(1, (*N_child), child_state);
    
    child_out[0] = idx;
    // check if this rank is due to spawn a child process
    // ATM all ranks spawn 5 children
    //for (int i=0; i<(*N_child); i++){
    //    child_launch<<<1,1,0,cudaStreamFireAndForget>>>(i, child_state[i], idx, child_out);
    //}
    // wait for all children
    
    // free memory
    cudaFree(child_state);
    // exit
}