#include "d_main.h"

__global__ void d_main(int *N_child){
    // get block rank and id
        // We are currently unintrested in using the 2nd dimension in threads and blocks
    int t_idx = threadIdx.x
    int b_idx = blockIdx.x*blockDim.x;
    int idx = t_idx+b_idx

    // check if this rank is due to spawn a child process
    // ATM all ranks spawn 5 children
    for (int i; i<N_child; i++){
        <<<1,1>>>child_launch(i);
    }
    // wait for all children
    // free memory
    // exit
}