#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>


__global__ void dev_main(int *N_child, curandState_t *d_state, int child_out[]);