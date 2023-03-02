#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "kernals.h"

__global__ void d_main(int *N_child);

