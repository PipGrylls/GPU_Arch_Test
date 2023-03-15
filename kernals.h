#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>


// Macro for error checking
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// Prototype for gpuAssert
void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

__global__ void child_launch(int i, curandState_t localstate, int p_idx, int *child_out);


__global__ void init_gpurand(unsigned long long seed, int N, curandState *state);