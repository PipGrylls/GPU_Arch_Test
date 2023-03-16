// -*- mode: C -*-

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

extern "C" {
    #include "mt19937ar.h"
}

#include "kernals.h"
#include "d_main.h"


int main () {

    printf("Hello World\n");

    unsigned long rngseed = 2894203475;  // RNG seed (fixed for development/testing)
    //unsigned long rngseed = (long)time(NULL);
    // Initialise host RNG
    init_genrand(rngseed);

    // Get GPU info
    cudaDeviceProp prop;
    int cudaDevice, \
        devGlobalMem, devSharedMemPerBlock, \
        devThreadsPerblock, devMultiProc;
    int maxThreads[3], maxGrid[3];
    gpuErrchk(cudaGetDevice(&cudaDevice));
    printf("%i\n", cudaDevice);
    gpuErrchk(cudaSetDevice(cudaDevice));
    // Really we should query property by property as this has some excess overhead,
    // while we dont know what properties we need this is preferable.
    cudaGetDeviceProperties(&prop, cudaDevice);
    int devConcurrentKernels = prop.concurrentKernels;
    if (devConcurrentKernels == 0) {
        printf("Error, this code requires concurrant kernal launches CC>5");
        exit(1);
    }
    for(int i=0; i<3; i++){
        maxThreads[i]=prop.maxThreadsDim[i];
        maxGrid[i]=prop.maxGridSize[i];
    }
    printf("Threads: %i,%i,%i\n", maxThreads[0], maxThreads[1], maxThreads[2]);
    printf("Grids: %i,%i,%i\n", maxGrid[0], maxGrid[1], maxGrid[2]);

    // Get Threads and blocks
    devGlobalMem = prop.totalGlobalMem;
    devSharedMemPerBlock = prop.sharedMemPerBlock;
    devThreadsPerblock = prop.maxThreadsPerBlock;
    devMultiProc = prop.multiProcessorCount;
    printf("Global Mem: %i\n", devGlobalMem);
    printf("Shared Mem: %i\n", devSharedMemPerBlock);
    printf("Threads Per Block: %i\n", devThreadsPerblock);
    printf("MultiProc: %i\n", devMultiProc);
    fflush(stdout);
       
    int N_bl = 5; //we are going to span 5 blocks
    int N_th = 5; //with 5 threads
    int *N_child; //= 5; // which all launch 5 children
    gpuErrchk(cudaMallocHost( (void **)&N_child, sizeof(int) ));
    *N_child = 5;
    // Initilise RNG on GPU
    curandState *d_state;
    gpuErrchk(cudaMalloc( (void **)&d_state, N_bl*N_th*sizeof(curandState) ));
    unsigned long long gpuseed = (unsigned long long)rngseed;

    // create global memory array for child output
    int *host_child_out;
    gpuErrchk(cudaMallocHost( (void **)&host_child_out, (*N_child)*N_th*N_bl*sizeof(int)));

    // Create varable to instuct dmain on how to launch children

    int *dev_N_child[N_bl];
    for (int i=0;i<N_bl;i++){
        gpuErrchk(cudaMalloc( (void**)&dev_N_child[i], sizeof((*N_child)) ));
    }

    // dynamically sized arrays
    int *dev_child_out[N_bl];
    cudaStream_t streams[N_bl];

    for (int i=0;i<N_bl;i++){
        gpuErrchk(cudaStreamCreate(&streams[i]));
        gpuErrchk(cudaMalloc( (void**)&dev_child_out[i], (*N_child)*N_th*sizeof(int) ));
    }
    for (int i=0;i<N_bl;i++){
        // init the RNG
        init_gpurand<<<1,N_th,0,streams[i]>>>(gpuseed, N_bl, d_state);
    }
    for (int i=0;i<N_bl;i++){
        // Launch d_main
        gpuErrchk(cudaMemcpyAsync(dev_N_child[i], N_child, sizeof(int), cudaMemcpyDeviceToHost,  streams[i]));
    }
    for (int i=0;i<N_bl;i++){
        dev_main<<<1,N_th,0,streams[i]>>>(N_child, d_state, dev_child_out[i]);
    }


    for (int i=0;i<N_bl;i++){
        gpuErrchk(cudaMemcpyAsync(host_child_out+i*(*N_child)*N_th, dev_child_out[i], (*N_child)*N_th*sizeof(int), cudaMemcpyDeviceToHost, streams[i]));
    }

    //Synchronise 
    for (int i=0;i<N_bl;i++){
        gpuErrchk(cudaStreamSynchronize(streams[i]));
    } 

    //Print
    fflush(stdout);
    printf("Host Out:");
    for (int i=0; i<(*N_child)*N_th*N_bl;i++){
        printf("%i ", host_child_out[i]);
    }
    printf("\n");
    fflush(stdout);


    // Free the memory
    for (int i=0;i<N_bl;i++){
        gpuErrchk(cudaFree(dev_child_out[i]));
        gpuErrchk(cudaFree(dev_N_child[i]));
        gpuErrchk(cudaStreamDestroy(streams[i]));
    }
    gpuErrchk(cudaFreeHost(host_child_out));


    // Finish
    printf("Done!\n");
    fflush(stdout);

    return 0;

}