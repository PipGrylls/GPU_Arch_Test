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
    cudaGetDevice(&cudaDevice);
    printf("%c", (char)(cudaDevice));
    cudaSetDevice(cudaDevice) ;
    // Really we should query property by property as this has some excess overhead,
    // while we dont know what properties we need this is preferable.
    cudaGetDeviceProperties(&prop, cudaDevice);
    int devConcurrentKernels = prop.concurrentKernels;
    if (devConcurrentKernels == 0) {
        printf("Error, this code requires concurrant kernal launches CC>5");
        exit(1);
    }
    printf("%c", maxThreads);
    printf("%c", maxGrid);

    // Get Threads and blocks
    devGlobalMem = prop.totalGlobalMem;
    devSharedMemPerBlock = prop.sharedMemPerBlock;
    devThreadsPerblock = prop.maxThreadsPerBlock;
    devMultiProc = prop.multiProcessorCount;
    printf("%c", devGlobalMem);
    printf("%c", devSharedMemPerBlock);
    printf("%c", devThreadsPerblock);
    printf("%c", devMultiProc);
       
    int N_bl = 5; //we are going to span 5 blocks
    int N_th = 5; //with 5 threads
    int *N_child; //= 5; // which all launch 5 children
    cudaMallocHost( (void **)&N_child, sizeof(int) );
    *N_child = 5;
    // Initilise RNG on GPU
    curandState *d_state;
    cudaMalloc( (void **)&d_state, N_bl*N_th*sizeof(curandState) );
    unsigned long long gpuseed = (unsigned long long)rngseed;

    // create global memory array for child output
    int *host_child_out;
    cudaMallocHost( (void **)&host_child_out, (*N_child)*N_th*N_bl*sizeof(int));

    // Create varable to instuct dmain on how to launch children

    int *dev_N_child[N_bl];
    for (int i=0;i<N_bl;i++){
        cudaMalloc( (void**)&dev_N_child[i], sizeof((*N_child)) );
    }

    // dynamically sized arrays
    int **dev_child_out;
    cudaStream_t streams[N_bl];

    for (int i=0;i<N_bl;i++){
        cudaStreamCreate(&streams[i]);
        cudaMalloc( (void**)&dev_child_out[i], (*N_child)*N_th*sizeof(int) );
    }
    for (int i=0;i<N_bl;i++){
        // init the RNG
        init_gpurand<<<1,N_th,0,streams[i]>>>(gpuseed, N_bl, d_state);
    }
    for (int i=0;i<N_bl;i++){
        // Launch d_main
        cudaMemcpyAsync(dev_N_child, N_child, sizeof(int), cudaMemcpyDeviceToHost,  streams[i]);
    }
    for (int i=0;i<N_bl;i++){
        dev_main<<<1,N_th,0,streams[i]>>>(N_child, d_state, dev_child_out[i]);
    }

    for (int i=0;i<N_bl;i++){
        cudaMemcpyAsync(host_child_out+i*(*N_child)*N_th, dev_child_out, (*N_child)*N_th*sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
    }

    //Synchronise

    for (int i=0;i<N_bl;i++){
        cudaFree(dev_child_out);
        cudaStreamDestroy(streams[i]);
    }


    printf("%s", host_child_out);
    // Free the memory
    cudaFreeHost(host_child_out);
    for (int i=0;i<N_bl;i++){
        cudaFree(dev_N_child[i]);
    }

    // Output
    printf("Done!");

    return 0;

}