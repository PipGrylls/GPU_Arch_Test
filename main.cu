// -*- mode: C -*-

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "d_main.h"
#include "kernals.h"


int main () {

    unsigned long rngseed = 2894203475;  // RNG seed (fixed for development/testing)
    //unsigned long rngseed = (long)time(NULL);
    // Initialise host RNG
    init_genrand(rngseed);

    // Get GPU info
    cudaDeviceProp prop;
    int cudaDevice, \
        devGlobalMem, devSharedMemPerBlock, 
        devThreadsPerblock, devMultiProc,
        devConcurrentKernals;
    int maxThreads[3], maxGrid[3];
    cudaGetDevice(&cudaDevice);
    printf(cudaDevice);
    cudaSetDevice(cudaDevice) ;
    // Really we should query property by property as this has some excess overhead,
    // while we dont know what properties we need this is preferable.
    cudaGetDeviceProperties(&prop, cudaDevice)
    devConcurrentKernals = prop.concurrantKernals;
    if (devConcurrentKernals == 0) {
        printf("Error, this code requires concurrant kernal launches CC>5")
        exit()
    }

    // Get Threads and blocks
    devGlobalMem = prop.totalGlobalMem;
    devSharedMemPerBlock = prop.sharedMemPerBlock;
    devThreadsPerblock = prop.maxThreadsPerBlock;
    devMultiProc = prop.multiProcessorCount;


       
    int N_bl = 5; //we are going to span 5 blocks 
    int N_th = 5; //with 5 threads
    int N_child = 5; // which all launch 5 children

    // Initilise RNG on GPU
    gpuErrchk (cudaMalloc((void **)&d_state, ngrids*sizeof(curandState)) );
    unsigned long long gpuseed = (unsigned long long)rngseed;
    init_gpurand<<<N_bl,N_th>>>(gpuseed, ngrids, d_state);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    // Put some data in memory
    int *dev_N_child;
    cudaMalloc( (void**)&dev_N_child, sizeof(N_child))

    // Launch d_main
    cudaMemcpy(dev_N_child, N_child, cudaMemcpyDeviceToHost)
    <<<N_bl, N_th>>>d_main();



    
    // Allocated memory to return info from streams
    int host_collect[2];
    int *dev_collect;
    cudaMalloc((void(**)&dev_collect))
    
    //Setup Collection streams
    // One stream to start with more later
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // LOOP
    for (int i; i<N_bl*N_th*N_child; i++){
        // Launch collection kernals
         <<<1,1, stream>>>child_collect(&dev_collect, 2*sizeof(int));
        // Launch Memcpy
        cudaMemCpy(dev_collect, host_collet, 2*sizeof(int));
        // This should print the rank then how long it waited in seconds
        printf(host_collect);
    }
    //ENDLOOP

    // Free the memory
    free(host_collect);
    cudaFree(dev_collect);

    // Output
    printf("Done!");

    return 0;

}