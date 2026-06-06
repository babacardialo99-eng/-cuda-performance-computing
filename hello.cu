#include <stdio.h>

// CUDA kernel - each GPU thread prints its index
__global__ void helloFromGPU() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    printf("Hello from CPU!\n");

    // Launch kernel: 1 block of 5 threads
    helloFromGPU<<<1, 5>>>();

    // Wait for the GPU to finish before exiting
    cudaDeviceSynchronize();

    return 0;
}
