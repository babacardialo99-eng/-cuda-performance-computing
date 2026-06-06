
#include <stdio.h>
#include <stdlib.h>

// CUDA kernel - each GPU thread adds one pair of elements
__global__ void vectoradd(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1 << 20;
    size_t size = n * sizeof(int);

    // Allocate CPU memory
    int *host_a = (int*)malloc(size);
    int *host_b = (int*)malloc(size);
    int *host_c = (int*)malloc(size);

    // Fill input arrays
    for (int i = 0; i < n; i++) {
        host_a[i] = i;
        host_b[i] = i * 2;
    }

    // Allocate GPU memory
    int *device_a, *device_b, *device_c;
    cudaMalloc(&device_a, size);
    cudaMalloc(&device_b, size);
    cudaMalloc(&device_c, size);

    // Copy CPU -> GPU
    cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectoradd<<<blocks, threadsPerBlock>>>(device_a, device_b, device_c, n);

    // Copy GPU -> CPU
    cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost);

    // Verify
    printf("Verifying results:\n");
    for (int i = 0; i < 5; i++) {
        printf("host_a[%d]=%d + host_b[%d]=%d = host_c[%d]=%d\n",
               i, host_a[i], i, host_b[i], i, host_c[i]);
    }
    printf("...\n");
    int lastIndex = n - 1;
    int expectedResult = lastIndex + (lastIndex * 2);
    printf("host_c[%d] = %d (expected: %d)\n",
           lastIndex, host_c[lastIndex], expectedResult);

    // Free memory
    cudaFree(device_a); cudaFree(device_b); cudaFree(device_c);
    free(host_a); free(host_b); free(host_c);

    printf("\nVector addition complete on %d elements!\n", n);
    return 0;
}