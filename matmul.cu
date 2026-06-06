#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024

// ======================================================
// GPU KERNEL
// Each GPU thread computes ONE element of output matrix C
// ======================================================
__global__ void matmul_gpu(int *a, int *b, int *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

// =============================================
// CPU VERSION (for speed comparison)
// =============================================
void matmul_cpu(int *a, int *b, int *c, int n) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            int sum = 0;
            for (int k = 0; k < n; k++) {
                sum += a[row * n + k] * b[k * n + col];
            }
            c[row * n + col] = sum;
        }
    }
}

int main() {
    int n = N;
    size_t size = n * n * sizeof(int);

    // Allocate CPU memory
    int *host_a = (int*)malloc(size);
    int *host_b = (int*)malloc(size);
    int *host_c_gpu = (int*)malloc(size);
    int *host_c_cpu = (int*)malloc(size);

    // Fill matrices BEFORE copying to GPU
    for (int i = 0; i < n * n; i++) {
        host_a[i] = 1;
        host_b[i] = 2;
    }

    // CPU timing
    printf("Running CPU matrix multiplication (%dx%d)...\n", n, n);
    clock_t cpu_start = clock();
    matmul_cpu(host_a, host_b, host_c_cpu, n);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
    printf("CPU time: %.3f seconds\n\n", cpu_time);

    // Allocate GPU memory
    int *device_a, *device_b, *device_c;
    cudaMalloc(&device_a, size);
    cudaMalloc(&device_b, size);
    cudaMalloc(&device_c, size);

    // Copy CPU -> GPU
    cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice);

    // Configure CUDA grid: 16x16 = 256 threads per block
    dim3 threadsPerBlock(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);

    // GPU timing using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Running GPU matrix multiplication (%dx%d)...\n", n, n);
    cudaEventRecord(start);
    matmul_gpu<<<blocks, threadsPerBlock>>>(device_a, device_b, device_c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    double gpu_time = gpu_time_ms / 1000.0;
    printf("GPU time: %.3f seconds\n\n", gpu_time);

    // Copy GPU -> CPU
    cudaMemcpy(host_c_gpu, device_c, size, cudaMemcpyDeviceToHost);

    // Verify
    int expected = 2 * n;
    int correct = (host_c_gpu[0] == expected && host_c_gpu[n*n - 1] == expected);
    printf("Verification: %s (expected %d, got %d)\n",
           correct ? "PASS" : "FAIL", expected, host_c_gpu[0]);

    // Speed comparison
    printf("\n======== Speed comparison ========\n");
    printf("CPU: %.3f sec\n", cpu_time);
    printf("GPU: %.3f sec\n", gpu_time);
    printf("Speedup: %.1fx\n", cpu_time / gpu_time);

    // Cleanup
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    free(host_a);
    free(host_b);
    free(host_c_gpu);
    free(host_c_cpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
