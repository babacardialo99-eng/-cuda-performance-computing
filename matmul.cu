#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 1024

 // =======================================================
 // GPU KERNEL: 
 // Each GPU threads computes 1 elements of output matric C. 
 // ========================================================

 __global__ void matmul_gpu(int*a, int*b, int*c, int n) {
    
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
     int sum = 0;
     for (int k = 0; k < n; k++) {
     sum += a[row * n + k] * b[k * n + col];
    }
     // store the output matrix into output matrix C.
     c[row * n + col] = sum;
} // close if 

  } // close the global
     // ===============
     // CPU Computation
     // ===============
     void matmul_cpu(int * a, int* b, int* c, int n) {
    
    // Loop through the matrix rows
    for (int row = 0; row < n; row ++) {
    for (int col = 0; col < n; col++)  {
    
    // init sum of matrix
    int sum = 0;
    
    // ===============================
    // perform matrix multiplication
    // ===============================
     for (int k = 0; k < n; k++) {
    
     sum += a[row * n + k] * b[k * n + col];
     } // k loop closing

    c[row * n + col] = sum; //For each row  for each col  reset sum  loop k  store result for that (row, col) pair  move on/
  
    } // column loop
 }  // row loop 
} // close void


int main  ( )  {
    
    int n = N; 

  // Compute the total number of bytes needed to 
  // store the entire matrix in memory
     size_t size = n * n * sizeof(int); 
 
 // ============================
 // 1- ALLOCATE CPU MEMORY
 // ============================   
 int *host_a = (int*)malloc(size);  // "int*": because malloc return memory address pointer not an integer. 
 int *host_b = (int*)malloc(size);
 
 int *host_c_cpu = (int*)malloc(size); // will store the CPU result
 int *host_c_gpu = (int*)malloc(size); // will store the GPU result

 // ============================
 // 2- FILL CPU ARRAY WITH DATA :::
 // ============================ 
 for (int i = 0; i < n * n; i++) { 
  host_a[i] = 1; // matrix A filled with 1's: 
  host_b[i] = 2; // matrix B filled with 2's: 
  }
      
 // ================================
 // 3- ALLOCATE GPU MEMORY(DEVICE)
 // ================================
 int *device_a, *device_b, *device_c; 
 cudaMalloc(&device_a, size); 
 cudaMalloc(&device_b, size);
 cudaMalloc(&device_c, size);


  // ================================
  // 4 - COPY DATA FROM  CPU --> GPU
  // ================================
  cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice);

  // ==========================
  // 5 - GPU timming using CUDA
  // ==========================
  cudaEvent_t  start;  // declare the start variable
  cudaEvent_t  stop;   // declare the stop variable

  // 1 - create GPU timing events
  cudaEventCreate(&start);  // we create a timing events for start 
  cudaEventCreate(&stop);   // we create a timing events for stop.


  printf("Running The matrix  multiplication (%dx%d)....", n, n); 
                                                        
  // 2- start the GPU timing recording
  cudaEventRecord(start); 

 
  dim3  threadsPerBlock(16, 16);

  // Compute number of blocks/threads
      dim3 blocks (
      (n + 15) / 16, 
      (n + 15) / 16 
       );
    
  //3- ========================
  // 6 - NOW LAUNCH THE KERNEL
  //   ========================
  matmul_gpu<<<blocks,  threadsPerBlock>>> (
   device_a,
   device_b, 
   device_c,
   n
   ); 
  
  //4-  stop GPU timer
  cudaEventRecord(stop);
   
 // ===========================================================================================
 // 7 - wait for GPU to finish computing.This forces CPU to stop and wait for the GPU to finish.  
 // ===========================================================================================
 cudaEventSynchronize(stop);


 // ===============================
 // 8 - COPY DATA FROM GPU --> CPU
 // ===============================
 cudaMemcpy(host_c_gpu, device_c, size, cudaMemcpyDeviceToHost);


 

// ==================================
// 9:  Compute the GPU's elapsed time
// ==================================

float Elapsed_Time = 0.0f; 
cudaEventElapsedTime(&Elapsed_Time, start, stop);
printf("GPU Time in ms: %.3f", Elapsed_Time);

// convert the time from ms --> Sec for better reading purpose
double Time_In_Sec = Elapsed_Time / 1000.0;

printf("GPU Time in Sec: %.3f", Time_In_Sec);



// =============================
// 10 - CPU timing for comparison
// =============================

// start the CPU Timer:
clock_t cpu_start = clock(); 

// 
matmul_cpu(host_a, host_b, host_c_cpu, n);

// Stop the CPU Timer:
clock_t cpu_end = clock(); 

// Compute the total CPU time for Execution: 
double Total_CPU_Time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;


printf("CPU Time: %.3f", Total_CPU_Time);



// ==========================
// 11- VERIFY the GPU's answer
// ==========================
  
 // 1- verify correctness:
 int expected = 2 * n;
 // if the first matrix computation equals expected value 
 // AND the second matrix computation equals expected value
 // ==> PASS, otherwise FAIL.
 int correct = (host_c_gpu[0] ==  expected &&
                host_c_gpu[n*n -1] == expected);
 printf("verification: %s (expected %d, got %d)",
         correct ? "PASS" : "FAIL",
         expected,
         host_c_gpu[0]
       );


// ==============================
// 12: FULL GPU Vs CPU Comparison
// ==============================

int mismatches = 0; 

for (int i = 0; i < n * n; i++)   {
 if (host_c_gpu [i] != host_c_cpu [i]) {
     mismatches++; 
   } // close if

 } // close for..
 
 if (mismatches == 0)  { //No Error Found yet
   printf("Full verification PASSED");  
  } else {
  printf("Full verification FAILED");
  } 
printf("========= SPEED COMPARISON ==============");
printf("CPU Time: %.3f", Total_CPU_Time);
printf("GPU Time in Sec: %.3f", Time_In_Sec);
printf("Speedup: %.1fx", (Total_CPU_Time / Time_In_Sec) );

// ========
// Clean UP  
// ========

// Free CPU memory 
free(host_a);
free(host_b);

free(host_c_cpu);
free(host_c_gpu);

// Free GPU Memory: 
cudaFree(device_a);
cudaFree(device_b); 
cudaFree(device_c);

// destroy the timing even created previously
cudaEventDestroy(start);
cudaEventDestroy(stop);

 return 0;

}  // main close
