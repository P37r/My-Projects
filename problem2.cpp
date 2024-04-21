#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <chrono>

using namespace std::chrono;

__global__ void add(int N, const float *x, float *y){
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;  
  if (i < N-1 && i > 0){
    y[i] = -x[i+1] +2*x[i] - x[i-1];
  }

  if (i==0){
    y[i] = -x[i+1] +2*x[i] - x[i];
  }

  if (i==N-1){
    y[i] = -x[i] +2*x[i] - x[i-1];
  }
}

int main(void){

  int N = 1e6;
  float * x = new float[N];
  float * y = new float[N];

  for (int i = 0; i < N; ++i){
    x[i] = 1.f;
  }

  int size = N * sizeof(float);

  // allocate memory and copy to the GPU
  float * d_x;
  float * d_y;
  cudaMalloc((void **) &d_x, size);
  cudaMalloc((void **) &d_y, size);
  
  // copy memory over to the GPU
  cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

  // call the add function  
  int blockSize = 128;
  int numBlocks = (N + blockSize - 1) / blockSize;
                                                                              // TIMER
  double total_elapsed_time = 0;
  high_resolution_clock::time_point start = high_resolution_clock::now();
  add<<<numBlocks, blockSize>>>(N, d_x, d_y);
  high_resolution_clock::time_point end = high_resolution_clock::now();
  duration<double> elapsed = end - start;
  total_elapsed_time += elapsed.count() * 1000; // Convert to milliseconds

  std::cout << "Total Elapsed Time: " << total_elapsed_time << " ms\n";


  // copy memory back to the CPU
  cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; ++i){
    if(y[i] != 0) {
        std::cout << "ERROR: Non Zero!"
    }
  }

  return 0;
}


// PART 2 

// __global__ void partial_reduction((int N, const float *x, float *y)){
  
//   __shared__ float s_x[BLOCKSIZE+2];

//   const int i = blockDim.x * blockIdx.x + threadIdx.x;
//   const int tid = threadIdx.x;
  
//   // coalesced reads in
//   s_x[tid] = 0.f;
//   if (i < N){
//     s_x[tid] = x[i];
//   }

//   // number of "live" threads per block
//   int alive = blockDim.x;
  

//   __syncthreads(); 
//   y[i]= s_x[tid + 1] + s_x[tid] - s_x[tid-1];

// }
