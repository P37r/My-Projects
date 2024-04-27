#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>


#include <iostream>
#include <vector>
#include <chrono>

using namespace std::chrono;


__global__ void add(int N, const float *x, float *y, int blocksize){
  
//   int i = blockIdx.x * blockDim.x + threadIdx.x;  
//   if (i < N-1 && i > 0){
//     y[i] = -x[i+1] +2*x[i] - x[i-1];
//   }

//   if (i==0){
//     y[i] = -x[i+1] +2*x[i] - x[i];
//   }

//   if (i==N-1){
//     y[i] = -x[i] +2*x[i] - x[i-1];
//   }


//   if (i>0 && i <N-1){
//     y[i] = -x[i+1] +2*x[i] - x[i-1];



   extern __shared__ float s_x[]; // shared memory for x


  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int tid = threadIdx.x;
  
  // coalesced reads in
  s_x[tid] = 0.f;
  
  
  if (i < N){
    if (tid <blockDim.x + 2) {
        if(i == 0) {
            s_x[tid] = x[0];
        }
        else if(i == N-1) {
            s_x[tid] = x[N-1];
        }
        else {
            s_x[tid] = x[i-1];
        }
        
    }


    // if (tid == blockDim.x + 2) {
    //     s_x[tid] = x[i];
    // }
  }

  // number of "live" threads per block
  
  __syncthreads(); 
                                             // I add +1 to the index so it adjusts for the shared memory, which has been shifted 1 unit
  y[i]= -s_x[tid + 1+1] + 2* s_x[tid+1] - s_x[tid-1+1];
}





__global__ void add2(int N, const float *x, float *y, int blocksize){
  
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


  if (i>0 && i <N-1){
    y[i] = -x[i+1] +2*x[i] - x[i-1];
}
}

void printArray(const float* y, int N) {
    std::cout << "[ ";
    for (int i = 0; i < 100; ++i) {
        std::cout << y[i] << " ";
    }
    std::cout << " ]" << std::endl;
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

                                                            //return here  

    int blockSize = 65536;
    
  
  int numBlocks = (N + blockSize - 1) / blockSize;
                                                                              // TIMER
  
   double total_elapsed_time2 = 0;

  for (int trials = 0; trials < 10; ++trials){
    high_resolution_clock::time_point start2 = high_resolution_clock::now();
  add2<<<numBlocks, blockSize>>>(N, d_x, d_y,blockSize);
  high_resolution_clock::time_point end2 = high_resolution_clock::now();
  duration<double> elapsed2 = end2 - start2;
  total_elapsed_time2 += elapsed2.count() * 1000; // Convert to milliseconds


  }
  std::cout << "BLOCK SIZE = " << blockSize << "\n";
  std::cout << "Total Elapsed Time: " << total_elapsed_time2 << " ms\n";
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
                                                                                // Verify check
    for (int i = 0; i < N; ++i){
    if(y[i] != 0) {
        std::cout << "ERROR: Non Zero!";
    }
    }
    std::cout << "\n";
    
   

                                                                                    // SECOND VERSION   
  
  double total_elapsed_time = 0;

  for (int trials = 0; trials < 10; ++trials){
    high_resolution_clock::time_point start = high_resolution_clock::now();
  add<<<numBlocks, blockSize>>>(N, d_x, d_y,blockSize);
  high_resolution_clock::time_point end = high_resolution_clock::now();
  duration<double> elapsed = end - start;
  total_elapsed_time += elapsed.count() * 1000; // Convert to milliseconds


  }
  std::cout << "BLOCK SIZE = " << blockSize << "\n";
  std::cout << "Total Elapsed Time: " << total_elapsed_time << " ms\n";


  // copy memory back to the CPU

   
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
