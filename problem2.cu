
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>


#include <iostream>
#include <vector>
#include <chrono>

using namespace std::chrono;


__global__ void add(int N, const float *x, float *y, int blocksize){

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

  // number of "live" threads per block
  
  __syncthreads(); 
  
                                             // I add +1 to the index so it adjusts for the shared memory, which has been shifted 1 unit
  if (i<N){

  y[i]= -s_x[tid + 1+1] + 2* s_x[tid+1] - s_x[tid-1+1];

  }

}
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
  

    #if 1

    float time;                                                                          
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    for (int trials = 0; trials < 10; ++trials){
    add2<<<numBlocks, blockSize>>>(N, d_x, d_y,blockSize);

    }
  
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
   
    printf("Time to run kernel 10x: %6.3f ms.\n", time);
    #endif
// 
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
                                                                                // Verify check
    for (int i = 0; i < N; ++i){
    if(y[i] != 0) {
        std::cout << "ERROR: Non Zero!";
    }
    }
    std::cout << "\n";
    
   
                                 // SECOND VERSION   
#if 1

float time2;                                                                          
cudaEvent_t start2, stop2;
cudaEventCreate(&start2);
cudaEventCreate(&stop2);
cudaEventRecord(start2, 0);



for (int trials = 0; trials < 10; ++trials){
  add<<<numBlocks, blockSize>>>(N, d_x, d_y,blockSize);
  }    
  
    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&time2, start2, stop2);
   
    printf("Time to run kernel 10x: %6.3f ms.\n", time2);

  // copy memory back to the CPU
#endif



return 0;
}


