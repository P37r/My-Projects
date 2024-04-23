// #include <stdio.h>
// #include <math.h>
// #include <cuda_runtime.h>


// #include <iostream>
// #include <vector>
// #include <chrono>

// using namespace std::chrono;

// #define BLOCKSIZE 128

// __global__ void partial_reduction(const int N, float *x_reduced, const float *x){
  
//   __shared__ float s_x[BLOCKSIZE];

//   const int i = blockDim.x * 2*blockIdx.x + threadIdx.x;
//   const int tid = threadIdx.x;
  
//   // coalesced reads in
//   s_x[tid] = 0.f;
//   if (i < N){
//     s_x[tid] = x[i] + x[i + blockDim.x];
//   }

//   // number of "live" threads per block
//   int alive = blockDim.x /2;

  
//   while (alive > 1){
//     __syncthreads(); 
//     alive /= 2; // update the number of live threads    
//     if (tid < alive){
//       s_x[tid] += s_x[tid + alive];
//     }
//   }

//   // write out once we're done reducing each block
//   if (tid==0){
//     x_reduced[blockIdx.x] = s_x[0];
//   }
// }
    




// int main(int argc, char * argv[]){

//   int N = 4096;
//   if (argc > 1){
//     N = atoi(argv[1]);
//   }

//   int blockSize = BLOCKSIZE;

//   // Next largest multiple of blockSize
//   int numBlocks = (N + blockSize - 1) / blockSize;

//   printf("N = %d, blockSize = %d, numBlocks = %d\n", N, blockSize, numBlocks);

//   float * x = new float[N];
//   float * x_reduced = new float[numBlocks];  

//   for (int i = 0; i < N; ++i){
//     x[i] = i + 1.f;
//   }

//   // allocate memory and copy to the GPU
//   float * d_x;
//   float * d_x_reduced;  
//   int size_x = N * sizeof(float);
//   int size_x_reduced = numBlocks * sizeof(float);
//   cudaMalloc((void **) &d_x, size_x);
//   cudaMalloc((void **) &d_x_reduced, size_x_reduced);
  
//   // copy memory over to the GPU
//   cudaMemcpy(d_x, x, size_x, cudaMemcpyHostToDevice);
//   cudaMemcpy(d_x_reduced, x_reduced, size_x_reduced, cudaMemcpyHostToDevice);

//   partial_reduction <<< numBlocks, blockSize >>> (N, d_x_reduced, d_x);

//   // copy memory back to the CPU
//   cudaMemcpy(x_reduced, d_x_reduced, size_x_reduced, cudaMemcpyDeviceToHost);

//   float sum_x = 0.f;
//   for (int i = 0; i < numBlocks; ++i){
//     sum_x += x_reduced[i];
//   }

//   float target = N * (N+1) / 2.f;
//   printf("error = %f\n", fabs(sum_x - target));

// #if 1
//   int num_trials = 10;
//   float time;
//   cudaEvent_t start, stop;
//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);
//   cudaEventRecord(start, 0);

//   for (int i = 0; i < num_trials; ++i){
//     partial_reduction <<< numBlocks, blockSize >>> (N, d_x_reduced, d_x);
//   }

//   cudaEventRecord(stop, 0);
//   cudaEventSynchronize(stop);
//   cudaEventElapsedTime(&time, start, stop);
  
//   printf("Time to run kernel 10x: %6.2f ms.\n", time);
  
// #endif

//   return 0;
// }


// ___________________________________________________________________________________________________


#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 128 


__global__ void partial_reduction_orig(const int N, float *x_reduced, const float *x){
  
  __shared__ float s_x[BLOCKSIZE];

  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int tid = threadIdx.x;
  
  // coalesced reads in
  s_x[tid] = 0.f;
  if (i < N){
    s_x[tid] = x[i];
  }

  // number of "live" threads per block
  int alive = blockDim.x;
  
  while (alive > 1){
    __syncthreads(); 
    alive /= 2; // update the number of live threads    
    if (tid < alive){
      s_x[tid] += s_x[tid + alive];
    }
  }

  // write out once we're done reducing each block
  if (tid==0){
    x_reduced[blockIdx.x] = s_x[0];
  }
}
__global__ void partial_reduction(const int N, float *x_reduced, const float *x){
  
  __shared__ float s_x[BLOCKSIZE];

  const int i = blockDim.x * 2*blockIdx.x + threadIdx.x;
  const int tid = threadIdx.x;
  
  // coalesced reads in
  s_x[tid] = 0.f;
  if (i < N){
    s_x[tid] = x[i] + x[i + blockDim.x];
  }

  // number of "live" threads per block
  int alive = blockDim.x;
  
  while (alive > 1){
    __syncthreads(); 
    alive /= 2; // update the number of live threads    
    if (tid < alive){
      s_x[tid] += s_x[tid + alive];
    }
  }

  // write out once we're done reducing each block
  if (tid==0){
    x_reduced[blockIdx.x] = s_x[0];
  }
}

__global__ void partial_reduction2(const int N, float *x_reduced, const float *x){
  
  __shared__ float s_x[BLOCKSIZE];

  const int i = blockDim.x * 2*blockIdx.x + threadIdx.x;
  const int tid = threadIdx.x;
  
  // coalesced reads in
  s_x[tid] = 0.f;
  if (i < N){
    s_x[tid] = x[i] + x[i + blockDim.x];
  }

  // number of "live" threads per block
  int alive = blockDim.x;
  int s = 1;
  while (alive > 1){
    __syncthreads(); 
    if (tid % (2*s) == 0){
      s_x[tid] += s_x[tid + s];
    }
    s *= 2;
    alive /= 2; // update the number of live threads    
  }


  // write out once we're done reducing each block
  if (tid==0){
    x_reduced[blockIdx.x] = s_x[0];
  }
}



    
int main(int argc, char * argv[]){

  int N = pow(2,22);
  if (argc > 1){
    N = atoi(argv[1]);
  }

  int blockSize = BLOCKSIZE;

  // Next largest multiple of blockSize
  int numBlocks = (N + blockSize - 1) / blockSize;

  printf("N = %d, blockSize = %d, numBlocks = %d\n", N, blockSize, numBlocks);

  float * x = new float[N];
  float * x_reduced = new float[numBlocks];  

  for (int i = 0; i < N; ++i){
    x[i] = 1.f;
  }

                                                                                      // Version 1
    // allocate memory and copy to the GPU
    float * d_x;
    float * d_x_reduced;  
    int size_x = N * sizeof(float);
    int size_x_reduced = numBlocks * sizeof(float);
    cudaMalloc((void **) &d_x, size_x);
    cudaMalloc((void **) &d_x_reduced, size_x_reduced);
    
    // copy memory over to the GPU
    cudaMemcpy(d_x, x, size_x, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_reduced, x_reduced, size_x_reduced, cudaMemcpyHostToDevice);

    partial_reduction <<< numBlocks, blockSize >>> (N, d_x_reduced, d_x);

    // copy memory back to the CPU
    cudaMemcpy(x_reduced, d_x_reduced, size_x_reduced, cudaMemcpyDeviceToHost);

    float sum_x = 0.f;
    for (int i = 0; i < numBlocks; ++i){
        sum_x += x_reduced[i];
    }

    //  float target = N * (N+1) / 2.f;
    float target = N;
    printf("error = %f\n", fabs(sum_x - target));

    #if 1
    int num_trials = 10;
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int i = 0; i < num_trials; ++i){
        partial_reduction <<< numBlocks, blockSize >>> (N, d_x_reduced, d_x);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    printf("Time to run kernel 10x: %6.2f ms.\n", time);
    
    #endif

    // next version

    printf("SECOND VERSION = %f\n");

                                                                                        // Version 2
    // allocate memory and copy to the GPU
    float * x_reduced2 = new float[numBlocks];  

    float * d_x2;
    float * d_x_reduced2;  
    cudaMalloc((void **) &d_x2, size_x);
    cudaMalloc((void **) &d_x_reduced2, size_x_reduced);
    
    // copy memory over to the GPU
    cudaMemcpy(d_x2, x, size_x, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_reduced2, x_reduced2, size_x_reduced, cudaMemcpyHostToDevice);

    partial_reduction2 <<< numBlocks, blockSize >>> (N, d_x_reduced2, d_x2);

    // copy memory back to the CPU
    cudaMemcpy(x_reduced2, d_x_reduced2, size_x_reduced, cudaMemcpyDeviceToHost);

    float sum_x2 = 0.f;
    for (int i = 0; i < numBlocks; ++i){
        sum_x2 += x_reduced2[i];
    }

    //  float target = N * (N+1) / 2.f;
    printf("error = %f\n", fabs(sum_x2 - target));
    #if 1
    int num_trials2 = 10;
    float time2;
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);

    for (int i = 0; i < num_trials2; ++i){
        partial_reduction2 <<< numBlocks, blockSize >>> (N, d_x_reduced2, d_x2);
    }

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&time2, start2, stop2);
    
    printf("Time to run kernel 10x: %6.2f ms.\n", time2);
    
    #endif


    
    printf("ORIGINAL VERSION = %f\n");


                                                                                        // Version 3
    // allocate memory and copy to the GPU
    float * x_reduced3 = new float[numBlocks];  


    float * d_x3;
    float * d_x_reduced3;  
    cudaMalloc((void **) &d_x3, size_x);
    cudaMalloc((void **) &d_x_reduced3, size_x_reduced);
   
    // copy memory over to the GPU
    cudaMemcpy(d_x3, x, size_x, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_reduced3, x_reduced3, size_x_reduced, cudaMemcpyHostToDevice);


    partial_reduction_orig <<< numBlocks, blockSize >>> (N, d_x_reduced3, d_x3);


    // copy memory back to the CPU
    cudaMemcpy(x_reduced3, d_x_reduced3, size_x_reduced, cudaMemcpyDeviceToHost);


    float sum_x3 = 0.f;
    for (int i = 0; i < numBlocks; ++i){
        sum_x3 += x_reduced3[i];
    }


    //  float target = N * (N+1) / 3.f;
    printf("error = %f\n", fabs(sum_x3 - target));
    #if 1
    int num_trials3 = 10;
    float time3;
    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    cudaEventRecord(start3, 0);


    for (int i = 0; i < num_trials3; ++i){
        partial_reduction_orig <<< numBlocks, blockSize >>> (N, d_x_reduced3, d_x3);
    }


    cudaEventRecord(stop3, 0);
    cudaEventSynchronize(stop3);
    cudaEventElapsedTime(&time3, start3, stop3);
   
    printf("Time to run kernel 10x: %6.3f ms.\n", time3);
   
    #endif


  return 0;
}
