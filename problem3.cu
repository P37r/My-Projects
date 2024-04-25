#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <chrono>

using namespace std::chrono;

#define BLOCKSIZE 32

__global__ void matmul1(int N, const float *A, const float *B, float *C) {

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < N && j < N) {
    float val = 0.f;
    for (int k = 0; k < N; ++k) {
      val += A[k + i * N] * B[j + k * N];
    }
    C[j + i * N] += val;    
  }
}

__global__ void matmul2(int N, const float *A, const float *B, float *C) {

  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;

  //  const int i = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  //  const int j = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);  

  if (i < N && j < N) {
    float val = 0.f;
    for (int k = 0; k < N; ++k) {
      val += A[k + i * N] * B[j + k * N];
    }
    C[j + i * N] += val;    
  }
}

__global__ void matmul3(int N, const float *A, const float *B, float *C) {
  
  // the output block that we want to compute in this threadblock
  const int cRow = blockIdx.x;
  const int cCol = blockIdx.y;

  // allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ float s_A[BLOCKSIZE * BLOCKSIZE];
  __shared__ float s_B[BLOCKSIZE * BLOCKSIZE];

  // the inner row & col that we're accessing in this thread
  const int threadCol = threadIdx.x % BLOCKSIZE;
  const int threadRow = threadIdx.x / BLOCKSIZE;

  // advance pointers to the starting positions
  A += cRow * BLOCKSIZE * N;                    // row=cRow, col=0
  B += cCol * BLOCKSIZE;                        // row=0, col=cCol
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  for (int blkIdx = 0; blkIdx < N; blkIdx += BLOCKSIZE) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    s_A[threadRow * BLOCKSIZE + threadCol] = A[threadRow * N + threadCol];
    s_B[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();

    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += s_A[threadRow * BLOCKSIZE + dotIdx] * s_B[dotIdx * BLOCKSIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  C[threadRow * N + threadCol] += tmp;
}
                                                                                        // Method 4

// // allocate thread-local cache for results in registerfile
// float threadResults[TM] = {0.0};

// // outer loop over block tiles
// for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
//   // populate the SMEM caches (same as before)
//   As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
//   Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
//   __syncthreads();

//   // advance blocktile for outer loop
//   A += BK;
//   B += BK * N;

//   // calculate per-thread results
//   for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
//     // we make the dotproduct loop the outside loop, which facilitates
//     // reuse of the Bs entry, which we can cache in a tmp var.
//     float Btmp = Bs[dotIdx * BN + threadCol];
//     for (uint resIdx = 0; resIdx < TM; ++resIdx) {
//       threadResults[resIdx] +=
//           As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
//     }
//   }
//   __syncthreads();
// }    
                                                                                        // Method 5

// for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
//   As[(innerRowA + loadOffset) * BK + innerColA] =
//       A[(innerRowA + loadOffset) * K + innerColA];
// }
// for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
//   Bs[(innerRowB + loadOffset) * BN + innerColB] =
//       B[(innerRowB + loadOffset) * N + innerColB];
// }
// __syncthreads();

int main(int argc, char * argv[]){


for (int version = 1; version <=5; ++version){
    // print out the version 'version'

    // print out the version number
    printf("Version %d", version);
    printf("\n");

    
    for (int power = 9; power <= 12; ++power){
    int N = pow(2,power);
    // print out N
    printf("N = %d", N);
    printf("\n");

    
  if (argc > 1){
    N = atoi(argv[1]);
  }

  float * A = new float[N * N];
  float * B = new float[N * N];
  float * C = new float[N * N];

  for (int i = 0; i < N * N; ++i){
    A[i] = 0.f;
    B[i] = 0.f;
    C[i] = 0.f;
  }
  for (int i = 0; i < N; ++i){
    A[i + i * N] = 1.f; // identity
    B[i + i * N] = 1.f; // identity
  }

  // allocate memory and copy to the GPU
  float * d_A;
  float * d_B;
  float * d_C;
  int size = N * N * sizeof(float);
  cudaMalloc((void **) &d_A, size);
  cudaMalloc((void **) &d_B, size);
  cudaMalloc((void **) &d_C, size);
  
  // copy memory over to the GPU
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);

  // Next largest multiple of blockSize
  int numBlocks = (N + BLOCKSIZE - 1) / BLOCKSIZE; 
//   printf("N = %d, numBlocks * blockSize = %d\n", N, numBlocks * BLOCKSIZE);
  dim3 gridDims(numBlocks, numBlocks);
  dim3 blockDims(BLOCKSIZE, BLOCKSIZE);
  
int num_trials = 10;
if (version ==1) {

    #if 1
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int i = 0; i < num_trials; ++i){
        matmul1 <<< gridDims, blockDims >>> (N, d_A, d_B, d_C);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    printf("Time to run kernel 10x: %6.2f ms.\n", time);
    #endif

}

if (version ==2) {
    #if 1
    float time2;
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);

    for (int i = 0; i < num_trials; ++i){
        matmul2 <<< gridDims, blockDims >>> (N, d_A, d_B, d_C);
    }

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&time2, start2, stop2);
    
    printf("Time to run kernel 10x: %6.2f ms.\n", time2);
    #endif
}

if (version ==3) {
    #if 1
    float time3;
    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    cudaEventRecord(start3, 0);

    for (int i = 0; i < num_trials; ++i){
        matmul3 <<< gridDims, blockDims >>> (N, d_A, d_B, d_C);
    }

    cudaEventRecord(stop3, 0);
    cudaEventSynchronize(stop3);
    cudaEventElapsedTime(&time3, start3, stop3);
    
    printf("Time to run kernel 10x: %6.2f ms.\n", time3);
    #endif
}
    
  


//   // copy memory back to the CPU
//   cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
  
//   float error = 0.f;
//   for (int i = 0; i < N; ++i){
//     for (int j = 0; j < N; ++j){
//       //      printf("C[%d,%d] = %f\n", i, j, C[j + i * N]);
//       float Cij = 0.f;
//       if (i==j){
// 	Cij = 1.f;
//       }
//       float diff = C[j + i * N] - Cij;
//       error += fabs(diff);
//     }
//   }
//   printf("error = %f\n", error);

}
}

return 0;
}
    



  
