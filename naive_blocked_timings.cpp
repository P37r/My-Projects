#include <iostream>
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;

// computes C = C + A*B
void matmul_naive(const int n, double* C, double* A, double* B){
  for (int i = 0; i < n; ++i){
    for (int j = 0; j < n; ++j){
      double Cij = C[j + i * n];
      for (int k = 0; k < n; ++k){
	double Aij = A[k + i * n];
	double Bjk = B[j + k * n];	
	Cij += Aij * Bjk;
      }
      C[j + i * n] = Cij;
    }
  }
}


void matmul_blocked(const int n, double* C, double* A, double* B, int BLOCK_SIZE){
  for (int i = 0; i < n; i += BLOCK_SIZE){
    for (int j = 0; j < n; j += BLOCK_SIZE){
      for (int k = 0; k < n; k += BLOCK_SIZE){

	// small matmul
	for (int ii = i; ii < i + BLOCK_SIZE; ii++){
	  for (int jj = j; jj < j + BLOCK_SIZE; jj++){
	    double Cij = C[jj + ii * n];
	    for (int kk = k; kk < k + BLOCK_SIZE; kk++){
	      Cij += A[kk + ii * n] * B[jj + kk * n]; // Aik * Bkj
	    }
	    C[jj + ii * n] = Cij;
	  }
	}
	
      }
    }
  }
}

int main(){
  int num_trials = 5;

  for (int i = 4; i <=10; ++i) {
    int n = pow(2,i);
    std::cout << "Matrix Size: " << n << "\n";

    double * A = new double[n * n];
    double * B = new double[n * n];
    double * C = new double[n * n];

    // make A, B = I
    for (int i = 0; i < n; ++i){
      A[i + i * n] = 1.0;
      B[i + i * n] = 1.0;
    }
    for (int i = 0; i < n * n; ++i){
      C[i] = 0.0;
    }
    // Measure performance
    high_resolution_clock::time_point start = high_resolution_clock::now();
    for (int i = 0; i < num_trials; ++i){
      matmul_naive(n, C, A, B);
    }
    high_resolution_clock::time_point end = high_resolution_clock::now();
    duration<double> elapsed_naive = (end - start) / num_trials;
    cout << "Naive elapsed time (ms) = " << elapsed_naive.count() * 1000 << endl;


  // Block Part
    for (int size = 1; size <= log2(n); ++size) {

      double * A = new double[n * n];
      double * B = new double[n * n];
      double * C = new double[n * n];

      // make A, B = I
      for (int i = 0; i < n; ++i){
        A[i + i * n] = 1.0;
        B[i + i * n] = 1.0;
      }
      for (int i = 0; i < n * n; ++i){
        C[i] = 0.0;
      }

      int block_n= pow(2, size);
      // Measure performance  
      start = high_resolution_clock::now();
      for (int i = 0; i < num_trials; ++i){  
        matmul_blocked(n, C, A, B,block_n);
      }
      end = high_resolution_clock::now();
      duration<double> elapsed_blocked = (end - start) / num_trials;
      
      cout << "BLOCK SIZE: " << block_n << " Blocked elapsed time (ms) = "<< elapsed_blocked.count() * 1000 << endl;  

      delete[] A;
      delete[] B;
      delete[] C;
      }

    



  }
  
  
  return 0;
}
