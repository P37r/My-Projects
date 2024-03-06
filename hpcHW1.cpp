
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

using namespace std::chrono;

// Define a type for matrices
using Matrix = std::vector<std::vector<int>>;

int RMM(Matrix* C, Matrix* A, Matrix* B, std::tuple<int, int> A_idx, std::tuple<int, int> B_idx ,int n, int block_n, std::tuple<int, int> initial_i_j) {
    int halfSize = n / 2;
    int C_adjust_i = std::get<0>(initial_i_j);
    int C_adjust_j = std::get<1>(initial_i_j);

    int A_adjust_i = std::get<0>(A_idx);
    int A_adjust_j = std::get<1>(A_idx);

    int B_adjust_i = std::get<0>(B_idx);
    int B_adjust_j = std::get<1>(B_idx);


    if (n == block_n) {
        // Perform multiplication
        for (int i = 0; i < block_n; ++i) {
            for (int j = 0; j < block_n; ++j) {
                for (int k = 0; k < block_n; ++k) {
                    (*C)[i+C_adjust_i][j+ C_adjust_j] += (*A)[i + A_adjust_i][k+ A_adjust_j]  * (*B)[k + B_adjust_i][j + B_adjust_j]; // Dereferencing pointers
                }
            }
        }

    } 
    else {
        
        auto Ctopleft = std::make_tuple(C_adjust_i,C_adjust_j);
        auto Ctopright = std::make_tuple(C_adjust_i,C_adjust_j + halfSize);
        auto Cbottomleft = std::make_tuple(C_adjust_i + halfSize,C_adjust_j);
        auto Cbottomright = std::make_tuple(C_adjust_i+ halfSize,C_adjust_j + halfSize);

        auto Atopleft = std::make_tuple(A_adjust_i,A_adjust_j);
        auto Atopright = std::make_tuple(A_adjust_i,A_adjust_j + halfSize);
        auto Abottomleft = std::make_tuple(A_adjust_i + halfSize,A_adjust_j);
        auto Abottomright = std::make_tuple(A_adjust_i+ halfSize,A_adjust_j + halfSize);

        auto Btopleft = std::make_tuple(B_adjust_i,B_adjust_j);
        auto Btopright = std::make_tuple(B_adjust_i,B_adjust_j + halfSize);
        auto Bbottomleft = std::make_tuple(B_adjust_i + halfSize,B_adjust_j);
        auto Bbottomright = std::make_tuple(B_adjust_i+ halfSize,B_adjust_j + halfSize);
        
    
        // C00
        RMM(C, A,B, Atopleft, Btopleft, halfSize, block_n, Ctopleft);
        RMM(C, A,B, Atopright, Bbottomleft, halfSize, block_n, Ctopleft);

        // C01
        RMM(C, A,B, Atopleft, Atopright, halfSize, block_n, Ctopright);
        RMM(C, A,B, Atopright, Bbottomright, halfSize, block_n, Ctopright);

         // C10
        RMM(C, A,B, Abottomleft, Btopleft, halfSize, block_n, Cbottomleft);
        RMM(C, A,B, Abottomright, Bbottomleft, halfSize, block_n, Cbottomleft);

        // C11  
        RMM(C, A,B, Abottomleft, Btopright, halfSize, block_n, Cbottomright);
        RMM(C, A,B, Abottomright, Bbottomright, halfSize, block_n, Cbottomright);
    }
    return 0;
}






int main() {
    std::cout << "____________________________________________ \n";

    std::cout << "Verify RMM Correctness \n";
    std::cout << "____________________________________________ \n";
    std::cout << " \n";


// Verify RMM Correctness
    int n = 1024;

    Matrix *A = new Matrix(n, std::vector<int>(n, 0));
    Matrix *B = new Matrix(n, std::vector<int>(n, 0));
    Matrix *C = new Matrix(n, std::vector<int>(n, 0));

    
    // Iterate over the rows
    for (int i = 0; i < n; ++i) {
        // Iterate over the columns
        for (int j = 0; j < n; ++j) {
            // Set the diagonal elements to 1
            if (i == j) {
            (*A)[i][j] = 1;
            (*B)[i][j] = 1;
            }
        }
    }

    int block_n = 2;
    
    RMM(C, A, B,std::make_tuple(0,0), std::make_tuple(0,0), n, block_n, std::make_tuple(0,0));


    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (j == i && (*C)[i][j] != 1) {
                std::cout << "ERROR!";

            }

            if (j != i && (*C)[i][j] != 0) {
                std::cout << "ERROR!";
            }
        }
    }

    std::cout << "If there are no 'ERRORS!', then the program has run successfully! \n";
    std::cout << " \n";

    std::cout << "____________________________________________ \n";
    std::cout << " Running RMM for different matrix sizes and block sizes \n";

    std::cout << "____________________________________________ \n";
    std::cout << " \n";



// Running RMM for different matrix sizes and block sizes
    for (int n_pow = 1; n_pow <= 10; ++n_pow) {

        int n = pow(2, n_pow);
        std::cout << "Matrix Size: " << n << "\n";
        Matrix *A = new Matrix(n, std::vector<int>(n, 0));
        Matrix *B = new Matrix(n, std::vector<int>(n, 0));
        Matrix *C = new Matrix(n, std::vector<int>(n, 0));


        // Iterate over the rows
        for (int i = 0; i < n; ++i) {
            // Iterate over the columns
            for (int j = 0; j < n; ++j) {
                // Set the diagonal elements to 1
                if (i==j) {
                    (*A)[i][j] = 1;
                    (*B)[i][j] = 1;
                }
            }
        }

        int num_trials = 30;
        for (int size = 1; size <= log2(n); ++size) {
            int block_n= pow(2, size);

            if (n < block_n) {
                block_n = n;
            }
            double total_elapsed_time = 0;
            for (int trial = 0; trial < num_trials; ++trial) {
                high_resolution_clock::time_point start = high_resolution_clock::now();
                RMM(C, A, B,std::make_tuple(0,0), std::make_tuple(0,0), n, block_n, std::make_tuple(0,0));
                high_resolution_clock::time_point end = high_resolution_clock::now();
                duration<double> elapsed = end - start;
                total_elapsed_time += elapsed.count() * 1000; // Convert to milliseconds
            }
            double avg_elapsed_time = total_elapsed_time / num_trials; // Average time
            std::cout << "Computation time (ms) for block size " << block_n << " = " << avg_elapsed_time << "\n";
        }

    }

    std::cout << " \n";

    std::cout << "____________________________________________ \n";
    std::cout << "Run TMM Timings \n";
    std::cout << "____________________________________________ \n";

    std::cout << " \n";

    // Run RMM Timings
    
    int number = 1024;
    int num_trials = 30;
    for (int size = 1; size <= log2(number); ++size) {
        int block_n_opt = 64;

        int n = pow(2,size);
        std::cout << "Matrix Size: " << n << "\n";
        Matrix *A = new Matrix(n, std::vector<int>(n, 0));
        Matrix *B = new Matrix(n, std::vector<int>(n, 0));
        Matrix *C = new Matrix(n, std::vector<int>(n, 0));


        // Iterate over the rows
        for (int i = 0; i < n; ++i) {
            // Iterate over the columns
            for (int j = 0; j < n; ++j) {
                // Set the diagonal elements to 1
                if (i==j) {
                    (*A)[i][j] = 1;
                    (*B)[i][j] = 1;
                }
            }
        }

        if (n < block_n_opt) {
            block_n_opt = n;
        }
        double total_elapsed_time = 0;
        for (int trial = 0; trial < num_trials; ++trial) {
            high_resolution_clock::time_point start = high_resolution_clock::now();
            RMM(C, A, B,std::make_tuple(0,0), std::make_tuple(0,0), n, block_n_opt, std::make_tuple(0,0));
            high_resolution_clock::time_point end = high_resolution_clock::now();
            duration<double> elapsed = end - start;
            total_elapsed_time += elapsed.count() * 1000; // Convert to milliseconds
        }
        double avg_elapsed_time = total_elapsed_time / num_trials; // Average time
        std::cout << "Computation time (ms) for block size " << block_n_opt << " = " << avg_elapsed_time << "\n";
    }



    return 0;

}
