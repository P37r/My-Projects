
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

using namespace std::chrono;

// Define a type for matrices
using Matrix = std::vector<std::vector<int>>;

int number = 0;


void RMM(Matrix* C, Matrix* A, Matrix* B, std::tuple<int, int> A_idx, std::tuple<int, int> B_idx ,int n, int block_n, std::tuple<int, int> initial_i_j) {
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
}


// void matrixAddition(Matrix* C, Matrix* A, Matrix* B, std::tuple<int, int> A_pos, std::tuple<int, int> B_pos, int n) {
    
//     int A_adjust_i = std::get<0>(A_pos);
//     int A_adjust_j = std::get<1>(A_pos);
//     int B_adjust_i = std::get<0>(B_pos);
//     int B_adjust_j = std::get<1>(B_pos);


    
//     // Assuming Matrix is a struct containing a 2D array to store matrix elements
//     for (int i = 0; i < n; ++i) {
//         for (int j = 0; j < n; ++j) {
//             (*C)[i][j] = (*A)[i+A_adjust_i][j+A_adjust_j] + (*B)[i+B_adjust_i][j + B_adjust_j];
//         }
//     }
// }


// void matrixSubtraction(Matrix* C, Matrix* A, Matrix* B, std::tuple<int, int> A_pos, std::tuple<int, int> B_pos, int n) {
    
//     int A_adjust_i = std::get<0>(A_pos);
//     int A_adjust_j = std::get<1>(A_pos);
//     int B_adjust_i = std::get<0>(B_pos);
//     int B_adjust_j = std::get<1>(B_pos);


    
//     // Assuming Matrix is a struct containing a 2D array to store matrix elements
//     for (int i = 0; i < n; ++i) {
//         for (int j = 0; j < n; ++j) {
//             (*C)[i][j] = (*A)[i+A_adjust_i][j+A_adjust_j] - (*B)[i+B_adjust_i][j + B_adjust_j];
//         }
//     }
// }


// void matrixSubtraction_C(Matrix* C, Matrix* A, int n, std::tuple<int, int> C_pos) {
    
//     int C_adjust_i = std::get<0>(C_pos);
//     int C_adjust_j = std::get<1>(C_pos);

//     // Assuming Matrix is a struct containing a 2D array to store matrix elements
//     for (int i = 0; i < n; ++i) {
//         for (int j = 0; j < n; ++j) {
//             (*C)[i + C_adjust_i][j + C_adjust_j] -= (*A)[i][j];
//         }
//     }
// }



// void matrixAddition_C(Matrix* C, Matrix* A, Matrix* B, Matrix* D, int n, std::tuple<int, int> C_pos) {
    
    
//     int C_adjust_i = std::get<0>(C_pos);
//     int C_adjust_j = std::get<1>(C_pos);

//     // Assuming Matrix is a struct containing a 2D array to store matrix elements
//     for (int i = 0; i < n; ++i) {
//         for (int j = 0; j < n; ++j) {
//             (*C)[i + C_adjust_i][j + C_adjust_j] += (*A)[i][j] + (*B)[i][j] + (*D)[i][j];
//         }
//     }
// }


// void matrixAddition_C2(Matrix* C, Matrix* A, Matrix* B, int n, std::tuple<int, int> C_pos) {
    
    
//     int C_adjust_i = std::get<0>(C_pos);
//     int C_adjust_j = std::get<1>(C_pos);

//     // Assuming Matrix is a struct containing a 2D array to store matrix elements
//     for (int i = 0; i < n; ++i) {
//         for (int j = 0; j < n; ++j) {
//             (*C)[i + C_adjust_i][j + C_adjust_j] += (*A)[i][j] + (*B)[i][j] ;
//         }
//     }
// }



// void Strassen(Matrix* C, Matrix* A, Matrix* B,int n, int block_n, std::tuple<int, int> initial_i_j, int M_int, int recurse_lvl, Matrix* M1, Matrix* M2, Matrix* M3,Matrix* M4, Matrix* M5, Matrix* M6, Matrix* M7
// ,Matrix* A11_A22, Matrix* B11_B22,Matrix* A21_A22, Matrix* B12_B22, Matrix* B21_B11, Matrix* A11_A12, Matrix* A21_A11, Matrix* B11_B12, Matrix* A12_A22, Matrix* B21_B22) {
//     std::cout << "M_int: "<< M_int << ""<<"recurse_lvl: " << recurse_lvl << "\n";

//     int halfSize = n / 2;

//     auto topleft = std::make_tuple(0,0);
//     auto topright = std::make_tuple(0,0 + halfSize);
//     auto bottomleft = std::make_tuple(0 + halfSize,0);
//     auto bottomright = std::make_tuple(0+ halfSize,0 + halfSize);

//     matrixAddition(A11_A22, A,A, topleft, bottomright, n );
//     matrixAddition(B11_B22, B,B, topleft, bottomright, n );

//     matrixAddition(A21_A22, A,A, bottomleft, bottomright, n );
//     matrixSubtraction(B12_B22, B,B, topright, bottomright, n );

//     matrixSubtraction(B21_B11, B,B, bottomleft, topleft, n );

//     matrixAddition(A11_A12, A,A, topleft, topright, n );
//     matrixSubtraction(A21_A11, A,A, bottomleft, topleft, n );

//     matrixAddition(B11_B12, B,B, topleft, topright, n );
//     matrixSubtraction(A12_A22, A,A, topright, bottomright, n );
//     matrixAddition(B21_B22, B,B, bottomleft, bottomright, n );

//     if (n == block_n) {
//         // Perform multiplication
//         for (int i = 0; i < block_n; ++i) {
//             for (int j = 0; j < block_n; ++j) {
//                 for (int k = 0; k < block_n; ++k) {
//                     (*C)[i][j] += (*A)[i][k]  * (*B)[k][j]; // Dereferencing pointers
//                 }
//             }
//         }

//     } 
//     else {
//         //    COO
//        // M1
//         Strassen(M1, A11_A22,B11_B22, halfSize, block_n, topleft, 1, recurse_lvl +1, M1, M2, M3, M4, M5, M6, M7, A11_A22, B11_B22, A21_A22, B12_B22, B21_B11, A11_A12, A21_A11, B11_B12, A12_A22, B21_B22);
//         // M2
//         Strassen(M2, A21_A22, B11, halfSize, block_n, topleft, 2, recurse_lvl +1, M1, M2, M3, M4, M5, M6, M7, A11_A22, B11_B22, A21_A22, B12_B22, B21_B11, A11_A12, A21_A11, B11_B12, A12_A22, B21_B22);
//         // M3
//         Strassen(M3, A_11, B12_B22, halfSize, block_n, topleft, 3, recurse_lvl +1, M1, M2, M3, M4, M5, M6, M7, A11_A22, B11_B22, A21_A22, B12_B22, B21_B11, A11_A12, A21_A11, B11_B12, A12_A22, B21_B22);
//         // M4
//         Strassen(M4, A22 ,B21_B11, halfSize, block_n, topleft, 4, recurse_lvl +1, M1, M2, M3, M4, M5, M6, M7, A21_A22, B11_B11, A21_A22, B12_B22, B21_B11, A11_A12, A21_A11, B11_B12, A12_A22, B21_B22);
//         // M5
//         Strassen(M5, A11_A12, B22, halfSize, block_n, topleft, 5, recurse_lvl +1, M1, M2, M3, M4, M5, M6, M7, A11_A11, B11_B12, A21_A22, B12_B22, B21_B11, A11_A12, A21_A11, B11_B12, A12_A22, B21_B22);
//         // M6
//         Strassen(M6, A21_A11, B11_B12, halfSize, block_n, topleft, 6, recurse_lvl +1, M1, M2, M3, M4, M5, M6, M7, A11_A11, B11_B12, A21_A22, B12_B22, B21_B11, A11_A12, A21_A11, B11_B12, A12_A22, B21_B22);
//         // M7
//         Strassen(M7, A12_A22,B21_B22, halfSize, block_n, topleft, 7, recurse_lvl +1, M1, M2, M3, M4, M5, M6, M7, A12_A22, B21_B22, A21_A22, B12_B22, B21_B11, A11_A12, A21_A11, B11_B12, A12_A22, B21_B22);
        

        


// // matrixAddition_C(Matrix* C, Matrix* A, Matrix* B, Matrix* D, int n, std::tuple<int, int> C_pos)
//     //    C00
//         matrixAddition_C(C, M1,  M4, M7, halfSize, topleft);
//         matrixSubtraction_C(C, M5, halfSize, topleft);



// // C01
//         matrixAddition_C2(C, M3,  M5, halfSize, topright);


// // C10
//         matrixAddition_C2(C, M2,  M4, halfSize, bottomleft);


// // C11
//         matrixAddition_C(C, M1,  M3, M6, halfSize, bottomright);
//         matrixSubtraction_C(C, M2, halfSize, bottomright);
//     }
// }
        





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

        int num_trials = 5;
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
    int num_trials = 5;
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
