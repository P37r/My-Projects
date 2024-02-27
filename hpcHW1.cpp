// #include <iostream>
// #include <vector>
// #include <cmath>
// #include <chrono>

// using namespace std::chrono;

// // Define a type for matrices
// using Matrix = std::vector<std::vector<int>>;

// // Function to add two matrices
// Matrix addMatrices(const Matrix A, const Matrix B) {
//     int numRows = A.size();
//     int numCols = A[0].size();
//     Matrix result(numRows, std::vector<int>(numCols));

//     for (int i = 0; i < numRows; ++i) {
//         for (int j = 0; j < numCols; ++j) {
//             result[i][j] = A[i][j] + B[i][j];
//         }
//     }

//     return result;
// }

// Matrix subtractMatrices(const Matrix A, const Matrix B) {
//     int numRows = A.size();
//     int numCols = A[0].size();
//     Matrix result(numRows, std::vector<int>(numCols));

//     for (int i = 0; i < numRows; ++i) {
//         for (int j = 0; j < numCols; ++j) {
//             result[i][j] = A[i][j] - B[i][j];
//         }
//     }

//     return result;
// }

// // Function to split a matrix into four quadrants
// std::tuple<Matrix, Matrix, Matrix, Matrix> splitMatrix(const Matrix A, Matrix A00 ,Matrix A01, Matrix A10, Matrix A11) {
//     int numRows = A.size();
//     int halfSize = numRows / 2;

//     // Populate the quadrants
//     for (int i = 0; i < halfSize; ++i) {
//         for (int j = 0; j < halfSize; ++j) {
//             A00[i][j] = A[i][j];
//             A01[i][j] = A[i][j + halfSize];
//             A10[i][j] = A[i + halfSize][j];
//             A11[i][j] = A[i + halfSize][j + halfSize];
//         }
//     }
//     return std::make_tuple(A00, A01, A10, A11);
// }


// // Matrix RMM(Matrix A, Matrix B, int n, int block_n) {
// //     int halfSize = n / 2;
    
// //     if (n == block_n) {
// //         Matrix result(block_n, std::vector<int>(block_n, 0));
        
// //     // Perform multiplication
// //         for (int i = 0; i < block_n; ++i) {
// //             for (int j = 0; j < block_n; ++j) {
// //                 for (int k = 0; k < block_n; ++k) {
// //                     result[i][j] += A[i][k] * B[k][j];
// //                 }
// //             }
// //         }


// //         return result;
// //     } else {
// //         Matrix C00(halfSize, std::vector<int>(halfSize));
// //         Matrix C01(halfSize, std::vector<int>(halfSize));
// //         Matrix C10(halfSize, std::vector<int>(halfSize));
// //         Matrix C11(halfSize, std::vector<int>(halfSize));

// //         Matrix A00(halfSize, std::vector<int>(halfSize));
// //         Matrix A01(halfSize, std::vector<int>(halfSize));
// //         Matrix A10(halfSize, std::vector<int>(halfSize));
// //         Matrix A11(halfSize, std::vector<int>(halfSize));

// //         Matrix B00(halfSize, std::vector<int>(halfSize));
// //         Matrix B01(halfSize, std::vector<int>(halfSize));
// //         Matrix B10(halfSize, std::vector<int>(halfSize));
// //         Matrix B11(halfSize, std::vector<int>(halfSize));


// //         auto A_split = splitMatrix(A, A00, A01, A10, A11);
// //         A00 = std::get<0>(A_split);
// //         A01 = std::get<1>(A_split);
// //         A10 = std::get<2>(A_split);
// //         A11 = std::get<3>(A_split);
        

// //         auto B_split = splitMatrix(B, B00, B01, B10, B11);
// //         B00 = std::get<0>(B_split);
// //         B01 = std::get<1>(B_split);
// //         B10 = std::get<2>(B_split);
// //         B11 = std::get<3>(B_split);

// //         C00 = addMatrices(RMM(A00, B00, halfSize,block_n), RMM(A01, B10, halfSize,block_n));
// //         C01 = addMatrices(RMM(A00, B01, halfSize,block_n), RMM(A01, B11, halfSize,block_n));
// //         C10 = addMatrices(RMM(A10, B00, halfSize,block_n), RMM(A11, B10, halfSize,block_n));
// //         C11 = addMatrices(RMM(A10, B01, halfSize,block_n), RMM(A11, B11, halfSize,block_n));
        
// //         Matrix result(n, std::vector<int>(n, 0));

// //         // Reassemble matrix
// //         for (int i = 0; i < halfSize; ++i) {
// //             for (int j = 0; j < halfSize; ++j) {
// //                 result[i][j] = C00[i][j];
// //                 result[i][j + halfSize] = C01[i][j];
// //                 result[i + halfSize][j] = C10[i][j];
// //                 result[i + halfSize][j + halfSize] = C11[i][j];
// //             }
// //         }

// //         return result;
// //     }
// // }

// Matrix RMM(Matrix* A, Matrix* B, int n, int block_n) {
//     int halfSize = n / 2;
    
//     if (n == block_n) {
//         Matrix result(block_n, std::vector<int>(block_n, 0));
        
//         // Perform multiplication
//         for (int i = 0; i < block_n; ++i) {
//             for (int j = 0; j < block_n; ++j) {
//                 for (int k = 0; k < block_n; ++k) {
//                     result[i][j] += (*A)[i][k] * (*B)[k][j]; // Dereferencing pointers
//                 }
//             }
//         }

//         return result;
//     } else {
//         Matrix C00(halfSize, std::vector<int>(halfSize));
//         Matrix C01(halfSize, std::vector<int>(halfSize));
//         Matrix C10(halfSize, std::vector<int>(halfSize));
//         Matrix C11(halfSize, std::vector<int>(halfSize));

//         Matrix A00(halfSize, std::vector<int>(halfSize));
//         Matrix A01(halfSize, std::vector<int>(halfSize));
//         Matrix A10(halfSize, std::vector<int>(halfSize));
//         Matrix A11(halfSize, std::vector<int>(halfSize));

//         Matrix B00(halfSize, std::vector<int>(halfSize));
//         Matrix B01(halfSize, std::vector<int>(halfSize));
//         Matrix B10(halfSize, std::vector<int>(halfSize));
//         Matrix B11(halfSize, std::vector<int>(halfSize));

//         auto A_split = splitMatrix(*A, A00, A01, A10, A11);
//         A00 = std::get<0>(A_split);
//         A01 = std::get<1>(A_split);
//         A10 = std::get<2>(A_split);
//         A11 = std::get<3>(A_split);
        
//         auto B_split = splitMatrix(*B, B00, B01, B10, B11);
//         B00 = std::get<0>(B_split);
//         B01 = std::get<1>(B_split);
//         B10 = std::get<2>(B_split);
//         B11 = std::get<3>(B_split);

//         auto C00_temp = RMM(&A00, &B00, halfSize, block_n);
//         auto C01_temp = RMM(&A00, &B01, halfSize, block_n);
//         auto C10_temp = RMM(&A10, &B00, halfSize, block_n);
//         auto C11_temp = RMM(&A10, &B01, halfSize, block_n);

//         C00 = addMatrices(C00_temp, C01_temp);
//         C01 = addMatrices(RMM(&A00, &B01, halfSize, block_n), RMM(&A00, &B11, halfSize, block_n));
//         C10 = addMatrices(RMM(&A10, &B00, halfSize, block_n), RMM(&A10, &B10, halfSize, block_n));
//         C11 = addMatrices(RMM(&A10, &B01, halfSize, block_n), RMM(&A10, &B11, halfSize, block_n));
        
//         Matrix result(n, std::vector<int>(n, 0));

//         // Reassemble matrix
//         for (int i = 0; i < halfSize; ++i) {
//             for (int j = 0; j < halfSize; ++j) {
//                 result[i][j] = C00[i][j];
//                 result[i][j + halfSize] = C01[i][j];
//                 result[i + halfSize][j] = C10[i][j];
//                 result[i + halfSize][j + halfSize] = C11[i][j];
//             }
//         }

//         return result;
//     }
// }


// Matrix Strassen(Matrix A, Matrix B, int n) {
//     int halfSize = n / 2;
    
//     if (n == 1) {
//         Matrix result(1, std::vector<int>(1, 0));
//         result[0][0] = A[0][0] * B[0][0];
//         return result;
//     } else {
//         Matrix C00(halfSize, std::vector<int>(halfSize));
//         Matrix C01(halfSize, std::vector<int>(halfSize));
//         Matrix C10(halfSize, std::vector<int>(halfSize));
//         Matrix C11(halfSize, std::vector<int>(halfSize));

//         Matrix A00(halfSize, std::vector<int>(halfSize));
//         Matrix A01(halfSize, std::vector<int>(halfSize));
//         Matrix A10(halfSize, std::vector<int>(halfSize));
//         Matrix A11(halfSize, std::vector<int>(halfSize));

//         Matrix B00(halfSize, std::vector<int>(halfSize));
//         Matrix B01(halfSize, std::vector<int>(halfSize));
//         Matrix B10(halfSize, std::vector<int>(halfSize));
//         Matrix B11(halfSize, std::vector<int>(halfSize));

//         auto A_split = splitMatrix(A, A00, A01, A10, A11);
//         A00 = std::get<0>(A_split);
//         A01 = std::get<1>(A_split);
//         A10 = std::get<2>(A_split);
//         A11 = std::get<3>(A_split);
        
//         auto B_split = splitMatrix(B, B00, B01, B10, B11);
//         B00 = std::get<0>(B_split);
//         B01 = std::get<1>(B_split);
//         B10 = std::get<2>(B_split);
//         B11 = std::get<3>(B_split);

//         auto M1 = Strassen(addMatrices(A00, A11), addMatrices(B00,B11),halfSize);
//         auto M2 =Strassen(addMatrices(A10, A11) , B00, halfSize);
//         auto M3 = Strassen(A00 ,subtractMatrices(B01 , B11), halfSize);
//         auto M4 = Strassen(A11 , subtractMatrices(B10 , B00), halfSize);
//         auto M5 = Strassen(addMatrices(A00, A01) , B11, halfSize);
//         auto M6 =  Strassen(subtractMatrices(A10,A00) , addMatrices(B00, B01), halfSize);
//         auto M7 = Strassen(subtractMatrices(A01, A11) , addMatrices(B10, B11) , halfSize);



//         C00 = addMatrices(subtractMatrices(addMatrices(M1, M4),M5), M7);
//         C01 = addMatrices(M3, M5);
//         C10 = addMatrices(M2, M4);
//         C11 = addMatrices(addMatrices(subtractMatrices(M1, M2),M3),M6);
// ;
        
//         Matrix result(n, std::vector<int>(n, 0));

//         // Reassemble matrix
//         for (int i = 0; i < halfSize; ++i) {
//             for (int j = 0; j < halfSize; ++j) {
//                 result[i][j] = C00[i][j];
//                 result[i][j + halfSize] = C01[i][j];
//                 result[i + halfSize][j] = C10[i][j];
//                 result[i + halfSize][j + halfSize] = C11[i][j];
//             }
//         }

//         return result;
//     }
// }


// int main() {
//     int n = 128;

//     Matrix *A = new Matrix(n, std::vector<int>(n, 0));
//     Matrix *B = new Matrix(n, std::vector<int>(n, 0));

//     // Iterate over the rows
//     for (int i = 0; i < n; ++i) {
//         // Iterate over the columns
//         for (int j = 0; j < n; ++j) {
//             // Set the diagonal elements to 1
//             if (i == j) {
//                 (*A)[i][j] = 1;
//                 (*B)[i][j] = 1;
//             }
//         }
//     }
    

//     int block_n = 2;

//     // Matrix C = RMM(A,B,n,block_n);
//     // Print the quadrants (for demonstration purposes)
//     // std::cout << "C:\n";
//     // for (const auto& row : C) {
//     //     for (int val : row) {
//     //         std::cout << val << " ";
//     //     }
//     //     std::cout << "\n";
//     // }

//     int num_trials = 10;

//     // // Measure performance for Strassen
//     // high_resolution_clock::time_point start = high_resolution_clock::now();
//     // for (int i = 0; i < num_trials; ++i){
//     //     RMM(A,B,n,block_n);
//     // }
//     // high_resolution_clock::time_point end = high_resolution_clock::now();
//     // duration<double> elapsed_naive = (end - start) / num_trials;
//     // std::cout << "Strassen (ms) = " << elapsed_naive.count() * 1000;
//     // return 0;

//     for (int blockSizeExp = 1; blockSizeExp <= 7; ++blockSizeExp) {
//         std::cout << "ice spice";

//         int block_n = pow(2, blockSizeExp);
//         double total_elapsed_time = 0;
//         for (int trial = 0; trial < num_trials; ++trial) {
//             high_resolution_clock::time_point start = high_resolution_clock::now();
//             RMM(A, B, n, block_n);
//             high_resolution_clock::time_point end = high_resolution_clock::now();
//             duration<double> elapsed = end - start;
//             total_elapsed_time += elapsed.count() * 1000; // Convert to milliseconds
//         }
//         double avg_elapsed_time = total_elapsed_time / num_trials; // Average time
//         std::cout << "Computation time (ms) for block size " << block_n << " = " << avg_elapsed_time << "\n";
//     }

//     return 0;

// }



// ____________________________________________________________________

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
        



        // std::cout << "recurse lvl: "<<recurse_lvl;
        // std::cout << "\n";
        // number = number + 1;
        // std::cout << "recurse number"<< number ;
        // std::cout << "\n";

        // std::cout << "TOP LEFT";
        // std::cout << "First element: " << std::get<0>(topleft) << std::endl;
        // std::cout << "Second element: " << std::get<1>(topleft) << std::endl;
        // std::cout << "\n";

        // std::cout << "TOP RIGHT";
        // std::cout << "First element: " << std::get<0>(topright) << std::endl;
        // std::cout << "Second element: " << std::get<1>(topright) << std::endl;

        // std::cout << "BOTTOM LEFT";
        // std::cout << "First element: " << std::get<0>(bottomleft) << std::endl;
        // std::cout << "Second element: " << std::get<1>(bottomleft) << std::endl;
        // std::cout << "\n";

        // std::cout << "BOTTOM RIGHT";
        // std::cout << "First element: " << std::get<0>(bottomright) << std::endl;
        // std::cout << "Second element: " << std::get<1>(bottomright) << std::endl;


        // std::cout << "INITIAL IJ";
        // std::cout << "First element: " << std::get<0>(initial_i_j) << std::endl;
        // std::cout << "Second element: " << std::get<1>(initial_i_j) << std::endl;


    
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



int main() {
    block_n = 16
    
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

   
    int num_trials = 10;
    for (int exp_num = 4; exp_num <= 10; ++exp_num) {
        std::cout << "ice spice ";
        int n = pow(2, blockSizeExp);
        double total_elapsed_time = 0;
        for (int trial = 0; trial < num_trials; ++trial) {
            high_resolution_clock::time_point start = high_resolution_clock::now();
            RMM(C, A, B,std::make_tuple(0,0), std::make_tuple(0,0), n, block_n, std::make_tuple(0,0));
            high_resolution_clock::time_point end = high_resolution_clock::now();
            duration<double> elapsed = end - start;
            total_elapsed_time += elapsed.count() * 1000; // Convert to milliseconds
        }
        double avg_elapsed_time = total_elapsed_time / num_trials; // Average time
        std::cout << ".Computation time (ms) for block size " << block_n << " = " << avg_elapsed_time << "\n";
    }

    return 0;

}
