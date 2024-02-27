
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
    int block_n = 16;
    
   
   
    int num_trials = 10;
    for (int exp_num = 4; exp_num <= 10; ++exp_num) {
        std::cout << "ice spice ";
        int n = pow(2, exp_num);

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
