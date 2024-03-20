#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>


#define NUM_THREADS 8
using namespace std::chrono;
using Matrix = std::vector<std::vector<int>>;


int backwardsSolve_static(Matrix A, Matrix B) {
    
    int n = B.size();
    Matrix x =  Matrix(n, std::vector<int>(1, 0));

    #pragma omp parallel for 
    for (int i = 0; i < n; ++i) {
        x[i][0] = B[i][0];
    }

    #pragma omp parallel for 

    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += A[i][j] * x[j][0];
        }
        x[i][0] -=  sum;
        x[i][0] /=  A[i][i];
    }

    return 0;
}


int backwardsSolve_dynamic(Matrix A, Matrix B) {
    
    int n = B.size();
    Matrix x =  Matrix(n, std::vector<int>(1, 0));

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        x[i][0] = B[i][0];
    }

    #pragma omp parallel for schedule(dynamic)

    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += A[i][j] * x[j][0];
        }
        x[i][0] -=  sum;
        x[i][0] /=  A[i][i];
    }




    return 0;
}



int backwardsSolve_single(Matrix A, Matrix B) {
    
    int n = B.size();
    Matrix x =  Matrix(n, std::vector<int>(1, 0));

    
    for (int i = 0; i < n; ++i) {
        x[i][0] = B[i][0];
    }

    
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += A[i][j] * x[j][0];
        }
        x[i][0] -=  sum;
        x[i][0] /=  A[i][i];
    }


    // std::cout << "Matrix:" << std::endl;
    // for (int i = 0; i < x.size(); ++i) {
    //     for (int j = 0; j < x[i].size(); ++j) {
    //         std::cout << x[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    return 0;
}


int main() {

    int trials = 20;
    double static_array[1000];
    double dynamic_array[1000];
    double single_array[1000];

    omp_set_num_threads(NUM_THREADS);
    for (int size = 1; size < 1001; ++size) {
        
        std::vector<std::vector<int>> upper_triangular(size, std::vector<int>(size, 0));
        std::vector<std::vector<int>> enumerated_matrix(size, std::vector<int>(1, 0));

        // Create the upper triangular ones matrix
        for (int i = 0; i < size; ++i) {
            for (int j = i; j < size; ++j) {
                upper_triangular[i][j] = 1;
            }
        }
        
       

        for (int i = 0; i < size; ++i) {
            enumerated_matrix[i][0] = i + 1;
        }
        
        // Static timing
        double total_elapsed_time = 0;
         for(int i = 0; i < trials; ++i) {
            high_resolution_clock::time_point start = high_resolution_clock::now();

            backwardsSolve_static(upper_triangular,enumerated_matrix);
            high_resolution_clock::time_point end = high_resolution_clock::now();
            duration<double> elapsed = end - start;
            total_elapsed_time += elapsed.count();
        }
        total_elapsed_time /= trials;
        static_array[size] = total_elapsed_time;

        // dynamic timing

        total_elapsed_time = 0;
         for(int i = 0; i < trials; ++i) {
            high_resolution_clock::time_point start = high_resolution_clock::now();

            backwardsSolve_dynamic(upper_triangular,enumerated_matrix);
            high_resolution_clock::time_point end = high_resolution_clock::now();
            duration<double> elapsed = end - start;
            total_elapsed_time += elapsed.count();
        }
        total_elapsed_time /= trials;
        dynamic_array[size] = total_elapsed_time;

        // single timing

        total_elapsed_time = 0;
         for(int i = 0; i < trials; ++i) {
            high_resolution_clock::time_point start = high_resolution_clock::now();

            backwardsSolve_single(upper_triangular,enumerated_matrix);
            high_resolution_clock::time_point end = high_resolution_clock::now();
            duration<double> elapsed = end - start;
            total_elapsed_time += elapsed.count();
        }
        total_elapsed_time /= trials;
        single_array[size] = total_elapsed_time;
    }
    std::cout << "single " << std::endl;
    std::cout  << std::endl;
    // static print out  

// 
    std::cout << "[";

    for (int i = 1; i < 1001; ++i) {
        std::cout << static_array[i];
        if (i < 1001 - 1) {
            std::cout << ",";
        }
    }

    std::cout << "]" << std::endl;
// 
    
    std::cout << "dynamic timings" << std::endl;
    std::cout  << std::endl;
    // dyanamic print out  
    // 
    std::cout << "[";

    for (int i = 1; i < 1001; ++i) {
        std::cout << dynamic_array[i];
        if (i < 1001 - 1) {
            std::cout << ",";
        }
    }

    std::cout << "]" << std::endl;
// 

    std::cout << "singular timings" << std::endl;
    std::cout  << std::endl;
    // singular print out  
    // 
    std::cout << "[";

    for (int i = 1; i < 1001; ++i) {
        std::cout << single_array[i];
        if (i < 1001 - 1) {
            std::cout << ",";
        }
    }

    std::cout << "]" << std::endl;
// 

    // const int size = 50;
    // std::vector<std::vector<int>> upper_triangular(size, std::vector<int>(size, 0));
    // std::vector<std::vector<int>> enumerated_matrix(size, std::vector<int>(1, 0));

    // // Create the upper triangular ones matrix
    // for (int i = 0; i < size; ++i) {
    //     for (int j = i; j < size; ++j) {
    //         upper_triangular[i][j] = 1;
    //     }
    // }

    // for (int i = 0; i < size; ++i) {
    //     enumerated_matrix[i][0] = i + 1;
    // }
    // high_resolution_clock::time_point start = high_resolution_clock::now();

    // backwardsSolve(upper_triangular,enumerated_matrix);
    // high_resolution_clock::time_point end = high_resolution_clock::now();
    // duration<double> elapsed = end - start;
    // std::cout << "Elapsed time: " << elapsed.count() << "s\n";
    

    return 0;
}