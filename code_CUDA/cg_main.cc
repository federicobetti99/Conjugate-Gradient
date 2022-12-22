#include "cg.hh"
#include <chrono>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

static void usage(const std::string & prog_name) {
  std::cerr << prog_name << " <grid_size> <block_size [default: 32]>" << std::endl;
  exit(0);
}

int main(int argc, char ** argv) {

    if (argc < 2) usage(argv[0]);

    /// get type of kernel for matrix vector product
    std::string KERNEL_TYPE(argv[2]);
    if (std::strcmp(KERNEL_TYPE.c_str(), "NAIVE") || std::strcmp(KERNEL_TYPE.c_str(), "NAIVE_T") ||
        std::strcmp(KERNEL_TYPE.c_str(), "EFFICIENT") || std::strcmp(KERNEL_TYPE.c_str(), "EFFICIENT_T")) {
        throw std::invalid_argument("Received non available kernel for matrix vector product.\n"
                                    "Please, select one of the followings:"
                                    "1. NAIVE (One thread per row)"
                                    "2. NAIVE_T (One thread per row with coalescing)"
                                    "3. EFFICIENT (Multiple threads per row)"
                                    "4. EFFICIENT_T (Multiple threads per row with coalescing)");
    }

    /// get number of threads per block
    int BLOCK_WIDTH = std::stoi(argv[3]);

    /// get block width for matrix vector product grid
    int BLOCK_HEIGHT = std::stoi(argv[4]);

    /// file where to save the results
    std::string OUTPUT_FILE(argv[4]);

    /// initialize solver and read matrix from file
    CGSolver solver;
    solver.read_matrix(argv[1]);

    /// get size of the matrix
    int n = solver.n();

    /// initialize global source term
    double h = 1. / n;
    solver.init_source_term(h);

    /// initialize solution vector, filled afterwards in solve function with CUDA kernel
    double *x_d;
    cudaMallocManaged(&x_d, n * sizeof(double));

    /// solve and print statistics
    auto t1 = clk::now();
    solver.solve(x_d, KERNEL_TYPE, BLOCK_WIDTH, BLOCK_HEIGHT);
    second elapsed = clk::now() - t1;
    std::cout << "Time for CG (dense solver)  = " << elapsed.count() << " [s]\n";

    /// save results to file
    std::ofstream outfile;
    outfile.open(OUTPUT_FILE.c_str(), std::ios_base::app);
    outfile << BLOCK_WIDTH << "," << BLOCK_HEIGHT << "," << elapsed.count() << std::endl;
    outfile.close();

    return 0;
}
