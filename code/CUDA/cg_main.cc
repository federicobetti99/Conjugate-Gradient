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

    /// number of threads per block
    int NUM_THREADS = std::stoi(argv[2]);

    /// block width for matrix vector product grid
    int BLOCK_WIDTH = std::stoi(argv[3]);

    /// type of kernel for matrix vector products
    bool T;
    std::string transpose(argv[4]);
    if (transpose == "true") T = true;
    else T = false;

    /// file where to save the results
    std::string OUTPUT_FILE(argv[5]);

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
    solver.solve(x_d, NUM_THREADS, BLOCK_WIDTH, T);
    second elapsed = clk::now() - t1;
    std::cout << "Time for CG (dense solver)  = " << elapsed.count() << " [s]\n";

    /// save results to file
    std::ofstream outfile;
    outfile.open(OUTPUT_FILE.c_str(), std::ios_base::app);
    outfile << NUM_THREADS << "," << BLOCK_WIDTH << "," << elapsed.count() << std::endl;
    outfile.close();

    return 0;
}
