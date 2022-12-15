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

    dim3 block_size;
    if (argc >= 3) {
        try {
            block_size.x = std::stoi(argv[2]);
        } catch(std::invalid_argument &) {
            usage(argv[0]);
        }
    }

    if (argc >= 4) {
        try {
            block_size.y = std::stoi(argv[3]);
        } catch(std::invalid_argument &) {
            usage(argv[0]);
        }
    }

    std::string OUTPUT_FILE(argv[4]);

    // initialize solver and read matrix from file
    CGSolver solver;
    solver.read_matrix(argv[1]);

    // get size of the matrix
    int n = solver.n();

    // initialize global source term
    double h = 1. / n;
    solver.init_source_term(h);

    // initialize solution vector
    double *x_d;
    cudaMallocManaged(&x_d, n*sizeof(double));

    // solve and print statistics
    auto t1 = clk::now();
    solver.kerneled_solve(x_d, block_size);
    second elapsed = clk::now() - t1;
    std::cout << "Time for CG (dense solver)  = " << elapsed.count() << " [s]\n";

    // save results to file
    std::ofstream outfile;
    outfile.open(OUTPUT_FILE.c_str(), std::ios_base::app);
    outfile << block_size.x << "," << block_size.y << "," << elapsed.count() << std::endl;
    outfile.close();

    return 0;
}
