#include "cg.hh"
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <fstream>
#include <cuda_runtime.h>

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;


int main(int argc, char ** argv) {
    MPI_Init(&argc, &argv);

    if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " [matrix-market-filename]"
              << std::endl;
    return 1;
    }

    dim3 block_size;
    if (argc >= 2) {
        try {
            block_size.x = std::stoi(argv[2]);
        } catch(std::invalid_argument &) {
            usage(argv[0]);
        }
    }

    if (argc == 3) {
        try {
            block_size.y = std::stoi(argv[3]);
        } catch(std::invalid_argument &) {
            usage(argv[0]);
        }
    }

    // initialize solver and read matrix from file
    CGSolver solver;
    solver.read_matrix(argv[1]);

    // get size of the matrix
    int n = solver.n();
    int m = solver.m();

    // initialize global source term
    double h = 1. / n;
    solver.init_source_term(h);

    // initialize solution vector
    std::vector<double> x_d(n);
    std::fill(x_d.begin(), x_d.end(), 0.);

    // solve and print statistics
    auto t1 = clk::now();
    else solver.kerneled_solve(x_d, block_size);
    second elapsed = clk::now() - t1;
    second max_time;
    MPI_Allreduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (prank == 0) std::cout << "Time for CG (dense solver)  = " << max_time.count() << " [s]\n";

    if (prank == 0) {
        // save results to file
        std::ofstream outfile;
        outfile.open("../results/sync_strong_scaling.txt", std::ios_base::app);
        outfile << psize << "," << max_time.count() << std::endl;
        outfile.close();
    } 

    /// MPI: Finalize
    MPI_Finalize();
    return 0;
}
