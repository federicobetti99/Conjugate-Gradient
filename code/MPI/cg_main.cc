#include "cg.hh"
#include <chrono>
#include <iostream>
#include <sstream>
#include <mpi.h>
#include <fstream>

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;


int main(int argc, char ** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    /// MPI: Initialize and get rank
    int prank, psize;
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);

    if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " [matrix-market-filename]"
              << std::endl;
    return 1;
    }

    // initialize solver and generate Laplacian matrix of user-defined size
    // see the function for detailed documentation
    CGSolver solver;
    solver.generate_lap2d_matrix(std::stoi(argv[1]));

    // get size of the matrix
    int n = solver.n();
    int m = solver.m();

    // possibility of reducing number of iterations for weak scaling experiments
    int maxIter;
    if (argc >= 4) {
        std::stringstream arg_0(argv[3]);
        arg_0 >> maxIter;
        solver.set_max_iter(maxIter);
    }

    // initialize global source term
    double h = 1. / n;
    solver.init_source_term(h);

    // initialize solution vector
    std::vector<double> x_d(n);
    std::fill(x_d.begin(), x_d.end(), 0.);

    // solve and print statistics
    auto t1 = clk::now();
    solver.solve(x_d);
    second elapsed = clk::now() - t1;

    if (prank == 0) {
        // save results to file
        std::string OUTPUT_FILE(argv[2]);
        std::ofstream outfile;
        outfile.open(OUTPUT_FILE.c_str(), std::ios_base::app);
        outfile << n << "," << psize << "," << elapsed.count() << std::endl;
        outfile.close();
    }

    /// MPI: Finalize
    MPI_Finalize();
    return 0;
}
