#include "cg.hh"
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <fstream>

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;


void partition_matrix(int N, int psize, int start_rows[], int offsets_lengths[])
{
    if (psize == 1)
    {
        start_rows[0] = 0;
        offsets_lengths[0] = N;
    }
    else
    {
        int N_loc = N / psize;
        start_rows[0] = 0;
        offsets_lengths[0] = N_loc;
        int i0 = N_loc;
        for(int prank = 1; prank < psize-1; prank++)
        {
            start_rows[prank] = i0;
            offsets_lengths[prank] = N_loc;
            i0 += N_loc;
        }
        start_rows[psize-1] = i0;
        offsets_lengths[psize-1] = N - i0;
    }
}


int main(int argc, char ** argv) {
    MPI_Init(&argc, &argv);

    /// MPI: Initialize and get rank
    int prank, psize;
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);

    if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " [matrix-market-filename]"
              << std::endl;
    return 1;
    }

    // initialize solver and read matrix from file
    CGSolver solver;
    solver.read_matrix(argv[1]);

    // get size of the matrix
    int n = solver.n();
    int m = solver.m();

    /// MPI: domain decomposition along rows
    int *start_rows;
    start_rows = new int [psize];
    int *offsets_lengths;
    offsets_lengths = new int [psize];
    partition_matrix(m, psize, start_rows, offsets_lengths);

    // initialize global source term
    double h = 1. / n;
    solver.init_source_term(h);

    // initialize solution vector
    std::vector<double> x_d(n);
    std::fill(x_d.begin(), x_d.end(), 0.);

    // solve and print statistics
    if (prank == 0) std::cout << "Call CG dense on matrix size (" << m << " x " << n << ")" << std::endl;
    auto t1 = clk::now();
    if (psize == 1) solver.serial_solve(x_d);
    else solver.solve(prank, start_rows, offsets_lengths, x_d);
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