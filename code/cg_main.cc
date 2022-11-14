#include "cg.hh"
#include <chrono>
#include <iostream>
#include <mpi.h>

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

void partition_matrix(int N, int psize, std::vector<int> start_rows, std::vector<int> offsets_lengths)
{
    start_rows.clear();
    start_rows.reserve(psize+1);

    offsets_lengths.clear();
    offsets_lengths.reserve(psize+1);

    if (psize == 1)
    {
        start_rows[0] = 0;
        offsets_lengths[0] = N;
    }
    else
    {
        int N_loc = N / psize;
        start_rows[0] = 0;
        offsets_lengths[0] = N_loc + 1;
        int i0 = N_loc - 1;
        for(int prank = 1; prank < (psize-1); prank++)
        {
            start_rows[prank] = i0;
            offsets_lengths[prank] = N_loc + 2;
            i0 += N_loc;
        }
        start_rows[psize] = i0;
        offsets_lengths[psize] = N - i0;
    }
}


int main(int argc, char ** argv) {
    MPI_Init(&argc, &argv);

    int prank, psize;
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);

    if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " [matrix-market-filename]"
              << std::endl;
    return 1;
    }

    CGSolver solver;
    solver.read_matrix(argv[1]);

    int n = solver.n();
    int m = solver.m();

    std::vector<int> start_rows;
    std::vector<int> offsets_lengths;
    partition_matrix(m, psize, start_rows, offsets_lengths);

    double h = 1. / n;
    solver.init_source_term(h);

    Matrix A_sub = solver.get_submatrix(offsets_lengths[prank], start_rows[prank]);
    std::vector<double> b_sub = solver.get_subvector(offsets_lengths[prank], start_rows[prank]);

    std::vector<double> x_d(n);
    std::fill(x_d.begin(), x_d.end(), 0.);

    std::cout << "Call CG dense on matrix size (" << m << " x " << n << ")"
            << std::endl;
    auto t1 = clk::now();
    solver.solve(A_sub, b_sub, start_rows, offsets_lengths, x_d);
    second elapsed = clk::now() - t1;
    std::cout << "Time for CG (dense solver)  = " << elapsed.count() << " [s]\n";

    MPI_Finalize();
    return 0;
}
