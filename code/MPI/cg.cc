#include "cg.hh"

#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <iostream>

const double NEARZERO = 1.0e-14;
const bool DEBUG = true;

/*
    Pseudocode of parallel MPI conjugate gradient algorithm to solve the linear equation A*x = b where A is
    of size n x n
    INPUT:  A_sub contains a subset of the rows of A
            b_sub contains the corresponding indexes of b
            x_sub is a copy of x rank-wise, but of reduced size to save memory

    r_sub = b_sub - A_sub * x;
    p_sub = r_sub;
    rsold = r_sub' * r_sub; (with reduce)

    for i = 1:length(b)
        Ap_sub = A_sub * p;
        denom = (p_sub' * Ap_sub); (with reduce)
        alpha = rsold / denom;
        x_sub = x_sub + alpha * p_sub;
        r_sub = r_sub - alpha Ap_sub;
        rsnew = r_sub' * r_sub; (with reduce)
        if sqrt(rsnew) < tolerance
              break;
        end
        p_sub = r_sub + (rsnew / rsold) * p_sub;
        rsold = rsnew;
        MPI_Allgatherv the rank-wise p_sub to all processes
    end
*/

void CGSolver::solve(std::vector<double> & x)
{
    /**
    * Solve the linear system Ax = b with conjugate gradient and MPI interface
    *
    * @param start_rows first row of the submatrix of each rank
    * @param num_rows   number of rows of the submatrix of each rank
    * @param x initial guess
    * @return void
    */

    int prank, psize;
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);

    // define all useful quantities
    std::vector<double> tmp(m_n);
    std::vector<double> r(m_n);
    std::vector<double> Ap(m_n);
    std::vector<double> p(m_n);

    /// MPI: domain decomposition along rows
    int *start_rows;
    start_rows = new int [psize];
    int *num_rows;
    num_rows = new int [psize];
    partition_matrix(m_m, psize, start_rows, num_rows);

    // compute subpart of the rows of the matrix destined to prank
    int count_rows = num_rows[prank];
    int start_row = start_rows[prank];

    /// rank dependent variables
    std::vector<double> tmp_sub(count_rows);
    std::vector<double> r_sub(&m_b[start_row], &m_b[start_row+count_rows]);
    std::vector<double> x_sub(&x[start_row], &x[start_row+count_rows]);
    std::vector<double> Ap_sub(count_rows);
    std::vector<double> p_sub(count_rows);

    /// compute residual
    // r = b - A * x;
    std::fill_n(Ap_sub.begin(), Ap_sub.size(), 0.);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, count_rows, m_n, 1., m_A.data() + start_row * m_n, m_n,
                x.data(), 1, 0., Ap_sub.data(), 1);
    cblas_daxpy(r_sub.size(), -1., Ap_sub.data(), 1, r_sub.data(), 1);

    /// copy p into r and initialize overall p vector
    p_sub = r_sub;

    MPI_Allgatherv(&p_sub.front(), count_rows, MPI_DOUBLE,
		   &p.front(), num_rows, start_rows, MPI_DOUBLE, MPI_COMM_WORLD);

    /// compute residual rank-wise and reduce
    auto rsold = cblas_ddot(r_sub.size(), r_sub.data(), 1, p_sub.data(), 1);
    MPI_Allreduce(MPI_IN_PLACE, &rsold, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // for i = 1:length(b)
    int k = 0;
    for (; k < m_maxIter; ++k) {

        /// MPI: we need to gather Ap to compute the step size alpha at every iteration
        // Ap = A * p;
        std::fill_n(Ap_sub.begin(), Ap_sub.size(), 0.);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, count_rows, m_n, 1., m_A.data() + start_row * m_n, m_n,
                    p.data(), 1, 0., Ap_sub.data(), 1);

        // alpha = rsold / (p' * Ap);
        auto conj = cblas_ddot(p_sub.size(), p_sub.data(), 1, Ap_sub.data(), 1);
        MPI_Allreduce(MPI_IN_PLACE, &conj, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        auto alpha = rsold / std::max(conj, rsold * NEARZERO);

        // x = x + alpha * p;
        cblas_daxpy(count_rows, alpha, p_sub.data(), 1, x_sub.data(), 1);

        // r = r - alpha * Ap;
        cblas_daxpy(count_rows, -alpha, Ap_sub.data(), 1, r_sub.data(), 1);

        // rsnew = r' * r;
        auto rsnew = cblas_ddot(count_rows, r_sub.data(), 1, r_sub.data(), 1);
        MPI_Allreduce(MPI_IN_PLACE, &rsnew, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // check convergence with overall residual, not rank-wise
        if (std::sqrt(rsnew) < m_tolerance)
            break;

        // compute ratio between overall residuals
        auto beta = rsnew / rsold;

        // p = r + (rsnew / rsold) * p;
        tmp_sub = r_sub;
        cblas_daxpy(count_rows, beta, p_sub.data(), 1, tmp_sub.data(), 1);
        p_sub = tmp_sub;

        // rsold = rsnew;
        rsold = rsnew;

        /// MPI collective communication: gather p_sub in a global vector p from all ranks to all ranks
        MPI_Allgatherv(&p_sub.front(), count_rows, MPI_DOUBLE,
                       &p.front(), num_rows, start_rows, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    /// MPI: construct the solution from x_sub to x by stacking together the x_sub on prank 0
    MPI_Gatherv(&x_sub.front(), count_rows, MPI_DOUBLE,
                &x.front(), num_rows, start_rows, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (DEBUG && prank == 0) {
	 std::fill_n(r.begin(), r.size(), 0.);
	 cblas_dgemv(CblasRowMajor, CblasNoTrans, m_m, m_n, 1., m_A.data(), m_n,
	       x.data(), 1, 0., r.data(), 1);
	 cblas_daxpy(m_n, -1., m_b.data(), 1, r.data(), 1);
	 auto res = std::sqrt(cblas_ddot(m_n, r.data(), 1, r.data(), 1)) /
	       std::sqrt(cblas_ddot(m_n, m_b.data(), 1, m_b.data(), 1));
	 auto nx = std::sqrt(cblas_ddot(m_n, x.data(), 1, x.data(), 1));
	 std::cout << "\t[STEP " << k << "] residual = " << std::scientific
	 << std::sqrt(rsold) << ", ||x|| = " << nx  << ", ||Ax - b||/||b|| = " << res << std::endl;
    }

}


void CGSolver::generate_lap2d_matrix(int size)
{
   /**
   * Generates a Laplacian 2d matrix of user-defined size, sets it to be the member m_A. The diagonal elements are set
   * to be equal to 4 on the main diagonal, and off-diagonal entries  are set to -1 with a frequency of sqrt(size)
   *
   * @param size size of the desired matrix
   * @return void
   */

    m_A.resize(size, size);
    m_m = size;
    m_n = size;
    m_maxIter = size;

    // approximate square root of the problem size
    int inc = (int) floor(sqrt(size));

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            m_A(i, j) = 0;
        }
        if (i > inc) m_A(i, i-1-inc) = -1;
        if (i > 0) m_A(i, i-1) = -1;
        m_A(i, i) = 4;
        if (i < size-1) m_A(i, i+1) = -1;
        if (i < size-1-inc) m_A(i, i+1+inc) = -1;
    }

}


void CGSolver::read_matrix(const std::string & filename)
{
    /**
   * Read a matrix from file
   *
   * @param filename filename
   * @return void
   */

    m_A.read(filename);

}

void CGSolver::set_max_iter(int maxIter)
{

    /**
   * Set maximum number of iterations for conjugate gradient, used for weak scaling experiments
   *
   * @param maxIter maximum number of iterations
   * @return void
   */

    m_maxIter = maxIter;

}

Matrix CGSolver::get_submatrix(Matrix A, int N_loc, int start_m)
{

    /**
    * Get only a subset of the rows of m_A for splitting the computation among threads
    *
    * @param A full matrix of size m x n
    * @param N_loc number of local rows
    * @param start_m first row to be considered
    * @return a submatrix of size N_loc x n
    */

    Matrix submatrix;
    submatrix.resize(N_loc, A.n());
    for (int i = 0; i < N_loc; i++) {
        for (int j = 0; j < A.n(); j++) {
            submatrix(i, j) = A(i + start_m, j);
        }
    }
    return submatrix;

}

void CGSolver::init_source_term(double h)
{

    /**
    * Initialization of source term
    *
    * @param h step size
    * @return void
    */

    m_b.resize(m_n);
    for (int i = 0; i < m_n; i++) {
        m_b[i] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) *
                 std::sin(10. * M_PI * i * h);
    }

}

void CGSolver::partition_matrix(int N, int psize, int start_rows[], int num_rows[])
{
    /**
    * Partition matrix into ranks for MPI interface
    *
    * @param N number of rows of the matrix
    * @param psize number of processors
    * @param start_rows to be filled with first row of the submatrix of each rank
    * @param num_rows to be filled with number of rows of the submatrix of each rank
    * @return void
    */

    if (psize == 1)
    {
        start_rows[0] = 0;
        num_rows[0] = N;
    }
    else
    {
        int N_loc = N / psize;
        start_rows[0] = 0;
        num_rows[0] = N_loc;
        int i0 = N_loc;
        for(int prank = 1; prank < psize-1; prank++)
        {
            start_rows[prank] = i0;
            num_rows[prank] = N_loc;
            i0 += N_loc;
        }
        start_rows[psize-1] = i0;
        num_rows[psize-1] = N - i0;
    }
}

