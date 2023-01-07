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

void CGSolver::serial_solve(std::vector<double> & x) {
    std::vector<double> r(m_n);
    std::vector<double> p(m_n);
    std::vector<double> Ap(m_n);
    std::vector<double> tmp(m_n);

    // r = b - A * x;
    std::fill_n(Ap.begin(), Ap.size(), 0.);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m_m, m_n, 1., m_A.data(), m_n,
                x.data(), 1, 0., Ap.data(), 1);

    r = m_b;
    cblas_daxpy(m_n, -1., Ap.data(), 1, r.data(), 1);

    // p = r;
    p = r;

    // rsold = r' * r;
    auto rsold = cblas_ddot(m_n, r.data(), 1, p.data(), 1);

    // for i = 1:length(b)
    int k = 0;
    for (; k < m_n; ++k) {
        // Ap = A * p;
        std::fill_n(Ap.begin(), Ap.size(), 0.);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, m_m, m_n, 1., m_A.data(), m_n,
                    p.data(), 1, 0., Ap.data(), 1);

        // alpha = rsold / (p' * Ap);
        auto alpha = rsold / std::max(cblas_ddot(m_n, p.data(), 1, Ap.data(), 1),
                                      rsold * NEARZERO);

        // x = x + alpha * p;
        cblas_daxpy(m_n, alpha, p.data(), 1, x.data(), 1);

        // r = r - alpha * Ap;
        cblas_daxpy(m_n, -alpha, Ap.data(), 1, r.data(), 1);

        // rsnew = r' * r;
        auto rsnew = cblas_ddot(m_n, r.data(), 1, r.data(), 1);

        // if sqrt(rsnew) < 1e-10
        //   break;
        if (std::sqrt(rsnew) < m_tolerance)
            break; // Convergence test

        auto beta = rsnew / rsold;
        // p = r + (rsnew / rsold) * p;
        tmp = r;
        cblas_daxpy(m_n, beta, p.data(), 1, tmp.data(), 1);
        p = tmp;

        // rsold = rsnew;
        rsold = rsnew;
        if (DEBUG) {
            std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                      << std::sqrt(rsold) << "\r" << std::flush;
        }
    }

    if (DEBUG) {
        std::fill_n(r.begin(), r.size(), 0.);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, m_m, m_n, 1., m_A.data(), m_n,
                    x.data(), 1, 0., r.data(), 1);
        cblas_daxpy(m_n, -1., m_b.data(), 1, r.data(), 1);
        auto res = std::sqrt(cblas_ddot(m_n, r.data(), 1, r.data(), 1)) /
                   std::sqrt(cblas_ddot(m_n, m_b.data(), 1, m_b.data(), 1));
        auto nx = std::sqrt(cblas_ddot(m_n, x.data(), 1, x.data(), 1));
        std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                  << std::sqrt(rsold) << ", ||x|| = " << nx
                  << ", ||Ax - b||/||b|| = " << res << std::endl;
    }
}

void CGSolver::solve(int start_rows[], int num_rows[], std::vector<double> & x)
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

    /// rank dependent variables
    // compute subpart of the rows of the matrix destined to prank
    int count_rows = num_rows[prank];
    int start_row = start_rows[prank];

    /// compute residual
    // r = b - A * x;
    std::fill_n(Ap.begin(), Ap.size(), 0.);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, count_rows, m_n, 1., m_A.data() + start_row * m_n, m_n,
                x.data(), 1, 0., Ap.data() + start_row, 1);
    MPI_Allgatherv(MPI_IN_PLACE, -1, MPI_DOUBLE, Ap.data(), num_rows, start_rows,
                   MPI_DOUBLE, MPI_COMM_WORLD);

    r = m_b;
    cblas_daxpy(r.size(), -1., Ap.data(), 1, r.data(), 1);

    /// copy p into r and initialize overall p vector
    p = r;

    /// compute residual
    auto rsold = cblas_ddot(r.size(), r.data(), 1, r.data(), 1);

    // for i = 1:length(b)
    int k = 0;
    for (; k < m_maxIter; ++k) {

        /// MPI: note that we need to gather p in the end to compute this matrix-vector product at every iteration
        // Ap = A * p;
        std::fill_n(Ap.begin(), Ap.size(), 0.);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, count_rows, m_n, 1., m_A.data() + start_row * m_n, m_n,
                    p.data(), 1, 0., Ap.data() + start_row, 1);
        MPI_Allgatherv(MPI_IN_PLACE, -1, MPI_DOUBLE, Ap.data(), num_rows, start_rows,
                       MPI_DOUBLE, MPI_COMM_WORLD);

        // alpha = rsold / (p' * Ap);
        auto alpha = rsold / std::max(cblas_ddot(p.size(), p.data(), 1, Ap.data(), 1), rsold * NEARZERO);

        // x = x + alpha * p;
        cblas_daxpy(m_n, alpha, p.data(), 1, x.data(), 1);

        // r = r - alpha * Ap;
        cblas_daxpy(m_n, -alpha, Ap.data(), 1, r.data(), 1);

        // rsnew = r' * r;
        auto rsnew = cblas_ddot(m_n, r.data(), 1, r.data(), 1);

        // check convergence with overall residual, not rank-wise
        if (std::sqrt(rsnew) < m_tolerance)
            break;

        // compute ratio between overall residuals
        auto beta = rsnew / rsold;

        // p = r + (rsnew / rsold) * p;
        tmp = r;
        cblas_daxpy(m_n, beta, p.data(), 1, tmp.data(), 1);
        p = tmp;

        // rsold = rsnew;
        rsold = rsnew;

    }

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
