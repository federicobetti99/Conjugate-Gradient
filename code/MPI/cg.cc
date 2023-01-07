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

    // define final residual
    std::vector<double> r(m_n);

    /// rank dependent variables
    // compute subpart of the rows of the matrix destined to prank
    int count_rows = num_rows[prank];
    int start_row = start_rows[prank];

    // initialize conjugated direction, residual and solution for current prank
    std::vector<double> Ap_sub(count_rows);
    std::vector<double> tmp_sub(count_rows);

    /// rank dependent variables
    // compute subparts of residual and get a copy of the solution on every rank
    std::vector<double> r_sub(&m_b[start_row], &m_b[start_row+count_rows]);
    std::vector<double> x_sub(&x[start_row], &x[start_row+count_rows]);

    /// compute residual
    // r = b - A * x;
    std::fill_n(Ap_sub.begin(), Ap_sub.size(), 0.);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, num_rows[prank], m_n, 1., m_A.data()+start_rows[prank], m_n,
                x.data(), 1, 0., Ap_sub.data(), 1);
    cblas_daxpy(r_sub.size(), -1., Ap_sub.data(), 1, r_sub.data(), 1);

    /// copy p_sub into r_sub and initialize overall p vector
    std::vector<double> p_sub = r_sub;
    std::vector<double> p(m_n);

    /// MPI: first gather outside the loop
    MPI_Allgatherv(&p_sub.front(), count_rows, MPI_DOUBLE,
                   &p.front(), num_rows, start_rows, MPI_DOUBLE, MPI_COMM_WORLD);

    /// MPI: compute residual rank-wise and reduce
    auto rsold = cblas_ddot(r_sub.size(), r_sub.data(), 1, r_sub.data(), 1);
    MPI_Allreduce(MPI_IN_PLACE, &rsold, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // for i = 1:length(b)
    int k = 0;
    for (; k < m_maxIter; ++k) {

        /// MPI: note that we need to gather p in the end to compute this matrix-vector product at every iteration
        // Ap = A * p;
        std::fill_n(Ap_sub.begin(), Ap_sub.size(), 0.);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, count_rows, m_n, 1., m_A.data()+start_rows[prank], m_n,
                    p.data(), 1, 0., Ap_sub.data(), 1);

        /// MPI: compute denominator for optimal step size rank-wise and reduce in-place
        // alpha = rsold / (p' * Ap);
        auto conj = cblas_ddot(p_sub.size(), p_sub.data(), 1, Ap_sub.data(), 1);
        MPI_Allreduce(MPI_IN_PLACE, &conj, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        conj = std::max(conj, rsold * NEARZERO);
        auto alpha = rsold / conj;

        /// MPI: compute update of x_sub rank-wise
        // x = x + alpha * p;
        cblas_daxpy(x_sub.size(), alpha, p_sub.data(), 1, x_sub.data(), 1);

        /// MPI: compute residual rank-wise
        // r = r - alpha * Ap;
        cblas_daxpy(r_sub.size(), -alpha, Ap_sub.data(), 1, r_sub.data(), 1);

        /// MPI: compute new residual norm rank-wise and reduce in-place
        // rsnew = r' * r;
        auto rsnew = cblas_ddot(r_sub.size(), r_sub.data(), 1, r_sub.data(), 1);
        MPI_Allreduce(MPI_IN_PLACE, &rsnew, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // check convergence with overall residual, not rank-wise
        if (std::sqrt(rsnew) < m_tolerance)
            break;

        // compute ratio between overall residuals
        auto beta = rsnew / rsold;

        /// MPI: update p rank-wise
        // p = r + (rsnew / rsold) * p;
        tmp_sub = r_sub;
        cblas_daxpy(p_sub.size(), beta, p_sub.data(), 1, tmp_sub.data(), 1);
        p_sub = tmp_sub;

        // rsold = rsnew;
        rsold = rsnew;

        /// MPI collective communication: gather p_sub in a global vector p from all ranks to all ranks
        MPI_Allgatherv(&p_sub.front(), count_rows, MPI_DOUBLE,
		       &p.front(), num_rows, start_rows, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    /// MPI: construct the solution from x_sub to x by stacking together the x_sub in precise order
    MPI_Gatherv(&x_sub.front(), num_rows[prank], MPI_DOUBLE,
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

int CGSolver::solve(std::vector<double> & x) {
    std::vector<double> r(m_n);
    std::vector<double> p(m_n);
    std::vector<double> Ap(m_n);
    std::vector<double> tmp(m_n);

    // MPI parameter computation!
    int prank, psize;
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);

    // Evenly distribute workload across processes even in "unlucky" cases
    std::vector<int> counts(psize, m_n/psize);
    for (int i = 0; i < m_n%psize; ++i) {
        counts[i]++;
    }
    std::vector<int> displacements(psize+1);
    for (int i = 0; i < psize; ++i) {
        displacements[i+1] = displacements[i] + counts[i];
    }
    int row_start = displacements[prank];
    int row_end = displacements[prank+1];

    second total_mult_duration = second::zero();

    // Define a helper for matrix-vector products
    auto multiply_mat_vector = [&] (const std::vector<double> & input, std::vector<double> & output) {
        auto t1 = clk::now();

        // multiply our submatrix
        cblas_dgemv(
                CblasRowMajor,
                CblasNoTrans,
                row_end - row_start,
                m_n,
                1.,
                // adjust matrix pointer for row_start
                m_A.data() + row_start * m_n,
                // "real" dimension of the matrix
                m_n,
                input.data(),
                1,
                0.,
                output.data() + row_start,
                1);

        // all gather to share the results! :)
        MPI_Allgatherv(
                // send
                MPI_IN_PLACE,
                // ignored params for send
                -1, MPI_DOUBLE,
                // recv buffer
                output.data(),
                // counts, displacements
                counts.data(), displacements.data(),
                MPI_DOUBLE,
                MPI_COMM_WORLD);

        auto t2 = clk::now();
        total_mult_duration += t2 - t1;
    };

    // r = b - A * x;
    multiply_mat_vector(x, Ap);

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
        multiply_mat_vector(p, Ap);

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
        multiply_mat_vector(x, r);
        cblas_daxpy(m_n, -1., m_b.data(), 1, r.data(), 1);
        auto res = std::sqrt(cblas_ddot(m_n, r.data(), 1, r.data(), 1)) /
                   std::sqrt(cblas_ddot(m_n, m_b.data(), 1, m_b.data(), 1));
        auto nx = std::sqrt(cblas_ddot(m_n, x.data(), 1, x.data(), 1));
        std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                  << std::sqrt(rsold) << ", ||x|| = " << nx
                  << ", ||Ax - b||/||b|| = " << res << std::endl;
    }

    if (DEBUG) {
        std::cout << "mul duration = " << total_mult_duration.count() << std::endl;
    }

    return std::min(m_n, k+1); // return number of steps
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
