#include "cg.hh"

#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <iostream>

const double NEARZERO = 1.0e-14;
const bool DEBUG = true;

/*
    cg-solver solves the linear equation A*x = b where A is
    of size m x n

Code based on MATLAB code (from wikipedia ;-)  ):

function x = conj-grad(A, b, x)
    r = b - A * x;
    p = r;
    rsold = r' * r;

    for i = 1:length(b)
        Ap = A * p;
        alpha = rsold / (p' * Ap);
        x =A_sub.n()alpha * p;
        r = r - alphaA_sub;
        rsnew = rA_sub;
        if sqrt(rsnew) < tolerance
              break;
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    end
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

    /// rank dependent variables
    // compute subpart of the matrix destined to prank
    Matrix A_sub;
    if (psize > 1) A_sub = get_submatrix(m_A, num_rows[prank], start_rows[prank]);
    else A_sub = m_A;

    // initialize conjugated direction, residual and solution for current prank
    int N_loc = A_sub.m();
    std::vector<double> Ap_sub(N_loc);
    std::vector<double> tmp_sub(N_loc);

    /// rank dependent variables
    // compute subparts of residual and get a copy of the solution on every rank
    std::vector<double> r_sub(&m_b[start_rows[prank]], &m_b[start_rows[prank]+num_rows[prank]]);
    std::vector<double> x_sub(&x[0], &x[x.size()]);

    /// compute residual
    // r = b - A * x;
    std::fill_n(Ap_sub.begin(), Ap_sub.size(), 0.);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, N_loc, A_sub.n(), 1., A_sub.data(), x.size(),
                x.data(), 1, 0., Ap_sub.data(), 1);
    cblas_daxpy(r_sub.size(), -1., Ap_sub.data(), 1, r_sub.data(), 1);

    /// copy p_sub into r_sub and initialize overall p vector
    std::vector<double> p_sub = r_sub;
    std::vector<double> p(m_n);

    /// MPI: first gather
    MPI_Allgatherv(&p_sub.front(), num_rows[prank], MPI_DOUBLE,
                   &p.front(), num_rows, start_rows, MPI_DOUBLE, MPI_COMM_WORLD);

    /// MPI: compute residual rank-wise and reduce
    auto rsold = cblas_ddot(r_sub.size(), r_sub.data(), 1, r_sub.data(), 1);
    MPI_Allreduce(MPI_IN_PLACE, &rsold, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // for i = 1:length(b)
    int k = 0;
    for (; k < m_maxIter; ++k) {

        /// MPI: gather p in the end to compute this matrix-vector product at every iteration
        // Ap = A * p;
        std::fill_n(Ap_sub.begin(), Ap_sub.size(), 0.);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, N_loc, p.size(), 1., A_sub.data(), p.size(),
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
        MPI_Allgatherv(&p_sub.front(), num_rows[prank], MPI_DOUBLE,
		       &p.front(), num_rows, start_rows, MPI_DOUBLE, MPI_COMM_WORLD);

        if (DEBUG && prank == 0) {
            std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                      << std::sqrt(rsold) << std::endl;
        }
    }

    /// MPI: construct the solution from x_sub to x by stacking together the x_sub in precise order
    MPI_Gatherv(&x_sub.front(), num_rows[prank], MPI_DOUBLE,
		&x.front(), num_rows, start_rows, MPI_DOUBLE,
		0, MPI_COMM_WORLD);

}

void CGSolver::generate_lap2d_matrix(int size)
{
   /**
   * Generates a Laplacian 2d matrix of user-defined size, sets it to be the member m_A
   *
   * @param size size of the desired matrix
   * @return void
   */

    m_A.resize(size, size);
    m_m = size;
    m_n = size;
    m_maxIter = size;

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
