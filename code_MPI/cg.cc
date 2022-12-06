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

/// initialization of the source term b
void Solver::init_source_term(double h) {
    m_b.resize(m_n);
    for (int i = 0; i < m_n; i++) {
        m_b[i] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) *
                 std::sin(10. * M_PI * i * h);
    }
}

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


void CGSolver::solve(int start_rows[],
                     int num_rows[],
                     std::vector<double> & x) {

    int prank;
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);

    /// rank dependent variables
    // compute subpart of the matrix destined to prank
    Matrix A_sub = this->get_submatrix(num_rows[prank], start_rows[prank]);

    // initialize conjugated direction, residual and solution for current prank
    int N_loc = A_sub.m();
    std::vector<double> Ap(N_loc);
    std::vector<double> tmp_sub(N_loc);

    /// rank dependent variables
    // compute subparts of solution and residual
    std::vector<double> r_sub = this->get_subvector(this->m_b, num_rows[prank], start_rows[prank]);
    std::vector<double> x_sub = this->get_subvector(x,         num_rows[prank], start_rows[prank]);

    /// compute residual
    // r = b - A * x;
    std::fill_n(Ap.begin(), Ap.size(), 0.);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, N_loc, A_sub.n(), 1., A_sub.data(), x.size(),
                x.data(), 1, 0., Ap.data(), 1);
    cblas_daxpy(r_sub.size(), -1., Ap.data(), 1, r_sub.data(), 1);

    /// copy p_sub into r_sub and initialize overall p vector
    std::vector<double> p_sub = r_sub;
    std::vector<double> p = this->m_b;

    /// MPI: compute residual rank-wise and reduce
    auto rsold = cblas_ddot(r_sub.size(), r_sub.data(), 1, r_sub.data(), 1);
    MPI_Allreduce(MPI_IN_PLACE, &rsold, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // for i = 1:length(b)
    int k = 0;
    for (; k < m_n; ++k) {

        /// MPI: gather p in the end to compute this matrix-vector product at every iteration
        // Ap = A * p;
        std::fill_n(Ap.begin(), Ap.size(), 0.);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, N_loc, p.size(), 1., A_sub.data(), p.size(),
                    p.data(), 1, 0., Ap.data(), 1);

        /// MPI: compute denominator for optimal step size rank-wise and reduce in-place
        // alpha = rsold / (p' * Ap);
        auto conj = std::max(cblas_ddot(p_sub.size(), p_sub.data(), 1, Ap.data(), 1), rsold * NEARZERO);
        MPI_Allreduce(MPI_IN_PLACE, &conj, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        auto alpha = rsold / conj;

        /// MPI: compute update of x_sub rank-wise
        // x = x + alpha * p;
        cblas_daxpy(x_sub.size(), alpha, p_sub.data(), 1, x_sub.data(), 1);

        /// MPI: compute residual rank-wise
        // r = r - alpha * Ap;
        cblas_daxpy(r_sub.size(), -alpha, Ap.data(), 1, r_sub.data(), 1);

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
    }

    /// MPI: construct the solution from x_sub to x by stacking together the x_sub in precise order
    MPI_Gatherv(&x_sub.front(), num_rows[prank], MPI_DOUBLE,
		&x.front(), num_rows, start_rows, MPI_DOUBLE,
		0, MPI_COMM_WORLD);

    // check overall residual norm at the end
    std::vector<double> r(m_A.m());

    if (DEBUG) {
        if (prank == 0) {
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
}


void CGSolver::read_matrix(const std::string & filename) {
  m_A.read(filename);
  m_m = m_A.m();
  m_n = m_A.n();
}

Matrix CGSolver::get_submatrix(int N_loc, int start_m) {
    Matrix submatrix;
    submatrix.resize(N_loc, this->m_n);
    for (int i = 0; i < N_loc; i++) {
        for (int j = 0; j < m_n; j++) {
            submatrix(i, j) = this->m_A(i + start_m, j);
        }
    }
    return submatrix;
}

std::vector<double> CGSolver::get_subvector(std::vector<double> & arr, int N_loc, int start_m) {
    std::vector<double> vector(N_loc);
    for (int i = 0; i < N_loc; i++) {
        vector[i] = arr[start_m + i];
    }
    return vector;
}

/*
Sparse version of the cg solver
*/

/// read the matrix from file
void CGSolverSparse::read_matrix(const std::string & filename) {
    m_A.read(filename);
    m_m = m_A.m();
    m_n = m_A.n();
}

void CGSolverSparse::serial_solve(std::vector<double> & x) {
    std::vector<double> r(m_n);
    std::vector<double> p(m_n);
    std::vector<double> Ap(m_n);
    std::vector<double> tmp(m_n);

    // r = b - A * x;
    m_A.mat_vec(x, Ap);
    r = m_b;
    cblas_daxpy(m_n, -1., Ap.data(), 1, r.data(), 1);

    // p = r, copy
    p = r;

    // rsold = r' * r;
    auto rsold = cblas_ddot(m_n, r.data(), 1, r.data(), 1);

    // for i = 1:length(b)
    int k = 0;
    for (; k < m_n; ++k) {
        // Ap = A * p;
        m_A.mat_vec(p, Ap);

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
                      << std::sqrt(rsold) << std::endl;
        }
    }

    if (DEBUG) {
        m_A.mat_vec(x, r);
        cblas_daxpy(m_n, -1., m_b.data(), 1, r.data(), 1);
        auto res = std::sqrt(cblas_ddot(m_n, r.data(), 1, r.data(), 1)) /
                   std::sqrt(cblas_ddot(m_n, m_b.data(), 1, m_b.data(), 1));
        auto nx = std::sqrt(cblas_ddot(m_n, x.data(), 1, x.data(), 1));
        std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                  << std::sqrt(rsold) << ", ||x|| = " << nx
                  << ", ||Ax - b||/||b|| = " << res << std::endl;
    }
}

void CGSolverSparse::solve(int start_rows[],
                           int num_rows[],
                           std::vector<double> & x) {

    int prank;
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);

    /// rank dependent variables
    // compute subpart of the matrix destined to prank
    MatrixCOO A_sub = this->get_submatrix(num_rows[prank], start_rows[prank]);

    // initialize conjugated direction, residual and solution for current prank
    std::vector<double> Ap(num_rows[prank]);
    std::vector<double> tmp_sub(num_rows[prank]);
   
    /// rank dependent variables
    // compute subparts of solution and residual
    std::vector<double> r_sub = this->get_subvector(this->m_b, num_rows[prank], start_rows[prank]);
    std::vector<double> x_sub = this->get_subvector(x,         num_rows[prank], start_rows[prank]);

    // r = b - A * x;
    A_sub.mat_vec(x, Ap);
    cblas_daxpy(r_sub.size(), -1., Ap.data(), 1, r_sub.data(), 1);

    /// copy p_sub into r_sub and initialize overall p vector
    std::vector<double> p_sub = r_sub;
    std::vector<double> p = this->m_b;

    // rsold = r' * r;
    auto rsold = cblas_ddot(r_sub.size(), r_sub.data(), 1, r_sub.data(), 1);
    MPI_Allreduce(MPI_IN_PLACE, &rsold, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // for i = 1:length(b)
    int k = 0;
    for (; k < m_n; ++k) {
        // Ap = A * p;
        A_sub.mat_vec(p, Ap);

        // alpha = rsold / (p' * Ap);
        auto conj = std::max(cblas_ddot(p_sub.size(), p_sub.data(), 1, Ap.data(), 1),
                                  rsold * NEARZERO);
        MPI_Allreduce(MPI_IN_PLACE, &conj, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        auto alpha = rsold / conj;

        // x = x + alpha * p;
        cblas_daxpy(x_sub.size(), alpha, p_sub.data(), 1, x_sub.data(), 1);

        // r = r - alpha * Ap;
        cblas_daxpy(r_sub.size(), -alpha, Ap.data(), 1, r_sub.data(), 1);

        // rsnew = r' * r;
        auto rsnew = cblas_ddot(r_sub.size(), r_sub.data(), 1, r_sub.data(), 1);
        MPI_Allreduce(MPI_IN_PLACE, &rsnew, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // if sqrt(rsnew) < 1e-10
        //   break;
        if (std::sqrt(rsnew) < m_tolerance)
            break; // Convergence test

        auto beta = rsnew / rsold;
        // p = r + (rsnew / rsold) * p;
        tmp_sub = r_sub;
        cblas_daxpy(p_sub.size(), beta, p_sub.data(), 1, tmp_sub.data(), 1);
        p_sub = tmp_sub;

        // rsold = rsnew;
        rsold = rsnew;
        if (DEBUG) {
            std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                      << std::sqrt(rsold) << std::endl;
        }

        /// MPI collective communication: gather p_sub in a global vector p from all ranks to all ranks
        MPI_Allgatherv(&p_sub.front(), num_rows[prank], MPI_DOUBLE,
                       &p.front(), num_rows, start_rows, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    /// MPI: construct the solution from x_sub to x by stacking together the x_sub in precise order
    MPI_Gatherv(&x_sub.front(), num_rows[prank], MPI_DOUBLE,
                &x.front(), num_rows, start_rows, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // if (DEBUG) {
    //    m_A.mat_vec(x, r_sub);
    //    cblas_daxpy(m_n, -1., m_b.data(), 1, r.data(), 1);
    //    auto res = std::sqrt(cblas_ddot(m_n, r.data(), 1, r.data(), 1)) /
    //               std::sqrt(cblas_ddot(m_n, m_b.data(), 1, m_b.data(), 1));
    //    auto nx = std::sqrt(cblas_ddot(m_n, x.data(), 1, x.data(), 1));
    //    std::cout << "\t[STEP " << k << "] residual = " << std::scientific
    //              << std::sqrt(rsold) << ", ||x|| = " << nx
    //              << ", ||Ax - b||/||b|| = " << res << std::endl;
    // }
}

MatrixCOO CGSolverSparse::get_submatrix(int N_loc, int start_m) {
    MatrixCOO submatrix;
    for (int z = 0; z < this->m_A.nz(); ++z) {
        auto i = this->m_A.irn[z];
        auto j = this->m_A.jcn[z];
        auto a_ = this->m_A.a[z];
        if (i >= start_m && i < start_m + N_loc) {
            submatrix.a.push_back(a_);
            submatrix.irn.push_back(i);
            submatrix.jcn.push_back(j);
        }
    }
    return submatrix;
} 

std::vector<double> CGSolverSparse::get_subvector(std::vector<double> &arr, int N_loc, int start_m) {
    std::vector<double> vector(N_loc); 
    for (int i = 0; i < N_loc; i++) {
        vector[i] = arr[start_m + i];
    }
    return vector;
}
