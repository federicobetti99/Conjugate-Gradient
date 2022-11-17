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
        r = r - alphaA_sub.n();
        rsnew = rA_sub.n();
        if sqrt(rsnew) <A_sub.n()0
              break;
        end
        p = r + (rsnew / rsold) * p;
        rsold = A_sub.n();
    end
end

*/

void CGSolver::solve(int prank,
                     int start_rows[],
                     int offsets_lengths[],
                     std::vector<double> & x) {

    /// global variables
    std::vector<double> p(m_n);
    std::vector<double> Ap_sub(m_n);
    std::vector<double> tmp_sub(m_n);

    /// rank dependent variables
    // compute subpart of the matrix destined to prank
    Matrix A_sub = this->get_submatrix(offsets_lengths[prank], start_rows[prank]);
    // initialize conjugated direction, residual and solution for current prank
    int N_loc = A_sub.m();
    std::vector<double> x_sub(N_loc);
    std::vector<double> r_sub(N_loc);
    std::vector<double> p_sub(N_loc);

    // r = b - A * x;
    std::fill_n(Ap_sub.begin(), Ap_sub.size(), 0.);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, N_loc, A_sub.n(), 1., A_sub.data(), x.size(),
                x.data(), 1, 0., Ap_sub.data(), 1);
    r_sub = this->m_b;

    cblas_daxpy(r_sub.size(), -1., Ap_sub.data(), 1, r_sub.data(), 1);

    // p = r;
    p_sub = r_sub;

    /// MPI: First call to gather collective communication
    MPI_Allgatherv(&p_sub.front(), offsets_lengths[prank], MPI_DOUBLE,
                   &p.front(), offsets_lengths, start_rows, MPI_DOUBLE, MPI_COMM_WORLD);

    // rsold = r' * r;
    auto rsold = cblas_ddot(r_sub.size(), r_sub.data(), 1, r_sub.data(), 1);

    // for i = 1:length(b)
    int k = 0;
    for (; k < m_n; ++k) {

        /// MPI: we need to gather p in the end to compute this matrix-vector product at every iteration
        // Ap = A * p;
        std::fill_n(Ap_sub.begin(), Ap_sub.size(), 0.);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, A_sub.m(), N_loc, 1., A_sub.data(), p.size(),
                    p.data(), 1, 0., Ap_sub.data(), 1);

        // alpha = rsold / (p' * Ap);
        auto alpha = rsold / std::max(cblas_ddot(p_sub.size(), p_sub.data(), 1, Ap_sub.data(), 1),
                                      rsold * NEARZERO);

        // x = x + alpha * p;
        cblas_daxpy(p_sub.size(), alpha, p_sub.data(), 1, x_sub.data(), 1);

        // r = r - alpha * Ap;
        cblas_daxpy(r_sub.size(), -alpha, Ap_sub.data(), 1, r_sub.data(), 1);

        // rsnew = r' * r;
        auto rsnew = cblas_ddot(r_sub.size(), r_sub.data(), 1, r_sub.data(), 1);
        
        if (std::sqrt(rsnew) < m_tolerance)
            break; // Convergence test

        auto beta = rsnew / rsold;

        // p = r + (rsnew / rsold) * p;
        tmp_sub = r_sub;
        cblas_daxpy(p_sub.size(), beta, p_sub.data(), 1, tmp_sub.data(), 1);
        p_sub = tmp_sub;

        // rsold = rsnew;
        rsold = rsnew;

        /// collective communication: gather p_sub in a global vector p from all ranks to all ranks
        MPI_Allgatherv(&p_sub.front(), offsets_lengths[prank], MPI_DOUBLE,
		       &p.front(), offsets_lengths, start_rows, MPI_DOUBLE, MPI_COMM_WORLD);

        if (DEBUG) {
            if (prank == 0) {
                std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                          << std::sqrt(rsold) << "\r" << std::endl;
            }
        }
    }

    /// MPI: construct the solution from x_sub to x by stacking together the x_sub in precise order
    MPI_Gatherv(&x_sub.front(), offsets_lengths[prank], MPI_DOUBLE,
		&x.front(), offsets_lengths, start_rows, MPI_DOUBLE,
		0, MPI_COMM_WORLD);
}


void CGSolver::read_matrix(const std::string & filename) {
  m_A.read(filename);
  m_m = m_A.m();
  m_n = m_A.n();
}

Matrix CGSolver::get_submatrix(int N_loc, int start_m) {
    Matrix submatrix;
    submatrix.resize(N_loc, m_n);
    for (int i = 0; i < N_loc; i++) {
        for (int j = 0; j < m_n; j++) {
            submatrix(i, j) = m_A(i + start_m, j);
        }
    }
    return submatrix;
}

std::vector<double> CGSolver::get_subvector(int N_loc, int start_m) {
    std::vector<double> vec(N_loc);
    for (int i = 0; i < N_loc; i++) {
        vec[i] = m_b[start_m + i];
    }
    return vec;
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

/// initialization of the source term b
void Solver::init_source_term(double h) {
    m_b.resize(m_n);
    for (int i = 0; i < m_n; i++) {
        m_b[i] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) *
             std::sin(10. * M_PI * i * h);
    }
}

void CGSolverSparse::solve(std::vector<double> & x) {
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
                      << std::sqrt(rsold) << "\r" << std::flush;
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
