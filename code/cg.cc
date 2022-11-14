#include "cg.hh"

#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <iostream>

const double NEARZERO = 1.0e-14;
const bool DEBUG = false;

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
        x = x + alpha * p;
        r = r - alpha * Ap;
        rsnew = r' * r;
        if sqrt(rsnew) < 1e-10
              break;
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    end
end

*/
void CGSolverSparse::solve(std::vector<double> & x) {
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

void CGSolver::read_matrix(const std::string & filename) {
  m_A.read(filename);
  m_m = m_A.m();
  m_n = m_A.n();
}

/*
Sparse version of the cg solver
*/
void CGSolver::solve(Matrix A_sub, std::vector<double> & b_sub,
                           std::vector<int> &start_rows,
                           std::vector<int> &offsets_lengths,
                           std::vector<double> & x) {

    int prank;
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);

    std::vector<double> p(m_n);
    std::vector<double> x_sub(A_sub.n());
    std::vector<double> r_sub(A_sub.n());
    std::vector<double> p_sub(A_sub.n());
    std::vector<double> Ap_sub(m_n);
    std::vector<double> tmp_sub(m_n);

    // r = b - A * x;
    A_sub.mat_vec(x, Ap_sub);
    r_sub = b_sub;
    cblas_daxpy(m_n, -1., Ap_sub.data(), 1, r_sub.data(), 1);

    // p = r;
    p_sub = r_sub;

    // rsold = r' * r;
    auto rsold = cblas_ddot(m_n, r_sub.data(), 1, r_sub.data(), 1);

    // for i = 1:length(b)
    int k = 0;
    for (; k < m_n; ++k) {

    // Ap = A * p;
    /// MPI: we need to gather p in the end to compute this matrix-vector product at every iteration
    A_sub.mat_vec(p, Ap_sub);

    // alpha = rsold / (p' * Ap);
    auto alpha = rsold / std::max(cblas_ddot(m_n, p_sub.data(), 1, Ap_sub.data(), 1),
                                  rsold * NEARZERO);

    /// MPI: reduce the coefficient computed on each submatrix with sum
    MPI_Allreduce(MPI_IN_PLACE, &alpha, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    // x = x + alpha * p;
    cblas_daxpy(m_n, alpha, p_sub.data(), 1, x_sub.data(), 1);

    // r = r - alpha * Ap;
    cblas_daxpy(m_n, -alpha, Ap_sub.data(), 1, r_sub.data(), 1);

    // rsnew = r' * r;
    auto rsnew = cblas_ddot(m_n, r_sub.data(), 1, r_sub.data(), 1);

    if (std::sqrt(rsnew) < m_tolerance)
      break; // Convergence test

    auto beta = rsnew / rsold;
    // p = r + (rsnew / rsold) * p;
    tmp_sub = r_sub;
    cblas_daxpy(m_n, beta, p_sub.data(), 1, tmp_sub.data(), 1);
    p_sub = tmp_sub;

    // rsold = rsnew;
    rsold = rsnew;

    MPI_Allgatherv(&p_sub.front(), offsets_lengths[prank],
                   MPI_DOUBLE, &p.front(),
                   offsets_lenghts, start_rows, MPI_DOUBLE, MPI_COMM_WORLD);

    if (DEBUG) {
      std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                << std::sqrt(rsold) << "\r" << std::flush;
    }

    }

    /// MPI: construct the solution from x_sub to x
    MPI_Gatherv(&x_sub.front(), offsets_lengths[prank],
                MPI_DOUBLE, &x.front(),
                offsets_lengths, start_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

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

Matrix CGSolver::get_submatrix(int N_loc, int start_m) {
    Matrix submatrix;
    submatrix.resize(N_loc, m_n);

    for (int i = 0; i < N_loc; i++) {
        for (int j = 0; j < m_n; j++) {
            submatrix[i][j] = m_A[i + start_m][j]
        }
    }

    return submatrix;
}

std::vector<double> CGSolver::get_subvector(int N_loc, int start_m) {
    std::vector<double> vec;
    vec.resize(N_loc);

    for (int i = 0; i < N_loc; i++) {
        vec[i] = m_b[start_m + i];
    }

    return vec;
}
