#include "cg.hh"
#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h> 

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

void CGSolver::solve(double *x, dim3 block_size) {
    this->kerneled_solve(x, block_size);
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

/// initialization of the source term b
void Solver::init_source_term(double h) {
    m_b.resize(m_n);
    for (int i = 0; i < m_n; i++) {
        m_b[i] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) *
             std::sin(10. * M_PI * i * h);
    }
}
