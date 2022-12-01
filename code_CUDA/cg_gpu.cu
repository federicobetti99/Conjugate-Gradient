/* -------------------------------------------------------------------------- */
#include "cg.hh"
#include "matrix.hh"
#include "matrix_coo.hh"
/* -------------------------------------------------------------------------- */
#include <iostream>
#include <exception>
/* -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- */
__global__ void matrix_vector_product(Matrix A, std::vector<double> & p, std::vector<double> & Ap) {
    int i = blockIdx.x;
    __shared__ float tile[A.n()];

    for (int j = 0; j < A.n(); ++j) {
	tile[j] = A(i, j);
    }

    __synchtreads();
    int j = threadIdx.x;
    Ap(j) += tile[j] * p(j); 
}

__global__ void vector_sum(std::vector<double> & a, double alpha, std::vector<double> & b) {
    int i = blockIdx.x;
    a[i] = a[i] + alpha * b[i];
}

void CGSolver::cg_step_kernel(std::vector<double> & Ap, std::vector<double> & p, std::vector<double> & x,
                              auto rsold, dim3 grid_size, dim3 block_size) {
    // Ap = A * p;
    std::fill_n(Ap.begin(), Ap.size(), 0.);
    matrix_vector_product<<<grid_size, block_size>>>(m_A, p, Ap);
    cudaDeviceSynchronize();

    // alpha = rsold / (p' * Ap);
    auto alpha = rsold / std::max(cblas_ddot(m_n, p.data(), 1, Ap.data(), 1),
                                  rsold * NEARZERO);

    // x = x + alpha * p;
    vector_sum<<<grid_size, block_size>>>(x, alpha, p);
    // r = r - alpha * Ap;
    vector_sum<<<grid_size, block_size>>>(r, -1.0 * alpha, p);

    cudaDeviceSynchronize();

    // rsnew = r' * r;
    auto rsnew = cblas_ddot(m_n, r.data(), 1, r.data(), 1);

    // if sqrt(rsnew) < 1e-10
    //   break;
    if (std::sqrt(rsnew) < m_tolerance)
        break; // Convergence test

    auto beta = rsnew / rsold;
    // p = r + (rsnew / rsold) * p;
    tmp = r;
    vector_sum<<<grid_size, block_size>>>(r, beta, p);
    cudaDeviceSynchronize();
    p = tmp;

    return rsnew;
}

