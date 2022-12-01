/* -------------------------------------------------------------------------- */
#include "cg.hh"
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


void CGSolver::kerneled_solve(std::vector<double> & x, dim3 block_size) {
    std::vector<double> r(m_n);
    std::vector<double> p(m_n);
    std::vector<double> Ap(m_n);
    std::vector<double> tmp(m_n);

    dim3 grid_size;
    grid_size.x = m_m/block_size.x;
    grid_size.y = m_n/block_size.y;

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

        // rsold = rsnew;
        rsold = rsnew;
    }

    if (DEBUG) {
        std::fill_n(r.begin(), r.size(), 0.);
        matrix_vector_product<<<grid_size, block_size>>>(m_A, x, r);
        cudaDeviceSynchronize();
        vector_sum<<<grid_size, block_size>>>(r, -1.0, m_b);
        auto res = std::sqrt(cblas_ddot(m_n, r.data(), 1, r.data(), 1)) /
                   std::sqrt(cblas_ddot(m_n, m_b.data(), 1, m_b.data(), 1));
        auto nx = std::sqrt(cblas_ddot(m_n, x.data(), 1, x.data(), 1));
        std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                  << std::sqrt(rsold) << ", ||x|| = " << nx
                  << ", ||Ax - b||/||b|| = " << res << std::endl;
    }
}
