/* -------------------------------------------------------------------------- */
#include "cg.hh"
#include "matrix.hh"
#include "matrix_coo.hh"
/* -------------------------------------------------------------------------- */
#include <iostream>
#include <exception>
/* -------------------------------------------------------------------------- */
const double NEARZERO = 1.0e-14;
const bool DEBUG = true;
/* -------------------------------------------------------------------------- */
__global__ void matrix_vector_product(Matrix A, double* p, double* Ap) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    Ap[j] = A(i, j) * p[j]; 
}

__global__ void vector_sum(double* a, double alpha, double* b) {
    int i = blockIdx.x;
    a[i] = a[i] + alpha * b[i];
}

__global__ void scalar_product(double * a, double * b, double result) {
    int i = blockIdx.x;
    result += a[i]*b[i];
}

void CGSolver::kerneled_solve(double *x, dim3 block_size) {
    double *r, *p, *Ap, *tmp;
    cudaMallocManaged(&r, m_n * sizeof(double));
    cudaMallocManaged(&p, m_n * sizeof(double));
    cudaMallocManaged(&Ap, m_n * sizeof(double));
    cudaMallocManaged(&tmp, m_n * sizeof(double));

    dim3 grid_size;
    grid_size.x = m_m/block_size.x;
    grid_size.y = m_n/block_size.y;

    // r = b - A * x;
    matrix_vector_product<<<grid_size, block_size>>>(m_A, x, Ap);

    r = m_b;
    vector_sum<<<grid_size, block_size>>>(r, -1., Ap);
    // p = r;
    p = r;

    // rsold = r' * r;
    double rsold = 0.;
    scalar_product<<<grid_size, block_size>>>(r, p, rsold);

    // for i = 1:length(b)
    bool conv;
    double rsnew = 0.;
    int k = 0;
    for (; k < m_n; ++k) {
        std::tie(rsnew, conv) = cg_step_kernel(Ap, p, x, r, rsold, grid_size, block_size);
        // rsold = rsnew;
        if (conv) break;
        rsold = rsnew;
    }

    if (DEBUG) {
        matrix_vector_product<<<grid_size, block_size>>>(m_A, x, r);
        cudaDeviceSynchronize();
        vector_sum<<<grid_size, block_size>>>(r, -1.0, m_b);
        scalar_product<<<grid_size, block_size>>>(r, r, rsold);
        scalar_product<<<grid_size, block_size>>>(m_b, m_b, rsnew);
        auto res = rsold/rsnew;
        double norm_x = 0.;
        scalar_product<<<grid_size, block_size>>>(x, x, norm_x);
        std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                  << std::sqrt(rsold) << ", ||x|| = " << norm_x
                  << ", ||Ax - b||/||b|| = " << res << std::endl;
    }

    cudaFree(&r);
    cudaFree(&tmp);
    cudaFree(&p);
    cudaFree(&Ap);
}

std::tuple<double, bool> CGSolver::cg_step_kernel(double* Ap, double* p, double* r, double* x,
                              double rsold, dim3 grid_size, dim3 block_size) {
    // Ap = A * p;
    bool conv = false;
    matrix_vector_product<<<grid_size, block_size>>>(m_A, p, Ap);
    cudaDeviceSynchronize();

    // alpha = rsold / (p' * Ap);
    double conj = 0.;
    scalar_product<<<grid_size, block_size>>>(p, Ap, conj);
    cudaDeviceSynchronize();
    auto alpha = rsold / std::max(conj, rsold * NEARZERO);

    // x = x + alpha * p;
    vector_sum<<<grid_size, block_size>>>(x, alpha, p);
    // r = r - alpha * Ap;
    vector_sum<<<grid_size, block_size>>>(r, -1.0 * alpha, p);
    cudaDeviceSynchronize();

    // rsnew = r' * r;
    double rsnew = 0.;
    scalar_product<<<grid_size, block_size>>>(r, r, rsnew);
    cudaDeviceSynchronize();

    // if sqrt(rsnew) < 1e-10
    //   break;
    if (std::sqrt(rsnew) < m_tolerance)
        conv = true; // Convergence test

    auto beta = rsnew / rsold;
    // p = r + (rsnew / rsold) * p
    vector_sum<<<grid_size, block_size>>>(p, beta, r);
    cudaDeviceSynchronize();

    return std::make_tuple(rsnew, conv);
}

