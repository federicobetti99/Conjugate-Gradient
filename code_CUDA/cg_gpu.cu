/* -------------------------------------------------------------------------- */
#include "cg.hh"
#include "matrix.hh"
#include "matrix_coo.hh"
/* -------------------------------------------------------------------------- */
#include <iostream>
#include <exception>
/* -------------------------------------------------------------------------- */
const double NEARZERO = 1.0e-14;

/* -------------------------------------------------------------------------- */
__global__ void matrix_vector_product(Matrix A, double* p, double* Ap) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    // Ap[j] = A(i, j); // * p[j]; 
}

__global__ void vector_sum(double* a, double alpha, double* b) {
    int i = blockIdx.x;
    a[i] = a[i] + alpha * b[i];
}

__global__ void scalar_product(double * a, double * b, double result) {
    int i = blockIdx.x;
    result += a[i]*b[i];
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
    // p = r + (rsnew / rsold) * p;
    double * tmp = r;
    vector_sum<<<grid_size, block_size>>>(p, beta, tmp);
    cudaDeviceSynchronize();
    p = tmp;

    return std::make_tuple(rsnew, conv);
}

