/* -------------------------------------------------------------------------- */
#include "cg.hh"
/* -------------------------------------------------------------------------- */
#include <iostream>
#include <exception>
/* -------------------------------------------------------------------------- */

const double NEARZERO = 1.0e-14;
const bool DEBUG = false;

__global__ void matrix_vector_product(Matrix A, double* p, double* Ap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    Ap[i] = 0.;
    for (unsigned int j = 0; j < A.n(); ++j) {
        Ap[i] += A(i, j) * p[j];
    }
    __syncthreads();
}

__global__ void vector_sum(double* a, double alpha, double* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    a[i] = a[i] + alpha * b[i];
    __syncthreads();
}

__global__ void scalar_product(double* a, double* b, double* result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    *result = 0.;
    *result += a[i] * b[i];
    __syncthreads();
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
    double* rsold;
    cudaMallocManaged(&rsold, sizeof(double));
    scalar_product<<<grid_size, block_size>>>(r, p, rsold);

    // for i = 1:length(b)
    int k = 0;
    for (; k < m_n; ++k) {
        // Ap = A * p
        matrix_vector_product<<<grid_size, block_size>>>(m_A, p, Ap);
        cudaDeviceSynchronize();

        // alpha = rsold / (p' * Ap);
        double* conj;
        cudaMallocManaged(&conj, sizeof(double));
        scalar_product<<<grid_size, block_size>>>(p, Ap, conj);

        cudaDeviceSynchronize();
        auto alpha = *rsold / std::max(*conj, *rsold * NEARZERO);

        // x = x + alpha * p;
        vector_sum<<<grid_size, block_size>>>(x, alpha, p);
        // r = r - alpha * Ap;
        vector_sum<<<grid_size, block_size>>>(r, -1.0 * alpha, p);
        cudaDeviceSynchronize();

        // rsnew = r' * r;
        double* rsnew;
        cudaMallocManaged(&rsnew, sizeof(double));
        scalar_product<<<grid_size, block_size>>>(r, r, rsnew);
        cudaDeviceSynchronize();

        if (std::sqrt(*rsnew) < m_tolerance)
            break; // Convergence test

        auto beta = *rsnew / *rsold;
        // p = r + (rsnew / rsold) * p
        vector_sum<<<grid_size, block_size>>>(p, beta, r);
        cudaDeviceSynchronize();

        // rsold = rsnew;
        rsold = rsnew;
    }

    cudaFree(r);
    cudaFree(tmp);
    cudaFree(p);
    cudaFree(Ap);
}

void CGSolver::read_matrix(const std::string & filename) {
  m_A.read(filename);
  m_m = m_A.m();
  m_n = m_A.n();
}

/*
Initialization of the source term b
*/
void CGSolver::init_source_term(double h) {
  cudaMallocManaged(&m_b, m_n*sizeof(double));
  for (int i = 0; i < m_n; i++) {
    m_b[i] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) *
             std::sin(10. * M_PI * i * h);
  }
}
