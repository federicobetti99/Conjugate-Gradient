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

    std::cout << "Good after first matrix vector product" << std::endl;

    r = m_b;
    vector_sum<<<grid_size, block_size>>>(r, -1., Ap);
    // p = r;
    p = r;

    // rsold = r' * r;
    double* rsold;
    cudaMallocManaged(&rsold, sizeof(double));
    scalar_product<<<grid_size, block_size>>>(r, p, rsold);

    std::cout << "Good after first scalar product" << std::endl;

    // for i = 1:length(b)
    bool conv;
    double* rsnew;
    cudaMallocManaged(&rsnew, sizeof(double));
    int k = 0;
    for (; k < m_n; ++k) {
        // Ap = A * p;
        bool conv = false;
        matrix_vector_product<<<grid_size, block_size>>>(m_A, p, Ap);
        cudaDeviceSynchronize();

        std::cout << "First matrix vector product" << std::endl;

        // alpha = rsold / (p' * Ap);
        double* conj;
        cudaMallocManaged(&conj, sizeof(double));
        scalar_product<<<grid_size, block_size>>>(p, Ap, conj);

        std::cout << "First scalar product " << std::endl;

        cudaDeviceSynchronize();
        auto alpha = *rsold / std::max(*conj, *rsold * NEARZERO);

        // x = x + alpha * p;
        vector_sum<<<grid_size, block_size>>>(x, alpha, p);
        // r = r - alpha * Ap;
        vector_sum<<<grid_size, block_size>>>(r, -1.0 * alpha, p);
        cudaDeviceSynchronize();

        std::cout << "Vector sums ok " << std::endl;

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

    if (DEBUG) {
        matrix_vector_product<<<grid_size, block_size>>>(m_A, x, r);
        cudaDeviceSynchronize();
        vector_sum<<<grid_size, block_size>>>(r, -1.0, m_b);
        scalar_product<<<grid_size, block_size>>>(r, r, rsold);
        scalar_product<<<grid_size, block_size>>>(m_b, m_b, rsnew);
        auto res = *rsold / *rsnew;
        double* norm_x;
        cudaMallocManaged(&norm_x, sizeof(double));
        scalar_product<<<grid_size, block_size>>>(x, x, norm_x);
        std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                  << std::sqrt(*rsold) << ", ||x|| = " << *norm_x
                  << ", ||Ax - b||/||b|| = " << res << std::endl;
    }

    cudaFree(r);
    cudaFree(tmp);
    cudaFree(p);
    cudaFree(Ap);
}

std::tuple<double*, bool> CGSolver::cg_step_kernel(double* Ap, double* p, double* r, double* x,
                                                  double* rsold, dim3 grid_size, dim3 block_size) {

    return std::make_tuple(rsnew, conv);
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
