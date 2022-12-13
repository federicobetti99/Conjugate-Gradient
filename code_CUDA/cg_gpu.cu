/* -------------------------------------------------------------------------- */
#include "cg.hh"
#include "matrix.hh"
#include "matrix_coo.hh"
/* -------------------------------------------------------------------------- */
#include <cmath>
#include <algorithm>
#include <iostream>
#include <exception>
/* -------------------------------------------------------------------------- */

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


/*
Sparse version of the cg solver
*/

__global__ void matrix_vector_product(Matrix A, double* p, double* Ap) {
    int thread_index = threadIdx.x;
    int block_index  = blockIdx.x;
    int i = block_index * blockDim.x + thread_index;
    for (unsigned int j = 0; j < A.n(); ++j) {
        Ap[i] += A(i, j) * p[j];
    }
    __syncthreads();
}

__global__ void vector_sum(double* a, double alpha, double* b) {
    int thread_index = threadIdx.x;
    int block_index  = blockIdx.x;
    int i = block_index * blockDim.x + thread_index;
    a[i] = a[i] + alpha * b[i];
    __syncthreads();
}

__global__ void scalar_product(double * a, double * b, double result) {
    int thread_index = threadIdx.x;
    int block_index  = blockIdx.x;
    int i = block_index * blockDim.x + thread_index;
    result += a[i]*b[i];
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
    double* rsold = 0.;
    cudaMallocManaged(&rsold, sizeof(double));
    scalar_product<<<grid_size, block_size>>>(r, p, rsold);

    // for i = 1:length(b)
    bool conv;
    double rsnew = 0.;
    int k = 0;
    for (; k < m_n; ++k) {

        // Ap = A * p;
        bool conv = false;
        matrix_vector_product<<<grid_size, block_size>>>(m_A, p, Ap);
        cudaDeviceSynchronize();

        // alpha = rsold / (p' * Ap);
        double* conj = 0.;
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
        double* rsnew = 0.;
        cudaMallocManaged(&rsnew, sizeof(double));
        scalar_product<<<grid_size, block_size>>>(r, r, rsnew);
        cudaDeviceSynchronize();

        if (std::sqrt(*rsnew) < m_tolerance)
            break; // Convergence test

        auto beta = *rsnew / *rsold;
        // p = r + (rsnew / rsold) * p
        vector_sum<<<grid_size, block_size>>>(p, beta, r);
        cudaDeviceSynchronize();
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

void CGSolver::read_matrix(const std::string & filename) {
    m_A.read(filename);
    m_m = m_A.m();
    m_n = m_A.n();
}

/// initialization of the source term b
void CGSolver::init_source_term(double h) {
    cudaMallocManaged(&m_b, m_n * sizeof(double));
    for (int i = 0; i < m_n; i++) {
        m_b[i] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) *
                 std::sin(10. * M_PI * i * h);
    }
}