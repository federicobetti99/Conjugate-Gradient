/* -------------------------------------------------------------------------- */
#include "cg.hh" 
/* -------------------------------------------------------------------------- */
#include <iostream>
#include <exception>
/* -------------------------------------------------------------------------- */

const double NEARZERO = 1.0e-14;

__global__ void matrix_vector_product(double* A, double* p, double* Ap, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double tmp = 0.;
    for (unsigned int j = 0; j < N; ++j) {
        tmp += A[i * N + j] * p[j];
    }
    Ap[i] = tmp;
}

__global__ void vector_sum(double* a, double alpha, double* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    a[i] += alpha * b[i];
}

__global__ void scalar_product(double* a, double* b, double* result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    result[i] = a[i] * b[i];
}

void CGSolver::kerneled_solve(double* x, dim3 block_size) {
    double *r, *p, *Ap, *tmp;
    cudaMallocManaged(&r, m_n * sizeof(double));
    cudaMallocManaged(&p, m_n * sizeof(double));
    cudaMallocManaged(&Ap, m_n * sizeof(double));
    cudaMallocManaged(&tmp, m_n * sizeof(double));
 
    for (int i = 0; i < m_n; i++) Ap[i] = 0.;
    for (int i = 0; i < m_n; i++) tmp[i] = 0.;

    dim3 grid_size;
    grid_size.x = m_m/block_size.x;
    grid_size.y = 1;

    double* conj;
    cudaMallocManaged(&conj, m_n * sizeof(double));
    for (int i = 0; i < m_n; i++) conj[i] = 0.;

    double* rsnew;
    cudaMallocManaged(&rsnew, m_n * sizeof(double));
    for (int i = 0; i < m_n; i++) rsnew[i] = 0.;

    // r = b - A * x;
    matrix_vector_product<<<grid_size, block_size>>>(m_A.data(), x, Ap, m_n);
    cudaDeviceSynchronize();

    r = m_b;
    vector_sum<<<grid_size, block_size>>>(r, -1., Ap);
    cudaDeviceSynchronize();

    // p = r
    p = r;
    
    // rsold = r' * r;
    double* rsold;
    cudaMallocManaged(&rsold, m_n * sizeof(double));
    for (int i = 0; i < m_n; i++) rsold[i] = 0.;

    scalar_product<<<grid_size, block_size>>>(r, p, rsold);
    cudaDeviceSynchronize();
    for (int i = 1; i < m_n; i++) *rsold += rsold[i];

    // for i = 1:length(b)
    int k = 0;
    for (; k < m_n; ++k) {

        // Ap = A * p
        matrix_vector_product<<<grid_size, block_size>>>(m_A.data(), p, Ap, m_n); 
        cudaDeviceSynchronize();        

        // alpha = rsold / (p' * Ap);
        scalar_product<<<grid_size, block_size>>>(p, Ap, conj);
        cudaDeviceSynchronize();
        for (int i = 1; i < m_n; i++) *conj += conj[i];
        double alpha = *rsold / std::max(*conj, *rsold * NEARZERO);
        
        // x = x + alpha * p;
        vector_sum<<<grid_size, block_size>>>(x, alpha, p);
        // r = r - alpha * Ap;
        vector_sum<<<grid_size, block_size>>>(r, -alpha, Ap);
        cudaDeviceSynchronize();

        // rsnew = r' * r;
        scalar_product<<<grid_size, block_size>>>(r, r, rsnew);
        cudaDeviceSynchronize();
        for (int i = 1; i < m_n; i++) *rsnew += rsnew[i]; 

        if (std::sqrt(*rsnew) < m_tolerance) break; // Convergence test
            
        double beta = *rsnew / *rsold;
        // p = r + (rsnew / rsold) * p
        tmp = r;
        vector_sum<<<grid_size, block_size>>>(tmp, beta, p);
        cudaDeviceSynchronize();
        p = tmp;

        *rsold = *rsnew;
        std::cout << "\t[STEP " << k << "] residual = " << std::scientific << std::sqrt(*rsold) << std::endl;
    }
   
    cudaFree(&r);
    cudaFree(&tmp);
    cudaFree(&p);
    cudaFree(&Ap);
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
