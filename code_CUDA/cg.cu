/* -------------------------------------------------------------------------- */
#include "cg.hh" 
/* -------------------------------------------------------------------------- */
#include <iostream>
#include <exception>
/* -------------------------------------------------------------------------- */

const double NEARZERO = 1.0e-14;

__global__ void MatVec(double* A, double* p, double* Ap, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double tmp = 0.;
    for (unsigned int j = 0; j < N; ++j) {
        tmp += A[i * N + j] * p[j];
    }
    Ap[i] = tmp;
}

__global__ void sumVec(double* a, double alpha, double* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    a[i] = a[i] + alpha * b[i];
}

__global__ void scalarCopy(double* a, double* b) {
    *a = *b;
}

__global__ void scalarDivide(double num, double denom, double* res)  {
    *res = num / denom;
}  

void CGSolver::kerneled_solve(double* x, dim3 block_size) {
    double *r, *p, *Ap, *tmp;
    cudaMallocManaged(&r, m_n * sizeof(double));
    cudaMallocManaged(&p, m_n * sizeof(double));
    cudaMallocManaged(&Ap, m_n * sizeof(double));
    cudaMallocManaged(&tmp, m_n * sizeof(double));
 
    for (int i = 0; i < m_n; i++) Ap[i] = 0.;

    double *conj, *rsnew, *rsold, *alpha, *beta;
    cudaMallocManaged(&conj, sizeof(double));
    cudaMallocManaged(&rsnew, sizeof(double));
    cudaMallocManaged(&rsold, sizeof(double));
    cudaMallocManaged(&alpha, sizeof(double));
    cudaMallocManaged(&beta, sizeof(double));

    dim3 grid_size;
    grid_size.x = m_m/block_size.x;
    grid_size.y = 1;

    cublasHandle_t h;
    cublasCreate(&h);

    // r = b - A * x;
    MatVec<<<grid_size, block_size>>>(m_A.data(), x, Ap, m_n);
    cudaDeviceSynchronize();
    r = m_b;
    sumVec<<<grid_size, block_size>>>(r, -1., Ap);

    // p = r
    p = r;
    
    // rsold = r' * r;
    cublasDdot(h, m_n, r, 1, p, 1, rsold);

    // for i = 1:length(b)
    int k = 0;
    for (; k < m_n; ++k) {

        // Ap = A * p;
        MatVec<<<grid_size, block_size>>>(m_A.data(), p, Ap, m_n);        

        // alpha = rsold / (p' * Ap);
        cublasDdot(h, m_n, p, 1, Ap, 1, conj);
        scalarDivide<<<1, 1>>>(*rsold, std::max(*conj, *rsold * NEARZERO), alpha);
        
        // x = x + alpha * p;
        sumVec<<<grid_size, block_size>>>(x, *alpha, p);

        // r = r - alpha * Ap;
        sumVec<<<grid_size, block_size>>>(r, -*alpha, Ap);

        // rsnew = r' * r;
        cublasDdot(h, m_n, r, 1, r, 1, rsnew);
        cudaDeviceSynchronize();
        
        if (std::sqrt(*rsnew) < m_tolerance) break; // Convergence test
            
        // p = r + (rsnew / rsold) * p;
        scalarDivide<<<1, 1>>>(*rsnew, *rsold, beta);
        sumVec<<<grid_size, block_size>>>(p, *beta-1., p);
        sumVec<<<grid_size, block_size>>>(p, 1., r);
        
        scalarCopy<<<1, 1>>>(rsold, rsnew);
        std::cout << "\t[STEP " << k << "] residual = " << std::scientific << std::sqrt(*rsold) << std::endl;
    }
   
    cudaFree(&r);
    cudaFree(&tmp);
    cudaFree(&p);
    cudaFree(&Ap);

    cublasDestroy(h);
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
  cudaMallocManaged(&m_b, m_n * sizeof(double));
  for (int i = 0; i < m_n; i++) {
    m_b[i] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) *
             std::sin(10. * M_PI * i * h);
  }
}
