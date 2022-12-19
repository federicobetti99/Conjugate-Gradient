/* -------------------------------------------------------------------------- */
#include "cg.hh" 
/* -------------------------------------------------------------------------- */
#include <math.h>
#include <iostream>
#include <exception>
/* -------------------------------------------------------------------------- */

const double NEARZERO = 1.0e-14;
const bool DEBUG = true;
#define BLOCK_WIDTH 10

__global__ void MatMulKernel(const int N, double* A, double* p, double* Ap) {
    // get variables for loop
    int blockElt;
    if (blockIdx.x * BLOCK_WIDTH + BLOCK_WIDTH <= N) blockElt = BLOCK_WIDTH;
    else blockElt = N % BLOCK_WIDTH;   

    // summing variable
    double cSum = 0.;
    int threadyInd = blockIdx.y * blockDim.x + threadIdx.x;

    // make sure we are inside the matrix vertically
    if (threadyInd < N) {

        // go through the threads vertically and sum them into a variable
        for (int i = 0; i < blockElt; i++)
            cSum += p[blockIdx.x * BLOCK_WIDTH + i] * A[threadyInd * N + blockIdx.x * BLOCK_WIDTH + i];

        // atomic add these variables to the corresponding output index
        atomicAdd(Ap + threadyInd, cSum);
    }

}

__global__ void MatVec(int N, Matrix A, double* p, double* Ap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        for (unsigned int j = 0; j < N; ++j) {
            Ap[i] = Ap[i] + A(i, j) * p[j];
        }
    }
}

__global__ void sumVec(int N, double alpha, double* a, double beta, double* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) a[i] = alpha * a[i] + beta * b[i];
}

__global__ void fill(int N, double* a, double val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) a[i] = val;
}

__global__ void copy(int N, double* a, double* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) a[i] = b[i];
}

void CGSolver::kerneled_solve(double* x, dim3 block_size, std::string KERNEL_TYPE) {
    double *r;
    double *p;
    double *Ap;
    double *tmp;

    cudaMallocManaged(&r, m_n * sizeof(double));
    cudaMallocManaged(&p, m_n * sizeof(double));
    cudaMallocManaged(&Ap, m_n * sizeof(double));
    cudaMallocManaged(&tmp, m_n * sizeof(double));

    double conj, rsnew, rsold;
    double *conj_, *rsnew_, *rsold_;
    cudaMallocManaged(&conj_, sizeof(double));
    cudaMallocManaged(&rsnew_, sizeof(double));
    cudaMallocManaged(&rsold_, sizeof(double));

    double* A_test;
    cudaMallocManaged(&A_test, 64 * sizeof(double));
    for (int i = 0; i < 64; i++) {
        if (i % 2 == 0) A_test[i] = 2.;
        else A_test[i] = 1.;
    }
    double* x_test;
    cudaMallocManaged(&x_test, 8 * sizeof(double));
    for (int i = 0; i < 8; i++) x_test[i] = 0.5;
    double* result;
    cudaMallocManaged(&result, 8 * sizeof(double));
    for (int i = 0; i < 8; i++) result[i] = 0.;
    dim3 test_block_size(2);
    dim3 test_grid_size(8 / 4, 8 / 2);
    MatMulKernel<<<test_grid_size, test_block_size>>>(8, A_test, x_test, result);
    cudaDeviceSynchronize();
    for (int i = 0; i < 8; i++) std::cout << result[i] << std::endl;

    // define grid size for linear combination of vectors
    dim3 vec_grid_size((int) ceil(m_m / (double) block_size.x));

    // define grid size for matrix vector products, check on input is done in cg_main.cc
    dim3 matvec_grid_size;
    if (!strcmp(KERNEL_TYPE.c_str(), "NAIVE")) matvec_grid_size = vec_grid_size;
    else {
        int blockRows = (int) ceil(m_m / (double) block_size.x);
        int blockCols = (int) ceil(m_n / (double) BLOCK_WIDTH);
        dim3 matvec_grid_size(blockCols, blockRows);
    }
    
    // initialize cublas handle
    cublasHandle_t h;
    cublasCreate(&h);

    // initialize vectors
    fill<<<vec_grid_size, block_size>>>(m_n,  x, 0.0);
    fill<<<vec_grid_size, block_size>>>(m_n, Ap, 0.0);
    cudaDeviceSynchronize();    

    // r = b - A * x;
    if (!strcmp(KERNEL_TYPE.c_str(), "NAIVE")) MatVec<<<matvec_grid_size, block_size>>>(m_n, m_A, x, Ap);
    else MatMulKernel<<<matvec_grid_size, block_size>>>(m_n, m_A.data(), x, Ap);
    cudaDeviceSynchronize();

    copy<<<vec_grid_size, block_size>>>(m_n, r, m_b);
    sumVec<<<vec_grid_size, block_size>>>(m_n, 1., r, -1., Ap);

    // p = r
    copy<<<vec_grid_size, block_size>>>(m_n, p, r);
    
    // rsold = r' * r;
    cublasDdot(h, m_n, r, 1, p, 1, rsold_);
    cudaMemcpy(&rsold, rsold_, sizeof(double), cudaMemcpyDeviceToHost);

    // for i = 1:length(b)
    int k = 0;
    for (; k < m_n; ++k) {

        // Ap = A * p;
        fill<<<vec_grid_size, block_size>>>(m_n, Ap, 0.0);
        cudaDeviceSynchronize();

        if (!strcmp(KERNEL_TYPE.c_str(), "NAIVE")) MatVec<<<matvec_grid_size, block_size>>>(m_n, m_A, p, Ap);
        else MatMulKernel<<<matvec_grid_size, block_size>>>(m_n, m_A.data(), p, Ap);
        cudaDeviceSynchronize();

        // alpha = rsold / (p' * Ap);
        cublasDdot(h, m_n, p, 1, Ap, 1, conj_);
        cudaMemcpy(&conj, conj_, sizeof(double), cudaMemcpyDeviceToHost);
        double alpha = rsold / std::max(conj, rsold * NEARZERO);
        
        // x = x + alpha * p;
        sumVec<<<vec_grid_size, block_size>>>(m_n, 1., x, alpha, p);

        // r = r - alpha * Ap;
        sumVec<<<vec_grid_size, block_size>>>(m_n, 1., r, -alpha, Ap);

        // rsnew = r' * r;
        cublasDdot(h, m_n, r, 1, r, 1, rsnew_);
        cudaMemcpy(&rsnew, rsnew_, sizeof(double), cudaMemcpyDeviceToHost);

        // synchronize to be sure about computation of the residual
        cudaDeviceSynchronize();
        
        if (std::sqrt(rsnew) < m_tolerance) break; // Convergence test
            
        // p = r + (rsnew / rsold) * p;
        double beta = rsnew / rsold;
        sumVec<<<vec_grid_size, block_size>>>(m_n, beta, p, 1., r);

        // prepare next iteration and print statistics
        rsold = rsnew;
        if (DEBUG) std::cout << "\t[STEP " << k << "] residual = " << std::scientific << std::sqrt(rsold) << std::endl;
    }

    if (DEBUG) {
        fill<<<vec_grid_size, block_size>>>(m_n, r, 0.0);
        if (!strcmp(KERNEL_TYPE.c_str(), "NAIVE")) MatVec<<<matvec_grid_size, block_size>>>(m_n, m_A, x, r);
        else MatMulKernel<<<matvec_grid_size, block_size>>>(m_n, m_A.data(), x, r);
        cudaDeviceSynchronize();
        sumVec<<<vec_grid_size, block_size>>>(m_n, 1., r, -1., m_b);
        double* num_;
        double* denom_;
        cudaMallocManaged(&num_, sizeof(double));
        cudaMallocManaged(&denom_, sizeof(double));
        double num = 0.;
        double denom = 0.;
        cublasDdot(h, m_n, r, 1, r, 1, num_);
        cublasDdot(h, m_n, m_b, 1, m_b, 1, denom_);
        cudaMemcpy(&num, num_, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&denom, denom_, sizeof(double), cudaMemcpyDeviceToHost);
        auto res = num / denom;
        double* nx_;
        cudaMallocManaged(&nx_, sizeof(double));
        double nx = 0.;
        cublasDdot(h, m_n, x, 1, x, 1, nx_);
        cudaMemcpy(&nx, nx_, sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                  << std::sqrt(rsold) << ", ||x|| = " << std::sqrt(nx)
                  << ", ||Ax - b||/||b|| = " << std::sqrt(res) << std::endl;
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
