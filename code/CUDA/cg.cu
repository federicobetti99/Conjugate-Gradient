/* -------------------------------------------------------------------------- */
#include "cg.hh" 
/* -------------------------------------------------------------------------- */
#include <math.h>
#include <iostream>
#include <exception>
#include <map>
#include <cstring>
/* -------------------------------------------------------------------------- */

const double NEARZERO = 1.0e-14;
const bool DEBUG = true;

__global__ void MatVec(const int N, const int NUM_THREADS, const int BLOCK_WIDTH,
                        Matrix A, double* p, double* Ap)
{

    /**
    * Efficient kernel for matrix vector product, every thread takes care of the dot product between a subpart of a
    * row of A and the corresponding subpart of p, then atomicAdd from the same thread in every block is done.
    * Coalesced memory accesses are not favoured by using symmetry of A, unlike the kernel MatVecT. In particular,
    * every thread takes care of BLOCK_WIDTH elements of a row of A, and threads with the same idx in the block
    * are collaborating to the computation of the corresponding entry of Ap.
    *
    * @param N Size of the matrix (always assumed square)
    * @param BLOCK_WIDTH width of the block
    * @param BLOCK_HEIGHT height of the block
    * @param A matrix
    * @param p vector
    * @param Ap vector for the result of A*p
    * @return void
    */

    __shared__ int blockElt;
    __shared__ int blockxInd;
    __shared__ int blockyInd;

    if ((blockIdx.x + 1) * BLOCK_WIDTH <= N)
        blockElt = BLOCK_WIDTH;
    else blockElt = N % BLOCK_WIDTH;
    blockxInd = blockIdx.x * BLOCK_WIDTH;
    blockyInd = blockIdx.y * NUM_THREADS;

    // summing variable
    double cSum = 0.;
    int threadyInd = blockyInd + threadIdx.x;

    // make sure we are inside the array horizontally
    if (threadyInd < N) {

        // go through the threads vertically and sum them into a variable
        for (int i = 0; i < blockElt; i++)
            cSum += A(threadyInd, blockxInd + i) * p[blockxInd + i];

        atomicAdd(Ap + threadyInd, cSum);
    }

}

__global__ void MatVecT(const int N, const int NUM_THREADS, const int BLOCK_WIDTH,
                                Matrix A, double* p, double* Ap)
{

    /**
    * Efficient kernel for matrix vector product, every thread takes care of the dot product between a subpart of a
    * row of A and the corresponding subpart of p, then atomicAdd from the same thread in every block is done.
    * Coalesced memory accesses are favoured by exploiting symmetry of A. In particular, every thread takes care of
    * BLOCK_WIDTH elements of a row of A, and threads with the same idx in the block are collaborating to the
    * computation of the corresponding entry of Ap. This kernel provided overall the best results.
    *
    * @param N Size of the matrix (always assumed square)
    * @param BLOCK_WIDTH width of the block
    * @param BLOCK_HEIGHT height of the block
    * @param A matrix
    * @param p vector
    * @param Ap vector for the result of A*p
    * @return void
    */

    // define common variables to all the elements of the block
    __shared__ int blockElt;
    __shared__ int blockxInd;
    __shared__ int blockyInd;

    if ((blockIdx.y + 1) * BLOCK_WIDTH <= N)
        blockElt = BLOCK_WIDTH;
    else blockElt = N % BLOCK_WIDTH;
    blockxInd = blockIdx.x * NUM_THREADS;
    blockyInd = blockIdx.y * BLOCK_WIDTH;

    // summing variable
    double cSum = 0.;
    int threadxInd = blockxInd + threadIdx.x;

    // make sure we are inside the array horizontally
    if (threadxInd < N) {

        // go through the threads vertically and sum them into a variable
        for (int i = 0; i < blockElt; i++)
            cSum += A(blockyInd + i, threadxInd) * p[blockyInd + i];

        atomicAdd(Ap + threadxInd, cSum);
    }

}

__global__ void sumVec(int N, double alpha, double* a, double beta, double* b)
{

    /**
    * Simple kernel for summing of two vectors, here the optimization of the topology behind leaves space to much
    * less details, every thread takes care of summing one element of the two vectors
    *
    * @param N Size of the vectors a and b
    * @param alpha coefficient to multiply a
    * @param a vector to be summed premultiplied by alpha
    * @param beta coefficient to multiply b
    * @param b vector to be summed premultiplied by beta
    * @return void
    */

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) a[i] = alpha * a[i] + beta * b[i];

}

__global__ void fill(int N, double* a, double val)
{

    /**
    * Simple kernel to set all the elements of the vector a to value val
    *
    * @param N Size of the vector a
    * @param a vector to be filled
    * @param val value to fill in all the elements of a
    * @return void
    */

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) a[i] = val;

}

__global__ void copy(int N, double* a, double* b)
{

    /**
    * Simple kernel to copy the content of vector b into the content of vector a
    *
    * @param N Size of the vector a
    * @param a vector to be filled
    * @param b vector to be copied into a
    * @return void
    */

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) a[i] = b[i];

}

void CGSolver::solve(double* x, const int NUM_THREADS, const int BLOCK_WIDTH, const bool T)
{

    /**
    * Main function to solve the linear system Ax=b with conjugate gradient
    *
    * @param x initial guess, zero vector usually
    * @param BLOCK_WIDTH width of the block for CUDA kernels
    * @param NUM_THREADS number of threads per block
    * @param T true to use transposed kernel for matrix vector products, thus favouring coalesced memory access
    * @return void
    */

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

    // define grid size for linear combination of vectors
    dim3 block_size(NUM_THREADS);
    dim3 vec_grid_size(ceil(m_n / (double) NUM_THREADS));

    // grid size for matrix vector products
    dim3 matvec_grid_size;
    if (T) {
        // blocks are arranged vertically exploiting symmetry of A
        matvec_grid_size.x = ceil(m_n / (double) NUM_THREADS);
        matvec_grid_size.y = ceil(m_m / (double) BLOCK_WIDTH);
    }
    else {
        // blocks are arranged horizontally, not exploiting symmetry of A
        matvec_grid_size.x = ceil(m_n / (double) BLOCK_WIDTH);
        matvec_grid_size.y = ceil(m_m / (double) NUM_THREADS);
    }

    // initialize cublas handle
    cublasHandle_t h;
    cublasCreate(&h);

    // initialize vectors
    fill<<<vec_grid_size, block_size>>>(m_n,  x, 0.0);
    fill<<<vec_grid_size, block_size>>>(m_n, Ap, 0.0);

    if (T) MatVecT<<<matvec_grid_size, block_size>>>(m_n, NUM_THREADS, BLOCK_WIDTH, m_A, x, Ap);
    else MatVec<<<matvec_grid_size, block_size>>>(m_n, NUM_THREADS, BLOCK_WIDTH, m_A, x, Ap);
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
        if (T) MatVecT<<<matvec_grid_size, block_size>>>(m_n, NUM_THREADS, BLOCK_WIDTH, m_A, p, Ap);
        else MatVec<<<matvec_grid_size, block_size>>>(m_n, NUM_THREADS, BLOCK_WIDTH, m_A, p, Ap);
        cudaDeviceSynchronize();  // synchronize as topology changes between matrix vector products and other operations

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

        /// CUDA: synchronize to be sure about computation of the residual
        cudaDeviceSynchronize();
        
        if (std::sqrt(rsnew) < m_tolerance) break; // Convergence test
            
        // p = r + (rsnew / rsold) * p;
        double beta = rsnew / rsold;
        sumVec<<<vec_grid_size, block_size>>>(m_n, beta, p, 1., r);

        // prepare next iteration and print statistics
        rsold = rsnew;
    }

    if (DEBUG) {
       double* r;
       double nx;
       double* nx_;
       double nb;
       double* nb_;
       cudaMallocManaged(&nb_, sizeof(double));
       cudaMallocManaged(&nx_, sizeof(double));
       cudaMallocManaged(&r, m_n * sizeof(double));
       fill<<<vec_grid_size, block_size>>>(m_n, r, 0.0);  
       if (T) MatVecT<<<matvec_grid_size, block_size>>>(m_n, NUM_THREADS, BLOCK_WIDTH, m_A, x, Ap);
       else MatVec<<<matvec_grid_size, block_size>>>(m_n, NUM_THREADS, BLOCK_WIDTH, m_A, x, Ap);
       cudaDeviceSynchronize();
       copy<<<vec_grid_size, block_size>>>(m_n, r, m_b);
       sumVec<<<vec_grid_size, block_size>>>(m_n, 1.0, r, -1.0, Ap);
       cublasDdot(h, m_n, r, 1, r, 1, rsnew_);
       cudaMemcpy(&rsnew, rsnew_, sizeof(double), cudaMemcpyDeviceToHost);
       cublasDdot(h, m_n, x, 1, x, 1, nx_);
       cudaMemcpy(&nx, nx_, sizeof(double), cudaMemcpyDeviceToHost);
       cublasDdot(h, m_n, m_b, 1, m_b, 1, nb_);
       cudaMemcpy(&nb, nb_, sizeof(double), cudaMemcpyDeviceToHost);         
       std::cout << "\t[STEP " << k << "] residual = " << std::scientific
              << std::sqrt(rsold) << ", ||x|| = " << std::sqrt(nx)
              << ", ||Ax - b||/||b|| = " << std::sqrt(rsnew) / std::sqrt(nb) << std::endl;
    }

    cudaFree(&r);
    cudaFree(&tmp);
    cudaFree(&p);
    cudaFree(&Ap);

    cublasDestroy(h);

}

void CGSolver::read_matrix(const std::string & filename)
{

    /**
    * Read matrix from file and set problem size
    *
    * @param filename filename
    * @return void
    */

    m_A.read(filename);
    m_m = m_A.m();
    m_n = m_A.n();

}


void CGSolver::init_source_term(double h)
{

    /**
    * Initialization of source term
    *
    * @param h step size
    * @return void
    */

    cudaMallocManaged(&m_b, m_n * sizeof(double));
    for (int i = 0; i < m_n; i++) {
    m_b[i] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) *
             std::sin(10. * M_PI * i * h);
    }

}
