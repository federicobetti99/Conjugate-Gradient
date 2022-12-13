/* -------------------------------------------------------------------------- */
#include "matrix.hh"
#include "matrix_coo.hh"
/* -------------------------------------------------------------------------- */
#include <iostream>
#include <exception>
#include <cuda_runtime.h>
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
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
