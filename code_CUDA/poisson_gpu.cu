/* -------------------------------------------------------------------------- */
#include "cg.hh"
#include "matrix.hh"
#include "matrix_coo.hh"
#include "grid.hh"
/* -------------------------------------------------------------------------- */
#include <iostream>
#include <exception>
/* -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- */
__global__ void matrix_vector_product(Matrix A, std::vector<double> & p, std::vector<double> & Ap) {
    int i = blockIdx.x;
    __shared__ float tile(A.n(), 1);

    for (int j = 0; j < A.n(); ++j) {
	tile(j) = A(i, j);
    }

    __synchtreads();
    int j = threadIdx.x;
    Ap(j) += tile(j) * p(j); 
}


__global__ void vector_sum(std::vector<double> & a, double alpha, std::vector<double> &b) {
    int i = blockIdx.x;
    a[i] = a[i] + alpha * b[i];
}
