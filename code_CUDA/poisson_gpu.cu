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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // send i-th row of the matrix to the shared memory, should fit in

    // do scalar product between i-th row and P to get (Ap)_i

    // reconstruct Ap total vector inside

}