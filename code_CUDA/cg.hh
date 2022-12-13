#include <cblas.h>
#include "matrix.hh"
#include "matrix_coo.hh"
#include <string>
#include <vector>
#include <tuple>
#include <iostream>
#include <cuda_runtime.h>

#ifndef __CG_HH__
#define __CG_HH__

class CGSolver
{
public:
    /// initialize solver
    CGSolver() = default;

    /// read matrix from .mtx file
    void read_matrix(const std::string & filename);

    /// initialize source term
    void init_source_term(double h);

    inline int m() const { return m_m; }
    inline int n() const { return m_n; }

    /// solve linear system with iterative CG
    void kerneled_solve(double *x, dim3 block_size);

protected:
    /// initialize m and n
    int m_m{0};
    int m_n{0};

    /// right hand side
    double* m_b;

    /// residual tolerance
    double m_tolerance{1e-10};

private:
    /// finite element matrix
    Matrix m_A;
};

#endif /* __CG_HH__ */
