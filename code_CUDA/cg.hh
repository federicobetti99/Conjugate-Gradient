#include <cblas.h>
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

    /// get submatrix for parallel computation
    Matrix get_submatrix(int N_loc, int start_m);    

    /// get subvector for parallel computation
    std::vector<double> get_subvector(std::vector<double>& arr, int N_loc, int start_m);

    /// solve linear system with iterative CG
    void kerneled_solve(double *x, dim3 block_size);

    std::tuple<double, bool> cg_step_kernel(double* Ap, double* p, double* r, double* x,
                                  double rsold, dim3 grid_size, dim3 block_size);

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
