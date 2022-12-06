#include "matrix.hh"
#include "matrix_coo.hh"
#include <cblas.h>
#include <string>
#include <vector>
#include <tuple>
#include <cuda_runtime.h> 

#ifndef __CG_HH__
#define __CG_HH__


class Solver {
public:
    /// read matrix from .mtx file
    virtual void read_matrix(const std::string & filename) = 0;

    /// initialize source term
    void init_source_term(double h);

    /// parallel solver for CG
    virtual void solve(double *x, dim3 block_size) = 0;

    /// initialize size of the matrix (in this case m = n)
    inline int m() const { return m_m; }
    inline int n() const { return m_n; }

    /// prescribe residual tolerance for ending of the algorithm
    void tolerance(double tolerance) { m_tolerance = tolerance; }

    virtual void kerneled_solve(double *x, dim3 block_size) = 0;

    virtual std::tuple<double, bool> cg_step_kernel(double* Ap, double* p, double* r, double *x,
                                                    double rsold, dim3 grid_size, dim3 block_size) = 0;

protected:
    /// initialize m and n
    int m_m{0};
    int m_n{0};

    /// right hand side
    double* m_b;

    /// residual tolerance
    double m_tolerance{1e-10};
};

class CGSolver : public Solver {
public:
    /// initialize solver
    CGSolver() = default;

    /// read matrix from .mtx file
    virtual void read_matrix(const std::string & filename);
   
    /// get submatrix for parallel computation
    Matrix get_submatrix(int N_loc, int start_m);    

    /// get subvector for parallel computation
    std::vector<double> get_subvector(std::vector<double>& arr, int N_loc, int start_m);

    /// solve linear system with iterative CG
    virtual void solve(double* x, dim3 block_size);

    /// solve linear system with iterative CG
    virtual void kerneled_solve(double *x, dim3 block_size);

    virtual std::tuple<double, bool> cg_step_kernel(double* Ap, double* p, double* r, double* x,
                                  double rsold, dim3 grid_size, dim3 block_size);

private:
    /// finite element matrix
    Matrix m_A;
};

class CGSolverSparse : public Solver {
public:
    /// initialize solver
    CGSolverSparse() = default;

    /// read matrix from .mtx file
    virtual void read_matrix(const std::string & filename);

private:
    /// finite element matrix
    MatrixCOO m_A;
};

#endif /* __CG_HH__ */
