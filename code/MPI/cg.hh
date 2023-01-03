#include "matrix.hh"
#include "matrix_coo.hh"
#include <cblas.h>
#include <string>
#include <vector>
#include <mpi.h> 

#ifndef __CG_HH__
#define __CG_HH__

class CGSolver {
public:
    /// initialize solver
    CGSolver() = default;

    /// read matrix from .mtx file
    virtual void read_matrix(const std::string & filename);

    /// initialize source term
    void init_source_term(double h);

    /// generate Laplacian 2d matrix for weak scaling and strong scaling experiment
    virtual void generate_lap2d_matrix(int size);
   
    /// get submatrix for parallel computation
    Matrix get_submatrix(Matrix A, int N_loc, int start_m);

    /// implements conjugate gradient with MPI interface
    virtual void solve(int start_rows[], int num_rows[], std::vector<double> & x);

    /// fix maximum number of iterations for weak scaling experiments
    virtual void set_max_iter(int maxIter);

    /// initialize size of the matrix (in this case m = n)
    inline int m() const { return m_m; }
    inline int n() const { return m_n; }

    /// prescribe residual tolerance for ending of the algorithm
    void tolerance(double tolerance) { m_tolerance = tolerance; }

private:
    /// initialize m and n
    int m_m{0};
    int m_n{0};

    /// finite element matrix
    Matrix m_A;

    /// maximum number of iterations
    int m_maxIter;

    /// right hand side
    std::vector<double> m_b;

    /// residual tolerance
    double m_tolerance{1e-10};
};

#endif /* __CG_HH__ */
