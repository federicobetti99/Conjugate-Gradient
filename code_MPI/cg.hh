#include "matrix.hh"
#include "matrix_coo.hh"
#include <cblas.h>
#include <string>
#include <vector>
#include <mpi.h> 

#ifndef __CG_HH__
#define __CG_HH__


class Solver {
public:
    /// read matrix from .mtx file
    virtual void read_matrix(const std::string & filename) = 0;

    /// initialize source term
    void init_source_term(double h);

    /// serial solver for CG
    virtual void serial_solve(std::vector<double> & x) = 0;

    /// solve linear system with iterative CG
    virtual void solve(int start_rows[],
                       int num_rows[],
                       std::vector<double> & x) = 0;

    /// initialize size of the matrix (in this case m = n)
    inline int m() const { return m_m; }
    inline int n() const { return m_n; }

    /// prescribe residual tolerance for ending of the algorithm
    void tolerance(double tolerance) { m_tolerance = tolerance; }

protected:
    /// initialize m and n
    int m_m{0};
    int m_n{0};

    /// right hand side
    std::vector<double> m_b;

    /// residual tolerance
    double m_tolerance{1e-10};
};

class CGSolver : public Solver {
public:
    /// initialize solver
    CGSolver() = default;

    /// read matrix from .mtx file
    virtual void read_matrix(const std::string & filename);

    /// reduce matrix size for weak scaling experiments
    void reduce_problem(int N_sub);

    void set_problem_size();
   
    /// get submatrix for parallel computation
    Matrix get_submatrix(Matrix A, int N_loc, int start_m);    
    
    /// serial solver   
    virtual void serial_solve(std::vector<double> & x);    

    /// solve linear system with iterative CG
    virtual void solve(int start_rows[],
               	       int num_rows[],
                       std::vector<double> & x);

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

    /// get submatrix for parallel computation
    MatrixCOO get_submatrix(MatrixCOO A, int N_loc, int start_m);

    /// get subvector for parallel computation
    std::vector<double> get_subvector(std::vector<double>& arr, int N_loc, int start_m);

    /// solve linear system with iterative CG
    virtual void serial_solve(std::vector<double> & x);

    /// solve linear system with iterative CG
    virtual void solve(int start_rows[],
                       int num_rows[],
                       std::vector<double> & x);

private:
    /// finite element matrix
    MatrixCOO m_A;
};

#endif /* __CG_HH__ */
