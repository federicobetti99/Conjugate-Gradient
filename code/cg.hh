#include "matrix.hh"
#include "matrix_coo.hh"
#include <cblas.h>
#include <string>
#include <vector>

#ifndef __CG_HH__
#define __CG_HH__


class Solver {
public:
    /// read matrix from .mtx file
    virtual void read_matrix(const std::string & filename) = 0;

    /// initialize source term
    void init_source_term(double h);

    /// solve linear system with iterative CG
    virtual void solve(std::vector<double> & x) = 0;

    /// initialize size of the matrix (m = n)
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

    /// solve linear system with iterative CG
    virtual void solve(std::vector<double> & x);

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

    /// solve linear system with iterative CG
    virtual void solve(std::vector<double> & x);

private:
    /// finite element matrix
    MatrixCOO m_A;
};

#endif /* __CG_HH__ */
