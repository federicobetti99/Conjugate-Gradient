#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#ifndef __MATRIX_H_
#define __MATRIX_H_

class Matrix {
public:
  Matrix(int m = 0, int n = 0) : m_m(m), m_n(n) {
  	cudaMallocManaged(&m_a, m * n * sizeof(double));  
  }

  void init_problem(int m, int n) {
    m_m = m;
    m_n = n;
    cudaMallocManaged(&m_a, m_m * m_n * sizeof(double));
  }

  __host__ __device__ inline double & operator()(int i, int j) { return m_a[i * m_n + j]; }
  __host__ __device__ int m() const { return m_m; }
  __host__ __device__ int n() const { return m_n; }
  __host__ __device__ double* data() { return m_a; }

  void read(const std::string & filename);

private:
  int m_m{0};
  int m_n{0};
  double* m_a;
};

#endif // __MATRIX_H_
