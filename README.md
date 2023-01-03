# Parallel and High Performance Computing Project

This repository contains the Conjugate Gradient
implementation with MPI Interface and CUDA kernels. The
project is carried out in the context of the
MATH-454 Parallel and High Performance Computing course at EPFL.

## Repository Description
- `code` - Conjugate Gradient implementation
  - `CUDA` - CUDA implementation
  - `MPI` - MPI implementation
- `figures` - Figures of the results
- `results` - Results for weak/strong scaling in MPI, runtime in CUDA
- `plots.ipynb` - Notebook for reproducibility of the results 

## Reproducibility of the results
To reproduce the obtained results, two files `cg.run` are available in the folders
`code/CUDA` and `code/MPI`. Running these files will fill the _.txt_ files
in the `results` folder. After this operation is completed, run the notebook
`plots.ipynb` to visualize the obtained results and reproduce the plots
shown in the report.

### MPI implementation details
A row-wise decomposition of the domain is followed as a parallelization
approach for the conjugated gradient to tackle the massive time spent doing
matrix vector products. This means that every rank works on a subpart of the rows of the matrix
and has a copy of the vector solution. To check the dependence of the problem
size on the achievable speedup, a function `generate_lap_2d_matrix` is implemented
for the construction of a Laplacian matrix of arbitrary size. This allows to
solve a similar problem to the original one but for different sizes. The command to launch
the execution should be of the following form
```
srun -n 2 ./cgsolver N ../results/strong_scaling.txt
```
e.g. for 2 processors and a matrix of size N. The third parameter is the output file where to write the running time
and simulation statistics. For weak scaling experiments, where the number of iterations needs to be fixed for a fair comparison,
the number of iterations should be set as the last command parameter, i.e.
```
srun -n 2 ./cgsolver N ../results/strong_scaling.txt 200
```
Note that if such a parameter is not passed, the maximum number of iterations is N as by
construction of the conjugated gradient we know convergence in exact arithmetics occurs in such a number of iterations.

### CUDA implementation details
The main bottleneck being the matrix vector products, the main focus is dedicated
again to the parallelization of such an operation. The file `cg.run` provided in the
folder `code_CUDA` proceeds as follows: first, it shows a comparison between
a naive CUDA kernels (one thread per row VS one thread per column, the latter exploiting
symmetry of the matrix to favour coalesced memory access) for the matrix 
vector product computation. As the latter shows much better results, this was
the starting topology for the grid. To handle the low GPU occupancy achieved
by having one thread handling N elements of the matrix, being N the problem size,
the column-wise computation is split among threads and atomic operations are used
to reduce the results without conflicts. The second portion of `cg.run` runs
the best kernel for different number of threads and block widths and stores the results
in the corresponding file. For these experiments, the provided matrix `lap2d_5pt_n100.mtx`
is always used. The general syntax to launch an execution reads as follows:
```
srun ./cgsolver lap2d_5pt_n100.mtx NUM_THREADS BLOCK_WIDTH true ../results/strong_scaling.txt
```
where NUM_THREADS is the number of threads per block, BLOCK_WIDTH is the number
of elements handled by every thread and the bool `true` indicates that the
efficient kernel with coalesced memory access is used. To use the inefficient kernel,
please set such a variable to `false` in the above. Note that for BLOCK_WIDTH equal to the problem size,
the two approaches of one thread per row (`false`) and one thread per column (`true`) are recovered.

## Authors
- Federico Betti
