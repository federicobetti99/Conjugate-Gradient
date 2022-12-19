# Parallel and High Performance Computing Project

This repository contains the Conjugate Gradient
implementation with MPI Interface and CUDA kernels. The
project is carried out in the context of the
MATH-454 Parallel and High Performance Computing course at EPFL.

## Repository Description
- `code_CUDA` - Conjugate Gradient CUDA implementation
- `code_MPI`  - Conjugate Gradient MPI implementation
- `figures` - Figures of the results
- `results` - Results for weak/strong scaling in MPI, runtime in CUDA
- `plots.ipynb` - Notebook for reproducibility of the results 

## Reproducibility of the results
To reproduce the obtained results, two files `cg.run` are available in the folders
`code_CUDA` and `code_MPI`. Running these files will fill the _.txt_ files
in the `results` folder. After this operation is completed, run the notebook
`plots.ipynb` to visualize the obtained results and reproduce the plots
shown in the report.

## Authors
- Federico Betti
