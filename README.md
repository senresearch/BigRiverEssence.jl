# BigRiverEssence.jl

BigRiverEssence.jl provides matrix decomposition and multivariate learning tools for high-dimensional data. It focuses on practical routines used in exploratory analysis, sparse modeling, supervised dimensionality reduction, and multi-block integration.

## Features

- Principal Component Analysis (PCA)
- Sparse Principal Component Analysis (SPCA)
- Penalized Matrix Decomposition (PMD)
- Canonical Correlation Analysis (CCA)
- Sparse Canonical Correlation Analysis (SCCA)
- Joint and Individual Variation Explained (JIVE)
- Partial Least Squares Discriminant Analysis (PLSDA)
- Sparse PLS-DA (SPLSDA)
- Partial Least Squares Kernel Regression (PLSkern)

## Installation

Install from the Julia package registry:

```julia
using Pkg
Pkg.add("BigRiverEssence")
```

Install the development version from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/senresearch/BigRiverEssence.jl", rev = "main")
```

## Quick Start

```julia
using BigRiverEssence
using Random

Random.seed!(1)
X = randn(200, 40)

# PCA example
m = pca(X; k = 6, method = :auto)

# Typical outputs depend on method; for PCA, loadings are available in `m.loadings`
@show size(m.loadings)
```

## Repository Layout

- `notebooks/`: Method-oriented notebooks (`pca.ipynb`, `cca.ipynb`, `jive.ipynb`, `pmd.ipynb`, `spc.ipynb`, `splsda.ipynb`, and others).
- `scripts/comprehensive_testing.jl`: Extensive cross-method correctness and performance sweep, including optional comparisons with R packages.
- `scripts/typetest.jl`: Type-specialization benchmark script.

## Benchmark And Validation Script

Run the comprehensive script from the project root:

```bash
julia --project scripts/comprehensive_testing.jl
```

Notes:

- The script runs large Julia benchmarks and correctness checks.
- Some sections use `RCall.jl` and optional R packages (`PMA`, `r.jive`, `mixOmics`, `microbenchmark`).
- R benchmarking is adaptively repeated and skipped for very large matrix sizes to avoid excessive runtime.

## Documentation

Development docs are published at:

- https://senresearch.github.io/BigRiverEssence.jl/dev

## Contributing

Contributions are welcome. Please open an issue for bug reports or feature proposals, and open a pull request for fixes and enhancements.

## License

This project is licensed under the GNU Affero General Public License v3.0. See `LICENSE`.

## References

1. Pearson, K. (1901). On Lines and Planes of Closest Fit to Systems of Points in Space.
2. Witten, D. M., Tibshirani, R., and Hastie, T. (2009). A Penalized Matrix Decomposition, with Applications to Sparse Principal Components and Canonical Correlation Analysis.
3. Weenink, D. (2003). Canonical Correlation Analysis.
4. Witten, D. M., and Tibshirani, R. (2009). Extensions of Sparse Canonical Correlation Analysis with Applications to Genomic Data.
5. Lock, E. F., Hoadley, K. A., Marron, J. S., and Nobel, A. B. (2013). Joint and Individual Variation Explained for Integrated Analysis of Multiple Data Types.
6. Perez-Enciso, M., and Tenenhaus, M. (2003). Prediction of clinical outcome with microarray data: a PLS-DA approach.
7. Le Cao, K.-A., Boitard, S., and Besse, P. (2011). Sparse PLS Discriminant Analysis for multiclass problems.
8. Le Cao, K.-A., Rossouw, D., Robert-Granie, C., and Besse, P. (2008). A Sparse PLS for Variable Selection when Integrating Omics Data.
9. Dayal, B. S., and MacGregor, J. F. (1997). Improved PLS Algorithms.

