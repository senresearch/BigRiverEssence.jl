# NMF reference fixtures (`generate_nmf_reference.jl`)

The `generate_nmf_reference.jl` is used in context of testing the outputs of `nmf` function of `BigRiverEssence.jl` with the outputs of the Julia implementation `LowRankModels.NNMF`. It produces the simulated data matrix and outputs of `LowRankModels.NNMF` which are used in `nmf_test.jl` to test similarity of outputs with `LowRankModels.NNMF`.

## It performs the following tasks:

- It loads the Julia package `LowRankModels` and fixes a random seed for reproducibility.
- It simulates two nonnegative matrices `Wtrue` and `Htrue`, multiplies them to create the exactly rank-three matrix `X`, and saves `X` as `X.csv`.
- It fits `LowRankModels.NNMF` to `X` with a fixed rank, convergence tolerances and maximum number of iterations.
- It writes the fitted nonnegative factors as `W.csv` and `H.csv`.
- It creates `meta.csv` containing the parameters the fixture was generated with.

Since NMF factors can be permuted and rescaled without changing their product, `nmf_test.jl` compares the reconstructed matrix `W * H` and reconstruction quality instead of comparing the raw factors directly.
