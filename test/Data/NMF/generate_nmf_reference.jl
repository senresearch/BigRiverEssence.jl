# generate_nmf_reference.jl — generate LowRankModels reference fixtures for nmf tests.
#
# Fits LowRankModels.NNMF on simulated data and writes the input and outputs as
# CSVs. nmf_test.jl loads these to compare BigRiverEssence.nmf against
# LowRankModels without importing LowRankModels at test time.
#
# Run from a REPL (or environment) that has LowRankModels available:
#     include("test/Data/NMF/generate_nmf_reference.jl")

using LowRankModels, Random, DelimitedFiles

const OUTDIR = @__DIR__            # write CSVs next to this script (test/Data/NMF)

# ---- parameters (single source of truth; meta.csv records them) -------------------
const seed     = 160
const N        = 40
const P        = 8
const K        = 3
const ABS_TOL  = 1e-12
const REL_TOL  = 1e-12
const MAX_ITER = 5000

Random.seed!(seed)

# ---- simulate an exactly rank-three nonnegative input -----------------------------
Wtrue = rand(N, K) .+ 0.2
Htrue = rand(K, P) .+ 0.2
X = Wtrue * Htrue

# ---- fit LowRankModels.NNMF -------------------------------------------------------
mod = LowRankModels.NNMF(;
	k = K,
	abs_tol = ABS_TOL,
	rel_tol = REL_TOL,
	max_iter = MAX_ITER,
)
W_lrm = LowRankModels.ScikitLearnBase.fit_transform!(mod, X)
H_lrm = mod.glrm.Y

# ---- write everything -------------------------------------------------------------
writedlm(joinpath(OUTDIR, "X.csv"), X, ',')
writedlm(joinpath(OUTDIR, "W.csv"), W_lrm, ',')
writedlm(joinpath(OUTDIR, "H.csv"), H_lrm, ',')

# meta.csv: parameters the fixtures were generated with (read back by the test)
writedlm(joinpath(OUTDIR, "meta.csv"),
	["seed" seed; "n" N; "p" P; "k" K; "abs_tol" ABS_TOL;
		"rel_tol" REL_TOL; "max_iter" MAX_ITER], ',')

println("Wrote nmf fixtures to ", OUTDIR)
