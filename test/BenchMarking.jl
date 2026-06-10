



# Benchmarking my PCA with Jchemo PCA for svd method .....


using BenchmarkTools, Jchemo, Statistics
using LinearAlgebra, Random
Random.seed!(1234)

# load YOUR pca
include(joinpath(@__DIR__, "..", "src", "BigRiverSchneider.jl"))
#include("src/BRMB.jl")

using .BigRiverSchneider

# test matrix — pick a shape; this one is tall
n, p, k = 5000, 200, 15
X = randn(n, p)

#  our SVD path 
b_mine = @btime BigRiverSchneider.pca($X; k = $k, method = :svd);
# Run 1: 37.356 ms (77 allocations: 24.62 MiB)
# Run 2: 37.436 ms (77 allocations: 24.62 MiB)
# Run 3: 37.451 ms (77 allocations: 24.62 MiB)
# Run 4: 38.253 ms (77 allocations: 24.62 MiB)
# Run 5: 37.438 ms (77 allocations: 24.62 MiB)


#  Jchemo's SVD path 
b_jchemo = @btime pcasvd($X; nlv = $k);
# Run 1: 37.755 ms (124 allocations: 17.76 MiB)
# Run 2: 38.371 ms (124 allocations: 17.76 MiB)
# Run 3: 38.209 ms (124 allocations: 17.76 MiB)
# Run 4: 37.988 ms (124 allocations: 17.76 MiB)
# Run 5:  38.036 ms (124 allocations: 17.76 MiB)











# Benchmarking PMD with .....R 's PMD implementation in the PMA package.
# This is exactly the architecture mismatch that's causing the issue. Your Julia is native Apple Silicon (arm64), but your R is an Intel (x86_64) build. RCall can't bridge them 
# because it relies on shared libraries that are not compatible across architectures. To fix this, you would need to install an Apple Silicon version of R and the PMA package, 
# and then ensure that RCall is configured to use that version of R. Alternatively, you could run your Julia code in an x86_64 environment (like Rosetta 2) to match the architecture of your R installation,
# but this is less efficient than using native Apple Silicon versions of both.

# Checking wrt PCA 
@btime BigRiverSchneider.pmd($X; k = $k, c = 3.0, maxiter = 5);
# Run 1: 43.353 ms (31260 allocations: 164.86 MiB)
# Run 2: 43.623 ms (31260 allocations: 164.86 MiB)
# Run 3: 42.953 ms (31260 allocations: 164.86 MiB)
# Run 4: 44.169 ms (31260 allocations: 164.86 MiB)
# Run 5: 43.723 ms (31260 allocations: 164.86 MiB)



# Checking with R SPC with same X metrix
using BenchmarkTools, DelimitedFiles, Random, LinearAlgebra, Statistics
Random.seed!(1234)

n, p = 200, 30
X = randn(n, p)
writedlm("Xshared.csv", X, ',')
println("saved Xshared.csv")

c = sqrt(p)                       # √30 ≈ 5.48, max budget (no sparsity)

# correctness (one run)
m     = BigRiverSchneider.pmd(X; k = 1, c = c)
v_pmd = m.loadings[:, 1]
v_ord = svd(X .- mean(X, dims = 1)).V[:, 1]
println("pmd vs ordinary PCA : ", round(abs(dot(v_pmd, v_ord)), digits = 6))
# JULIA: pmd vs ordinary PCA : 1.0
# R: pmd vs ordinary PCA : 0.999542 
println("first 5 of v_pmd    : ", round.(v_pmd[1:5], digits = 4))
#  JULIA: first 5 of v_pmd    : [-0.2522, -0.0305,  0.3382,  0.2709,  0.0251]
#  R: first 5 of v_pmd    :     [0.2547,   0.0415, -0.3333, -0.2807, -0.0224 ]

# benchmark
println("\n--- timing ---")
@btime BigRiverSchneider.pmd($X; k = 1, c = $c);
# 127.417 μs (412 allocations: 317.17 KiB)
#Julia pmd: ~127 μs = 0.127 ms (median)
# R SPC: 5.45 ms (median)







# Benchmarking my PLSkern with Jchemo's plskern (both = Dayal & MacGregor algo #1)
# Comparing results AND speed, with OLS as the ground-truth anchor.
Random.seed!(1234)
n, p, nlv = 5000, 200, 15
X = randn(n, p)
y = randn(n)                      # single response (q = 1)

# --- RESULTS: compare regression coefficients B (uniquely determined, no sign ambiguity) ---

# mine (algo1)
m_mine        = BigRiverSchneider.plskern(X, y; nlv = nlv, method = :algo1)
B_mine, _     = BigRiverSchneider.plskerncoef(m_mine)

# Jchemo (build-then-fit! workflow)
mod_jc = Jchemo.plskern(; nlv = nlv)     # NOTE: scal defaults to false — matches our standardize=false
fit!(mod_jc, X, y)
B_jc   = coef(mod_jc).B

# OLS ground truth: with nlv = p, full PLS = OLS. But here nlv=15 < p=200, so PLS is a
# RANK-REDUCED regression, NOT equal to OLS. So we compare mine-vs-Jchemo directly,
# and separately confirm both equal OLS only at full rank (nlv = p) below.
println("PLS RESULTS  (nlv = $nlv)")
println("  max |B|  mine vs Jchemo   : ", round(maximum(abs.(B_mine .- B_jc)), digits = 10))
# max |B|  mine vs Jchemo   : 0.0

# full-rank check: at nlv = p, BOTH must equal OLS
m_full       = BigRiverSchneider.plskern(X, y; nlv = p, method = :algo1)
B_full, int_full = BigRiverSchneider.plskerncoef(m_full)
Xc    = X .- mean(X, dims = 1)
B_ols = Xc \ (y .- mean(y))
ŷ_mine = BigRiverSchneider.plskernpredict(m_full, X)
ŷ_ols  = mean(y) .+ Xc * B_ols
println("  max |pred| full-PLS vs OLS: ", round(maximum(abs.(vec(ŷ_mine) .- ŷ_ols)), digits = 10))
#  max |pred| full-PLS vs OLS: 0.0

mod_jc_full = Jchemo.plskern(; nlv = p); fit!(mod_jc_full, X, y)
ŷ_jc_full   = predict(mod_jc_full, X).pred
println("  max |pred| Jchemo vs OLS  : ", round(maximum(abs.(vec(ŷ_jc_full) .- ŷ_ols)), digits = 10))
#  max |pred| Jchemo vs OLS  : 0.0

# --- SPEED ---
println("\n--- timing ---")
print("mine (algo1) : "); @btime BigRiverSchneider.plskern($X, $y; nlv = $nlv, method = :algo1);
# mine (algo1) :   3.169 ms (756 allocations: 17.52 MiB)
print("mine (algo2) : "); @btime BigRiverSchneider.plskern($X, $y; nlv = $nlv, method = :algo2);
# mine (algo2) :   3.938 ms (759 allocations: 17.84 MiB)
print("Jchemo       : "); @btime (m = Jchemo.plskern(; nlv = $nlv); fit!(m, $X, $y));
# Jchemo       :   4.281 ms (1158 allocations: 8.69 MiB)



