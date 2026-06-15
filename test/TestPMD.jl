
using BigRiverSchneider
using LinearAlgebra, Statistics, Random
Random.seed!(123456)


#  Data: 300 observations, 40 features, 3 hidden signals

n, p, r = 300, 40, 3  # 300 observations, 40 features, 3 hidden signals
X = randn(n, r) * randn(r, p) .+ 0.1 .* randn(n, p)
println("X is $n × $p\n")


# PMD with deflation (Section 3.1). Each component is computed on the deflated data matrix, so the u's are not guaranteed to be orthogonal.

# TEST 1 — correctness: at c = √p there's no sparsity, so the first sparse
# loading must equal the ordinary top PC (paper eq. 2.10), up to sign.
m_sp  = BigRiverSchneider.pmd(X; k = 1, c = sqrt(p))
v_sp  = m_sp.loadings[:, 1]
v_ord = svd(X .- mean(X, dims = 1)).V[:, 1]
println("TEST 1  |⟨v_ordinary, v_sparse⟩| = ",
        round(abs(dot(v_ord, v_sp)), digits = 6), "   (want ≈ 1.0)\n")  # the inner product of the ordinary and sparse loading vectors should be close to 1 in absolute value, indicating they are essentially the same vector up to sign.

# TEST 2 — sparsity appears as c shrinks
println("TEST 2  smaller c ⇒ fewer nonzero loadings")
for c in (sqrt(p), 4.0, 2.0, 1.2)    # as c decreases, we expect the number of nonzero loadings to decrease, showing that the solution is becoming sparser; this is a key property of the PMD method, where c controls the sparsity level of the solution.
    local m = BigRiverSchneider.pmd(X; k = 1, c = c)
    println("  c = $(round(c, digits = 2))  →  ",
            count(!iszero, m.loadings[:, 1]), " / $p features used")
end

# TEST 3 — multiple components, shapes
m3 = BigRiverSchneider.pmd(X; k = 4, c = 2.0)
println("\nTEST 3  k=4, c=2.0")
println("  loadings size     : ", size(m3.loadings))
println("  nonzeros / column : ", [count(!iszero, m3.loadings[:, j]) for j in 1:4])



# TEST 4 — orthogonal variant: the u's are forced to be orthogonal, so the score correlations should be near zero; 
# this is a more stringent test of the orthogonality constraint in the PMD method, where we expect that by forcing the u's to be orthogonal, 
# the resulting score correlations will be close to zero, indicating that the components are indeed capturing distinct sources of variation in the data.


mort = BigRiverSchneider.pmd_orth(X; k = 4, c = 2.0)
println("loadings size : ", size(mort.loadings))
println("nonzeros/col  : ", [count(!iszero, mort.loadings[:, j]) for j in 1:4])

Xc = X .- mean(X, dims = 1)

# scores from each method, then their correlation matrices
function offdiag_corr(scores)
    C = cor(scores)                      # correlations between component scores
    n = size(C, 1)
    maximum(abs(C[i, j]) for i in 1:n for j in 1:n if i != j)   # worst off-diagonal
end

T_pmd  = Xc * m3.loadings                # plain deflation (from Test 3)
T_orth = Xc * mort.loadings              # orthogonal variant
println("\nORTHOGONALITY  (max |off-diagonal score correlation|, lower = more orthogonal)")
println("  pmd      : ", round(offdiag_corr(T_pmd),  digits = 4))
println("  pmd_orth : ", round(offdiag_corr(T_orth), digits = 4))

# the real check: recover the u's and confirm they're mutually orthogonal.
# (pmd_orth doesn't store U, so recompute scores T = Xc*V and orthogonality
#  shows up as near-zero off-diagonal dot products among the u directions)



















# optimization test — full output comparison
using BigRiverSchneider, BenchmarkTools, Random, LinearAlgebra, Statistics
X = randn(500, 300)

# helper: compare two pcaStructures field by field
function compare_pmd(a, b; label="")
    println("--- $label ---")
    println("  mean      ‖diff‖     : ", norm(a.mean .- b.mean))
    println("  scale     ‖diff‖     : ", norm(a.scale .- b.scale))
    println("  loadings  ‖diff‖(abs): ", norm(abs.(a.loadings) .- abs.(b.loadings)))   # sign arbitrary
    println("  variances ‖diff‖     : ", norm(a.variances .- b.variances))
    println("  propOFvar ‖diff‖     : ", norm(a.propOFvar .- b.propOFvar))
    println("  loadings size match  : ", size(a.loadings) == size(b.loadings))
    # sparsity pattern: the SET of selected (nonzero) variables per component must match
    sel(M) = [Set(findall(!iszero, M[:, j])) for j in 1:size(M,2)]
    println("  sparsity pattern match: ", sel(a.loadings) == sel(b.loadings))
end

# seed identically before each call (random init in spca_component)
Random.seed!(1234); ref      = pmd(X; k=5)
Random.seed!(1234); opt      = pmd_opt(X; k=5)
Random.seed!(1234); ref_orth = pmd_orth(X; k=5)
Random.seed!(1234); opt_orth = pmd_orth_opt(X; k=5)

compare_pmd(ref, opt; label="pmd: orig vs opt")
compare_pmd(ref_orth, opt_orth; label="pmd_orth: orig vs opt")

# ---- also verify the transform/reconstruction round-trips match ----
println("\n--- transform & reconstruction ---")
sc_ref = pca_transform(ref, X)
sc_opt = pca_transform(opt, X)
println("  pmd scores ‖diff‖(abs)        : ", norm(abs.(sc_ref) .- abs.(sc_opt)))
rc_ref = pca_invtransform(ref, sc_ref)
rc_opt = pca_invtransform(opt, sc_opt)
println("  pmd reconstruction ‖diff‖     : ", norm(rc_ref .- rc_opt))

sc_ref_o = pca_transform(ref_orth, X)
sc_opt_o = pca_transform(opt_orth, X)
println("  pmd_orth scores ‖diff‖(abs)   : ", norm(abs.(sc_ref_o) .- abs.(sc_opt_o)))
rc_ref_o = pca_invtransform(ref_orth, sc_ref_o)
rc_opt_o = pca_invtransform(opt_orth, sc_opt_o)
println("  pmd_orth reconstruction ‖diff‖: ", norm(rc_ref_o .- rc_opt_o))

# ---- benchmark ----
println("\n=== pmd ===")
print("orig: "); @btime pmd($X; k=5);
print("opt : "); @btime pmd_opt($X; k=5);
println("=== pmd_orth ===")
print("orig: "); @btime pmd_orth($X; k=5);
print("opt : "); @btime pmd_orth_opt($X; k=5);
#=
--- pmd: orig vs opt ---
  mean      ‖diff‖     : 0.0
  scale     ‖diff‖     : 0.0
  loadings  ‖diff‖(abs): 3.035806350154647e-15
  variances ‖diff‖     : 3.845925372767128e-15
  propOFvar ‖diff‖     : 1.309687503460227e-17
  loadings size match  : true
  sparsity pattern match: true
--- pmd_orth: orig vs opt ---
  mean      ‖diff‖     : 0.0
  scale     ‖diff‖     : 0.0
  loadings  ‖diff‖(abs): 1.0674422605731333e-15
  variances ‖diff‖     : 3.1714313080323688e-15
  propOFvar ‖diff‖     : 1.0551910960100405e-17
  loadings size match  : true
  sparsity pattern match: true

--- transform & reconstruction ---
  pmd scores ‖diff‖(abs)        : 1.010448390320123e-13
  pmd reconstruction ‖diff‖     : 1.3245071170843573e-13
  pmd_orth scores ‖diff‖(abs)   : 3.835939221360121e-14
  pmd_orth reconstruction ‖diff‖: 5.322783051703354e-14

=== pmd ===
orig:   238.930 ms (926827 allocations: 783.45 MiB)
opt :   191.536 ms (139 allocations: 2.44 MiB)
=== pmd_orth ===
orig:   306.623 ms (1120253 allocations: 938.18 MiB)
opt :   166.059 ms (147 allocations: 1.30 MiB)
=#
