
using BigRiverSchneider   # this makes the exported names from BigRiverSchneider.jl available in this test file, so we can call pca, pca_transform, etc. directly without prefixing with BigRiverSchneider.

using Random, LinearAlgebra # we need Random for seeding the random number generator, and LinearAlgebra for matrix operations in the test code.
Random.seed!(1234) 





################################### TESTING ###################################
n, p, r = 5000, 200, 10    # 5000 observations, 200 features, 10 hidden signals
latent  = randn(n, r)              #  10 latent signals (5000 × 10) (components) that we will try to recover with PCA
mixing  = randn(r, p)              # random mixing matrix (10 × 200) that mixes the latent signals into the observed features
X       = latent * mixing .+ 0.05 .* randn(n, p)    # observed data (5000 × 200) is the mixed latent signals plus some noise
println("X is $(size(X,1)) × $(size(X,2))  ($(length(X)) entries)\n")
k    = 15
msvd = BigRiverSchneider.pca(X; k = k, method = :svd)
mcov = BigRiverSchneider.pca(X; k = k, method = :cov)
println("propOFvar (svd), first $k components:")
println(round.(msvd.propOFvar, digits = 4))
println("\ncumulative variance explained:")
println(round.(cumsum(msvd.propOFvar), digits = 4))
println("\nmax |variance| difference, svd vs cov : ",maximum(abs.(msvd.variances .- mcov.variances)))

scores = BigRiverSchneider.pca_transform(msvd, X)
println("\nscores size : ", size(scores))          # (5000, 15)
 
Xhat = BigRiverSchneider.pca_invtransform(msvd, scores)
rmse = sqrt(sum(abs2, X .- Xhat) / length(X))
println("reconstruction RMSE (k = $k) : ", round(rmse, digits = 5))
 

m10    = BigRiverSchneider.pca(X; k = 10, method = :svd)
rmse10 = sqrt(sum(abs2, X .- BigRiverSchneider.pca_invtransform(m10, BigRiverSchneider.pca_transform(m10, X))) / length(X))
println("reconstruction RMSE (k = 10) : ", round(rmse10, digits = 5))

println("\n-- timing (compilation already warmed up) --")
BigRiverSchneider.pca(X; k = k, method = :svd); BigRiverSchneider.pca(X; k = k, method = :cov)   # warmup
print("svd : "); msvd = @time BigRiverSchneider.pca(X; k = k, method = :svd);  
print("cov : "); mcov = @time BigRiverSchneider.pca(X; k = k, method = :cov);




 
 








# bench_pca.jl — correctness + speed for BigRiverSchneider.pca vs MultivariateStats.PCA
# Uses @btime so time AND allocations print inline for every method/size.
# Run:  julia --project bench_pca.jl

using BigRiverSchneider
using LinearAlgebra, Statistics, Random, BenchmarkTools, Printf
import MultivariateStats; const MVS = MultivariateStats
const BRS = BigRiverSchneider

# sign-invariant loading agreement (0 = identical up to per-column sign)
colmisalign(A, B) = maximum(1 - abs(dot(normalize(A[:,j]), normalize(B[:,j]))) for j in 1:size(A,2))

hr() = println("="^80)

# (n, p, k, label) — tall, square, wide
cases = [(500,   40,  6, "tall   n>>p"),
         (2000, 100, 10, "tall   n>>p (bigger)"),
         (5000, 200, 20, "tall   n>>p (large)"),
         (100,  100, 10, "square n=p"),
         (200,  500, 20, "wide   p>n"),
         (300, 1000, 30, "wide   p>n (bigger)")]

hr()
println("  PCA  —  BigRiverSchneider.pca  vs  MultivariateStats.PCA")
println("  correctness: sign-invariant loading misalignment (1-|cos|), want < 1e-8")
println("  timing: @btime (min time + allocations) for BRS :auto/:cov/:svd and MVS")
hr()

for (n, p, k, label) in cases
    Random.seed!(1)
    X = randn(n, p)                                    # BRS.pca: observations in ROWS
    Xt = permutedims(X)                                # MVS: variables in ROWS

    rm   = MVS.fit(MVS.PCA, Xt; maxoutdim = k, pratio = 1.0)
    Vmvs = MVS.projection(rm)

    println()
    @printf("  [%s]   n=%d p=%d k=%d   (auto picks %s)\n",
            label, n, p, k, n >= p ? ":cov" : ":svd")

    # --- correctness: each BRS path vs MVS (loadings + variances) ---
    for meth in (:auto, :cov, :svd)
        m  = BRS.pca(X; k = k, method = meth)
        d  = colmisalign(m.loadings, Vmvs)
        dv = maximum(abs.(sort(m.variances) .- sort(MVS.principalvars(rm)[1:k])))
        @printf("    match :%-5s  loading 1-|cos| = %.2e   var Δ = %.2e   %s\n",
                meth, d, dv, (d < 1e-8 && dv < 1e-6) ? "✓" : "✗")
    end

    # --- timing: @btime prints time + allocations inline for each ---
    println("    timing (@btime):")
    print("      BRS :auto : "); @btime BRS.pca($X; k = $k, method = :auto);
    print("      BRS :cov  : "); @btime BRS.pca($X; k = $k, method = :cov);
    print("      BRS :svd  : "); @btime BRS.pca($X; k = $k, method = :svd);
    print("      MVS       : "); @btime MVS.fit(MVS.PCA, $Xt; maxoutdim = $k, pratio = 1.0);
end

hr()
println("  Done.")
println("  :auto picks :cov for tall (n>=p), :svd for wide (p>n).")
println("  Watch the :cov match column on tall sizes — it validates the XᵀX−nμμᵀ")
println("  scatter trick. If :cov misaligns more than :svd, revert :cov to explicit")
println("  centering. On tall shapes BRS :auto (=:cov) should beat MVS.")
hr()