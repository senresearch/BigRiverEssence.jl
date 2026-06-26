# Test/pca_test.jl — formal tests for pca (svd & cov).
# Tolerances (tol_ord / tol_julia / tol_r) are defined in runtests.jl.
# PCA is deterministic and has no external reference ⇒ everything uses tol_ord.

loaddiff(A, B) = maximum(norm(abs.(A[:, j]) .- abs.(B[:, j])) for j in 1:size(A, 2))

@testset "output structure & invariants" begin
    Random.seed!(1)
    n, p, k = 200, 30, 5
    X = randn(n, p)
    m = pca(X; k=k, method=:svd)

    @test m isa pcaStructure
    @test length(m.mean)      == p
    @test length(m.scale)     == p
    @test size(m.loadings)    == (p, k)
    @test length(m.variances) == k
    @test length(m.propOFvar) == k

    # loadings are unit-norm columns
    for j in 1:k
        @test isapprox(norm(m.loadings[:, j]), 1.0; atol=tol_ord)
    end
    # variances non-negative and sorted descending (top-k, largest first)
    @test all(m.variances .>= -tol_ord)
    @test issorted(m.variances; rev=true)
    # proportions in [0,1]
    @test all(0 .<= m.propOFvar .<= 1 + tol_ord)

    # mean recovers the column means; no standardize → scale all ones
    @test isapprox(m.mean, vec(mean(X, dims=1)); atol=tol_ord)
    @test all(m.scale .== 1.0)
    ms = pca(X; k=k, standardize=true)
    @test isapprox(ms.scale, vec(std(X, dims=1)); atol=tol_ord)
end

@testset "loadings are orthonormal" begin
    # PCA directions must be mutually orthogonal as well as unit-norm: Vᵀ V = I.
    Random.seed!(21)
    n, p, k = 300, 15, 6
    X = randn(n, p)
    for method in (:svd, :cov)
        m = pca(X; k=k, method=method)
        @test isapprox(m.loadings' * m.loadings, I(k); atol=tol_ord)
    end
end

@testset "ground-truth: recovers a planted direction" begin
    # data along a known unit direction w → PC1 ≈ ±w, explains ~all variance.
    Random.seed!(7)
    n, p = 500, 10
    w = normalize(randn(p))
    scores = randn(n) .* 5.0
    X = scores * w' .+ 0.01 .* randn(n, p)

    for method in (:svd, :cov)
        m = pca(X; k=2, method=method)
        @test abs(dot(m.loadings[:, 1], w)) > 1 - tol_r   # PC1 aligns with w (noisy ground truth)
        @test m.propOFvar[1] > 1 - tol_r                  # PC1 dominates variance
    end
end

@testset "ground-truth: variances equal the data spectrum" begin
    # for centered X, the PCA variances must equal svdvals(Xc).^2 / (n-1).
    # checks the values, not just that they're sorted — needs no reference impl.
    Random.seed!(31)
    n, p, k = 400, 12, 12
    X = randn(n, p)
    Xc = X .- mean(X, dims=1)
    truevars = (svdvals(Xc) .^ 2) ./ (n - 1)
    for method in (:svd, :cov)
        m = pca(X; k=k, method=method)
        @test isapprox(m.variances, truevars[1:k]; rtol=tol_ord)
    end
end

@testset ":svd and :cov agree" begin
    # the two methods compute the same PCA two ways → must agree.
    Random.seed!(3)
    n, p, k = 300, 20, 8
    X = randn(n, p)
    msvd = pca(X; k=k, method=:svd)
    mcov = pca(X; k=k, method=:cov)
    @test isapprox(msvd.variances, mcov.variances; rtol=tol_ord)
    @test isapprox(msvd.propOFvar, mcov.propOFvar; rtol=tol_ord)
    @test loaddiff(msvd.loadings, mcov.loadings) < tol_ord
end

@testset "proportions sum correctly" begin
    # with all components kept, propOFvar must sum to 1.
    Random.seed!(17)
    n, p = 200, 10
    X = randn(n, p)
    m = pca(X; k=p, method=:svd)
    @test isapprox(sum(m.propOFvar), 1.0; atol=tol_ord)
end

@testset "sign consistency" begin
    # SignConsistency_opt! pins each column's largest-|·| entry positive.
    Random.seed!(5)
    X = randn(100, 8)
    for method in (:svd, :cov)
        m = pca(X; k=4, method=method)
        for j in 1:4
            c = m.loadings[:, j]
            @test c[argmax(abs.(c))] > 0
        end
    end
end

@testset "transform / inverse round-trip" begin
    Random.seed!(9)
    n, p = 150, 12
    X = randn(n, p)
    m = pca(X; k=p)
    scores = pca_transform(m, X)
    @test size(scores) == (n, p)
    # all-component reconstruction returns X
    Xrec = pca_invtransform(m, scores)
    @test isapprox(Xrec, X; atol=tol_ord)
    # scores = centered data projected onto loadings
    Xc = X .- mean(X, dims=1)
    @test isapprox(scores, Xc * m.loadings; atol=tol_ord)
end

@testset "standardize round-trip" begin
    Random.seed!(13)
    X = randn(120, 10) .* (1:10)' .+ 5           # columns on different scales
    m = pca(X; k=10, standardize=true)
    scores = pca_transform(m, X)
    Xrec   = pca_invtransform(m, scores)
    @test isapprox(Xrec, X; atol=tol_ord)
end

@testset "determinism" begin
    # pca has no random component → identical results on repeat calls.
    Random.seed!(99)
    X = randn(180, 14)
    a = pca(X; k=6, method=:svd)
    b = pca(X; k=6, method=:svd)
    @test a.loadings == b.loadings
    @test a.variances == b.variances
end

@testset "argument validation" begin
    Random.seed!(0)
    X = randn(50, 8)
    @test_throws ArgumentError pca(X; k=0)
    @test_throws ArgumentError pca(X; k=9)                    # k > min(n,p)=8
    @test_throws ArgumentError pca(X; method = :IWontMention)  # unknown method
end