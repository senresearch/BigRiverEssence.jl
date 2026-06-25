# Test/cca_test.jl — formal tests for cca (canonical correlation analysis),
# cca_transform, and the internal solvers (_cca_svd_opt, _cca_cov_opt, _qnormalize!).
# Tolerances (tol_ord / tol_julia / tol_r) are defined in runtests.jl.
#
# Reference is MultivariateStats.CCA (pure Julia — no fixtures, no R). CCA is a
# deterministic eigenproblem, so it agrees bit-identically up to per-column sign:
# the MVS cross-check uses tol_ord, projections compared sign-invariantly (column-wise
# |·|) and correlations directly (sign-free by construction).

const BRS = BigRiverSchneider
const cca           = BRS.cca
const cca_transform = BRS.cca_transform
const ccaStructure  = BRS.ccaStructure
const _svdcca = BRS._cca_svd_opt
const _covcca = BRS._cca_cov_opt
const _qnorm  = BRS._qnormalize!

# column-wise abs difference (per-column SVD/eigen sign is arbitrary)
projdiff(A, B) = maximum(norm(abs.(@view A[:, j]) .- abs.(@view B[:, j])) for j in 1:size(A, 2))

@testset "output structure & invariants" begin
    Random.seed!(1)
    dx, dy, n = 6, 5, 400
    X = randn(dx, n); Y = randn(dy, n)
    p = min(dx, dy)
    M = cca(X, Y; method = :svd)

    @test M isa ccaStructure
    @test length(M.xmean) == dx && length(M.ymean) == dy
    @test size(M.xproj) == (dx, p)
    @test size(M.yproj) == (dy, p)
    @test length(M.corrs) == p
    @test M.xmean ≈ vec(mean(X, dims = 2))
    @test M.ymean ≈ vec(mean(Y, dims = 2))
    # canonical correlations are in [0,1] and descending
    @test all(0 .<= M.corrs .<= 1 + tol_ord)
    @test issorted(M.corrs; rev = true)
end

@testset "outdim controls number of components" begin
    Random.seed!(2)
    X = randn(6, 300); Y = randn(5, 300)
    M = cca(X, Y; method = :svd, outdim = 3)
    @test length(M.corrs) == 3
    @test size(M.xproj, 2) == 3 && size(M.yproj, 2) == 3
end

@testset "ground truth: recovers a planted shared latent" begin
    # X and Y share one latent factor z ⇒ leading canonical corr ≈ 1, rest small.
    Random.seed!(42)
    dx, dy, n = 5, 4, 300
    z = randn(1, n)
    X = randn(dx, 1) * z .+ 0.05 .* randn(dx, n)
    Y = randn(dy, 1) * z .+ 0.05 .* randn(dy, n)
    M = cca(X, Y; method = :svd)
    @test M.corrs[1] > 0.99                       # leading correlation ≈ 1 (recovery floor)
    @test M.corrs[2] < 0.5                        # remaining clearly smaller
    # the leading canonical variates achieve that correlation
    Zx = cca_transform(M, X, :x)
    Zy = cca_transform(M, Y, :y)
    @test isapprox(abs(cor(Zx[1, :], Zy[1, :])), M.corrs[1]; atol = tol_ord)
end

@testset "cca_transform: canonical variates have the right correlations" begin
    Random.seed!(3)
    dx, dy, n = 6, 5, 400
    X = randn(dx, n); Y = randn(dy, n)
    sh = randn(2, n); X[1:2, :] .+= sh; Y[1:2, :] .+= sh
    M = cca(X, Y; method = :svd)
    Zx = cca_transform(M, X, :x)
    Zy = cca_transform(M, Y, :y)
    @test size(Zx) == (length(M.corrs), n)
    # each pair of canonical variates correlates at the reported canonical corr
    for j in 1:length(M.corrs)
        @test isapprox(abs(cor(Zx[j, :], Zy[j, :])), M.corrs[j]; atol = tol_ord)
    end
    @test_throws ArgumentError cca_transform(M, X, :z)   # invalid component
end

@testset ":svd and :cov agree (internal consistency)" begin
    Random.seed!(7)
    dx, dy, n = 6, 5, 400
    X = randn(dx, n); Y = randn(dy, n)
    sh = randn(2, n); X[1:2, :] .+= sh; Y[1:2, :] .+= sh
    Ms = cca(X, Y; method = :svd)
    Mc = cca(X, Y; method = :cov)
    @test norm(sort(Ms.corrs) .- sort(Mc.corrs)) < tol_ord
end

@testset "argument validation" begin
    Random.seed!(0)
    X = randn(6, 100); Y = randn(5, 100)
    @test_throws DimensionMismatch cca(X, randn(5, 80))       # mismatched n
    @test_throws ArgumentError cca(X, Y; outdim = 0)          # outdim < 1
    @test_throws ArgumentError cca(X, Y; outdim = 6)          # outdim > min(dx,dy)
    @test_throws ArgumentError cca(X, Y; method = :bogus)     # bad method
end

# ----------------------------------------------------------------------------
# internal helpers
# ----------------------------------------------------------------------------

@testset "internal: _qnormalize! (C-normalize columns)" begin
    Random.seed!(5)
    d, p = 6, 3
    Craw = randn(d, d); C = Craw * Craw' + I        # SPD
    P = randn(d, p)
    _qnorm(P, C)
    # each column satisfies pⱼᵀ C pⱼ = 1
    for j in 1:p
        @test isapprox(dot(@view(P[:, j]), C * @view(P[:, j])), 1.0; atol = tol_ord)
    end
end

@testset "internal: _cca_svd_opt (SVD solver)" begin
    Random.seed!(11)
    dx, dy, n = 6, 5, 400
    X = randn(dx, n); Y = randn(dy, n)
    sh = randn(2, n); X[1:2, :] .+= sh; Y[1:2, :] .+= sh
    xm = vec(mean(X, dims = 2)); ym = vec(mean(Y, dims = 2))
    M = _svdcca(copy(X) .- xm, copy(Y) .- ym, xm, ym, min(dx, dy))
    @test M isa ccaStructure
    @test issorted(M.corrs; rev = true)
    @test all(0 .<= M.corrs .<= 1 + tol_ord)
    @test M.nobs == n                               # svd path records nobs
end

@testset "internal: _cca_cov_opt (covariance solver, both dx≤dy and dx>dy)" begin
    Random.seed!(12)
    for (dx, dy) in ((5, 8), (8, 5))                # exercise BOTH branches
        n = 400
        X = randn(dx, n); Y = randn(dy, n)
        k = min(dx, dy)
        sh = randn(2, n); X[1:2, :] .+= sh; Y[1:2, :] .+= sh
        xm = vec(mean(X, dims = 2)); ym = vec(mean(Y, dims = 2))
        Zx = X .- xm; Zy = Y .- ym
        Cxx = (Zx * Zx') ./ (n - 1)
        Cyy = (Zy * Zy') ./ (n - 1)
        Cxy = (Zx * Zy') ./ (n - 1)
        M = _covcca(Cxx, Cyy, Cxy, xm, ym, k)
        @test length(M.corrs) == k
        @test all(0 .<= M.corrs .<= 1 + tol_ord)
        @test issorted(M.corrs; rev = true)
        # projections are C-normalized: PₓᵀCxxPₓ has unit diagonal
        D = M.xproj' * Cxx * M.xproj
        @test all(isapprox.(diag(D), 1.0; atol = tol_ord))
    end
end

# ----------------------------------------------------------------------------
# matches MultivariateStats (pure-Julia reference, both methods)
# ----------------------------------------------------------------------------

@testset "matches MultivariateStats.CCA" begin
    Random.seed!(7)
    for (dx, dy, n) in ((6, 5, 400), (50, 40, 2000))
        X = randn(dx, n); Y = randn(dy, n)
        nsh = min(dx, dy, 5)
        sh = randn(nsh, n); X[1:nsh, :] .+= sh; Y[1:nsh, :] .+= sh

        ref  = MVS.fit(MVS.CCA, X, Y; method = :svd)
        rc   = MVS.correlations(ref)
        rPx  = MVS.xprojection(ref); rPy = MVS.yprojection(ref)

        for meth in (:svd, :cov)
            M = cca(X, Y; method = meth)
            # correlations match to machine precision (sign-free, pure-Julia eigenproblem)
            @test norm(sort(M.corrs) .- sort(rc)) < tol_ord
            # projections match up to per-column sign
            @test projdiff(M.xproj, rPx) < tol_ord
            @test projdiff(M.yproj, rPy) < tol_ord
        end
    end
end