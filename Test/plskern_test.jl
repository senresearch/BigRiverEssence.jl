# Test/plskern_test.jl — formal tests for plskern (Dayal & MacGregor kernel PLS)
# and its supporting functions (plskerncoef, plskernpredict, plskerntransform).
# Tolerances (tol_ord / tol_julia / tol_r) are defined in runtests.jl.

const BRS = BigRiverSchneider
const plskern          = BRS.plskern
const plskerncoef      = BRS.plskerncoef
const plskernpredict   = BRS.plskernpredict
const plskerntransform = BRS.plskerntransform
const plskernStructure = BRS.plskernStructure

const HAS_JCHEMO = let
    try; @eval import Jchemo; true; catch; false; end
end
HAS_JCHEMO || @info "Jchemo not available; cross-implementation tests will be skipped."

@testset "output structure & invariants" begin
    Random.seed!(1)
    n, p, q, nlv = 80, 30, 4, 5
    X = randn(n, p); Y = randn(n, q)
    m = plskern(X, Y; nlv = nlv)

    @test m isa plskernStructure
    @test size(m.W) == (p, nlv)
    @test size(m.P) == (p, nlv)
    @test size(m.Q) == (q, nlv)
    @test size(m.R) == (p, nlv)
    @test size(m.T) == (n, nlv)
    @test m.xmeans ≈ vec(mean(X, dims = 1))
    @test m.ymeans ≈ vec(mean(Y, dims = 1))
    @test all(m.xscales .== 1.0)                       # standardize=false default
    @test all(m.yscales .== 1.0)
    for a in 1:nlv
        @test isapprox(norm(m.W[:, a]), 1.0; atol = tol_ord)   # unit-norm weights
    end
    @test all(isfinite, m.T)
end

@testset "nlv is clamped to min(nlv, n, p)" begin
    Random.seed!(2)
    m = plskern(randn(20, 8), randn(20); nlv = 50)     # absurd request
    @test size(m.R, 2) == 8                            # clamped to p
    m2 = plskern(randn(6, 40), randn(6); nlv = 30)
    @test size(m2.R, 2) == 6                           # clamped to n
end

@testset "scores T are orthogonal (PLS property)" begin
    Random.seed!(3)
    X = randn(100, 25); y = randn(100)
    m = plskern(X, y; nlv = 10)
    G = m.T' * m.T
    offdiag = maximum(abs(G[i, j]) for i in 1:10 for j in 1:10 if i != j)
    @test offdiag < tol_ord
    @test all(diag(G) .> 0)                            # nonzero score variance
end

@testset "T = Xc·R (scores are linear in deflated X)" begin
    Random.seed!(4)
    X = randn(60, 20); Y = randn(60, 3)
    m = plskern(X, Y; nlv = 6)
    Xc = (X .- m.xmeans') ./ m.xscales'
    @test m.T ≈ Xc * m.R
end

@testset "algo1 and algo2 give identical results" begin
    Random.seed!(5)
    X = randn(120, 40); Y = randn(120, 2)
    m1 = plskern(X, Y; nlv = 12, method = :algo1)
    m2 = plskern(X, Y; nlv = 12, method = :algo2)
    @test isapprox(m1.R, m2.R; rtol = tol_ord)
    @test isapprox(m1.T, m2.T; rtol = tol_ord)
    @test isapprox(m1.Q, m2.Q; rtol = tol_ord)
    B1, i1 = plskerncoef(m1); B2, i2 = plskerncoef(m2)
    @test isapprox(B1, B2; rtol = tol_ord)             # B is uniquely determined
    @test isapprox(i1, i2; rtol = tol_ord)
end

@testset "full-rank PLS equals OLS (theorem anchor)" begin
    # at nlv = p (well-conditioned), PLS spans the full column space ⇒ = OLS.
    Random.seed!(6)
    n, p = 100, 20
    X = randn(n, p); y = randn(n)
    m = plskern(X, y; nlv = p)
    ŷ = vec(plskernpredict(m, X))
    Xc = X .- mean(X, dims = 1)
    B_ols = Xc \ (y .- mean(y))
    ŷ_ols = mean(y) .+ Xc * B_ols
    @test isapprox(ŷ, ŷ_ols; rtol = tol_ord)
    # multi-response full rank
    Y = randn(n, 3)
    mY = plskern(X, Y; nlv = p)
    Yc = Y .- mean(Y, dims = 1)
    B_olsY = Xc \ Yc
    @test isapprox(plskernpredict(mY, X), (mean(Y, dims = 1) .+ Xc * B_olsY); rtol = tol_ord)
end

@testset "plskerncoef: B and intercept shapes & reconstruction" begin
    Random.seed!(7)
    n, p, q = 70, 15, 3
    X = randn(n, p); Y = randn(n, q)
    m = plskern(X, Y; nlv = 8)
    B, intercept = plskerncoef(m)
    @test size(B) == (p, q)
    @test size(intercept) == (1, q)
    @test plskernpredict(m, X) ≈ intercept .+ X * B    # predict ≡ coef-then-apply
    # truncating nlv in coef matches a model fit at that nlv (nested property)
    B5, _ = plskerncoef(m; nlv = 5)
    m5    = plskern(X, Y; nlv = 5)
    B5b, _ = plskerncoef(m5)
    @test isapprox(B5, B5b; rtol = tol_ord)
end

@testset "plskerntransform: scores on training data == m.T" begin
    Random.seed!(8)
    X = randn(50, 12); y = randn(50)
    m = plskern(X, y; nlv = 6)
    @test isapprox(plskerntransform(m, X), m.T; rtol = tol_ord)
    @test isapprox(plskerntransform(m, X; nlv = 3), m.T[:, 1:3]; rtol = tol_ord)
end

@testset "standardize=true scales X and Y" begin
    Random.seed!(9)
    X = randn(60, 10) .* (1:10)'; Y = randn(60, 2)
    m = plskern(X, Y; nlv = 5, standardize = true)
    @test m.xscales ≈ vec(std(X, dims = 1))
    @test m.yscales ≈ vec(std(Y, dims = 1))
    md = plskern(X, Y; nlv = 5, standardize = false)
    @test all(md.xscales .== 1.0)
end

@testset "accepts vector or matrix Y" begin
    Random.seed!(10)
    X = randn(40, 8); y = randn(40)
    mv = plskern(X, y; nlv = 4)                         # Vector
    mm = plskern(X, reshape(y, :, 1); nlv = 4)          # n×1 Matrix
    @test size(mv.Q) == (1, 4)
    Bv, _ = plskerncoef(mv); Bm, _ = plskerncoef(mm)
    @test Bv ≈ Bm
end

@testset "argument validation" begin
    Random.seed!(0)
    X = randn(30, 6); y = randn(30)
    @test_throws ArgumentError plskern(X, y; nlv = 2, method = :bogus)
end

@testset "matches Jchemo.plskern (live, if available)" begin
    if !HAS_JCHEMO
        @test_skip "Jchemo not installed"
    else
        Random.seed!(1234)
        n, p, nlv = 400, 50, 12
        X = randn(n, p); y = randn(n)

        m_mine   = plskern(X, y; nlv = nlv, method = :algo1)
        B_mine,_ = plskerncoef(m_mine)

        mod = Jchemo.plskern(; nlv = nlv)               # scal=false ↔ standardize=false
        Jchemo.fit!(mod, X, y)
        B_jc = Jchemo.coef(mod).B

        @test maximum(abs.(B_mine .- B_jc)) < tol_julia # B is sign-unambiguous
        ŷ_mine = vec(plskernpredict(m_mine, X))
        ŷ_jc   = vec(Jchemo.predict(mod, X).pred)
        @test maximum(abs.(ŷ_mine .- ŷ_jc)) < tol_julia
        # scores agree up to per-column sign (latent factors carry sign ambiguity)
        T_jc = Jchemo.transf(mod, X)
        for a in 1:nlv
            @test abs(dot(m_mine.T[:, a] ./ norm(m_mine.T[:, a]),
                          T_jc[:, a]      ./ norm(T_jc[:, a]))) > 1 - tol_julia
        end
        # algo2 path must also match Jchemo
        m2 = plskern(X, y; nlv = nlv, method = :algo2)
        B2, _ = plskerncoef(m2)
        @test maximum(abs.(B2 .- B_jc)) < tol_julia
        # multi-response cross-check
        Y = randn(n, 3)
        mYmine = plskern(X, Y; nlv = nlv); BYmine, _ = plskerncoef(mYmine)
        modY = Jchemo.plskern(; nlv = nlv); Jchemo.fit!(modY, X, Y)
        @test maximum(abs.(BYmine .- Jchemo.coef(modY).B)) < tol_julia
    end
end