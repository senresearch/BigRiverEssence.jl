# Test/pmd_test.jl — formal tests for the two-sided PMD(L1,L1).
# Tolerances (tol_ord / tol_julia / tol_r) are defined in runtests.jl.

@testset "output structure & invariants" begin
    Random.seed!(1)
    n, p, K = 80, 60, 3
    X = randn(n, p)
    m = pmd(X; sumabs=0.4, K=K)

    @test m isa pmdStructure
    @test size(m.u) == (n, K)
    @test size(m.v) == (p, K)
    @test length(m.d) == K
    @test m.K == K

    # both factors unit-norm (or zero) per component
    for k in 1:K
        nu = norm(m.u[:, k]); nv = norm(m.v[:, k])
        @test isapprox(nu, 1.0; atol=tol_ord) || nu == 0.0
        @test isapprox(nv, 1.0; atol=tol_ord) || nv == 0.0
    end
    # d values are real and finite
    @test all(isfinite, m.d)
    # budgets were resolved into the recorded range
    @test 1 <= m.sumabsu <= sqrt(n) + tol_ord
    @test 1 <= m.sumabsv <= sqrt(p) + tol_ord
    # grand mean recorded when centering
    @test isapprox(m.meanx, mean(X); atol=tol_ord)
end

@testset "max budget reduces to rank-1 SVD (theorem anchor)" begin
    # At sumabsu=√n, sumabsv=√p neither penalty binds → no soft-thresholding →
    # plain power iteration → the rank-1 SVD of the centered data.
    Random.seed!(7)
    n, p = 60, 40
    X = randn(n, p)
    Xc = X .- mean(X)
    m = pmd(X; sumabsu=sqrt(n), sumabsv=sqrt(p), K=1)
    F = svd(Xc)
    @test abs(dot(m.u[:, 1], F.U[:, 1])) > 1 - tol_julia   # u → U₁ (iterative)
    @test abs(dot(m.v[:, 1], F.V[:, 1])) > 1 - tol_julia   # v → V₁
    # and both factors are fully dense at max budget
    @test count(!iszero, m.u[:, 1]) == n
    @test count(!iszero, m.v[:, 1]) == p
end

@testset "two-sided sparsity contract" begin
    Random.seed!(11)
    n, p = 80, 60
    X = randn(n, p)
    # smaller sumabs → sparser on BOTH factors; max budget → dense
    dense  = pmd(X; sumabsu=sqrt(n), sumabsv=sqrt(p), K=1)
    mid    = pmd(X; sumabs=0.5, K=1)
    tight  = pmd(X; sumabs=0.25, K=1)
    @test count(!iszero, dense.u[:,1]) == n
    @test count(!iszero, dense.v[:,1]) == p
    # tightening drives nonzeros down (monotone) on both sides
    @test count(!iszero, tight.u[:,1]) <= count(!iszero, mid.u[:,1]) <= n
    @test count(!iszero, tight.v[:,1]) <= count(!iszero, mid.v[:,1]) <= p
    @test count(!iszero, tight.v[:,1]) < p           # tight actually zeros some
end

@testset "ground-truth: recovers planted sparse factors" begin
    # plant two well-separated sparse rank-1 factors (sparse u AND sparse v).
    # strong signal + SVD init ⇒ unique optimum ⇒ recovery is clean.
    Random.seed!(42)
    n, p = 80, 60
    u1 = [randn(20); zeros(60)];            v1 = [randn(15); zeros(45)]
    u2 = [zeros(40); randn(20); zeros(20)]; v2 = [zeros(30); randn(15); zeros(15)]
    X = 6.0*(u1*v1') .+ 4.0*(u2*v2') .+ 0.3 .* randn(n, p)
    m = pmd(X; sumabsu=0.45*sqrt(n), sumabsv=0.45*sqrt(p), K=2, center=true)

    for (k, (ut, vt)) in enumerate(((u1,v1),(u2,v2)))
        # recovery floors on noisy planted data — deliberately literal (looser than 1-tol_r)
        @test abs(dot(m.u[:,k] ./ norm(m.u[:,k]), ut ./ norm(ut))) > 0.9
        @test abs(dot(m.v[:,k] ./ norm(m.v[:,k]), vt ./ norm(vt))) > 0.9
        selu = Set(findall(!iszero, m.u[:,k])); trueu = Set(findall(!iszero, ut))
        selv = Set(findall(!iszero, m.v[:,k])); truev = Set(findall(!iszero, vt))
        @test length(intersect(selu, trueu)) / length(trueu) > 0.8
        @test length(intersect(selv, truev)) / length(truev) > 0.8
    end
end

@testset "multiple components (deflation)" begin
    Random.seed!(5)
    n, p, K = 60, 50, 4
    X = randn(n, p)
    m = pmd(X; sumabs=0.4, K=K)
    @test size(m.u) == (n, K)
    @test size(m.v) == (p, K)
    @test length(m.d) == K
    # every component sparse at this budget
    for k in 1:K
        @test count(!iszero, m.v[:,k]) < p
    end
end

@testset "sumabsu/sumabsv override sumabs" begin
    Random.seed!(15)
    n, p = 70, 50
    X = randn(n, p)
    # explicit budgets are recorded directly (ignoring sumabs)
    su = 0.3*sqrt(n); sv = 0.6*sqrt(p)
    m = pmd(X; sumabs=0.9, sumabsu=su, sumabsv=sv, K=1)
    @test isapprox(m.sumabsu, su; atol=tol_ord)
    @test isapprox(m.sumabsv, sv; atol=tol_ord)
end

@testset "center=false skips centering" begin
    Random.seed!(9)
    n, p = 50, 40
    X = randn(n, p) .+ 5.0                # nonzero mean
    m = pmd(X; sumabs=0.5, K=1, center=false)
    @test isnan(m.meanx)                  # no mean recorded
end

@testset "determinism (SVD init, no random dependence)" begin
    # despite the randn loop-primer, the SVD init pins the result → repeat calls
    # converge to the same factors (up to the primer not changing the fixed point).
    Random.seed!(99)
    n, p = 60, 40
    X = randn(n, p)
    a = pmd(X; sumabs=0.4, K=2)
    b = pmd(X; sumabs=0.4, K=2)
    for k in 1:2
        @test abs(dot(a.v[:,k] ./ norm(a.v[:,k]), b.v[:,k] ./ norm(b.v[:,k]))) > 1 - tol_ord
        @test isapprox(a.d[k], b.d[k]; rtol=tol_ord)
    end
end

@testset "argument validation" begin
    Random.seed!(0)
    n, p = 100, 16
    X = randn(n, p)
    # K out of range
    @test_throws ArgumentError pmd(X; K=0)
    @test_throws ArgumentError pmd(X; K=min(n,p)+1)
    # sumabs out of (0,1]
    @test_throws ArgumentError pmd(X; sumabs=0.0)
    @test_throws ArgumentError pmd(X; sumabs=1.5)
    # explicit budgets out of range
    @test_throws ArgumentError pmd(X; sumabsu=0.5, sumabsv=2.0)        # sumabsu < 1
    @test_throws ArgumentError pmd(X; sumabsu=2.0, sumabsv=sqrt(p)+1)  # sumabsv > √p
end

@testset "internal: _pmd_soft (soft-threshold operator)" begin
    soft = BigRiverSchneider._pmd_soft
    @test soft(5.0, 2.0)  == 3.0          # shrink toward 0 by λ
    @test soft(-5.0, 2.0) == -3.0         # sign preserved
    @test soft(1.0, 2.0)  == 0.0          # |a| ≤ λ ⇒ exactly 0
    @test soft(-1.0, 2.0) == 0.0
    @test soft(3.0, 0.0)  == 3.0          # λ=0 is identity
    @test soft(0.0, 1.0)  == 0.0          # sign(0)=0 ⇒ 0, no NaN
    # matches the closed form elementwise on a random vector
    a = randn(100); λ = 0.7
    @test soft.(a, λ) ≈ sign.(a) .* max.(abs.(a) .- λ, 0.0)
end

@testset "internal: _pmd_l2n (guarded L2 norm)" begin
    l2n = BigRiverSchneider._pmd_l2n
    @test l2n([3.0, 4.0]) == 5.0
    @test l2n(zeros(5))   == 0.05         # PMA zero-guard (avoids /0)
    a = randn(50)
    @test l2n(a) ≈ norm(a)                # equals Euclidean norm when nonzero
end

@testset "internal: _pmd_l1_norm & _pmd_l1_norm_soft (L1/L2 ratio)" begin
    l1n  = BigRiverSchneider._pmd_l1_norm
    l1ns = BigRiverSchneider._pmd_l1_norm_soft
    soft = BigRiverSchneider._pmd_soft
    # ratio = ‖a‖₁ / ‖a‖₂ ; for ones(m) this is √m
    @test l1n([3.0, 4.0]) ≈ 7 / 5
    @test l1n(ones(9))    ≈ 3.0           # 9 / 3
    @test l1n(zeros(4))   == 0.0          # 0 / 0.05 guard ⇒ 0
    # soft-then-ratio variant equals applying soft, then the plain ratio
    a = randn(80); λ = 0.5
    @test l1ns(a, λ) ≈ l1n(soft.(a, λ))
    # bounded below by 1 for any nonzero vector (Cauchy–Schwarz)
    @test l1n(randn(30)) >= 1 - tol_ord
end

@testset "internal: _pmd_l1diff (L1 distance)" begin
    l1diff = BigRiverSchneider._pmd_l1diff
    @test l1diff([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    @test l1diff([1.0, 0.0], [0.0, 1.0]) == 2.0
    a = randn(40); b = randn(40)
    @test l1diff(a, b) ≈ sum(abs, a .- b)
end

@testset "internal: _pmd_binary_search (λ solving the L1 budget)" begin
    bs   = BigRiverSchneider._pmd_binary_search
    l1n  = BigRiverSchneider._pmd_l1_norm
    l1ns = BigRiverSchneider._pmd_l1_norm_soft
    # already within budget ⇒ no thresholding needed ⇒ λ = 0
    a = ones(4)                            # l1_norm = 2.0
    @test l1n(a) ≈ 2.0
    @test bs(a, 3.0) == 0.0                # budget 3 ≥ 2 ⇒ 0
    @test bs(zeros(6), 1.0) == 0.0         # all-zero input ⇒ 0
    # over budget ⇒ positive λ that drives the soft ratio to the target
    Random.seed!(3)
    z = randn(60)
    @test l1n(z) > 3.0                     # ensure thresholding is actually required
    λ = bs(z, 3.0)
    @test λ > 0
    @test λ <= maximum(abs, z)             # never exceeds the largest coefficient
    @test isapprox(l1ns(z, λ), 3.0; atol = tol_r)   # hits the requested budget (search precision)
    # monotone: a tighter budget needs a larger threshold
    @test bs(z, 2.0) >= bs(z, 4.0)
end

@testset "internal: _pmd_soft_normalize! (soft-threshold then unit-normalize)" begin
    sn   = BigRiverSchneider._pmd_soft_normalize!
    soft = BigRiverSchneider._pmd_soft
    arg = [3.0, -4.0, 0.5]; out = similar(arg)
    sn(out, arg, 1.0)                      # soft = [2,-3,0] ; normalize by √13
    @test out ≈ [2.0, -3.0, 0.0] ./ sqrt(13)
    @test isapprox(norm(out), 1.0; atol=tol_ord)  # unit L2 norm when entries survive
    @test sign.(out) == sign.([2.0, -3.0, 0.0])   # signs preserved
    # matches the explicit construction on a random vector
    a = randn(50); o = similar(a); λ = 0.6
    sn(o, a, λ)
    s = soft.(a, λ)
    @test o ≈ s ./ norm(s)
    # everything thresholded away ⇒ zeros / 0.05 guard ⇒ all-zero output, no NaN
    o2 = similar(a); sn(o2, a, maximum(abs, a) + 1.0)
    @test all(iszero, o2)
end

@testset "internal: _pmd_check_v (SVD-based initialization)" begin
    cv = BigRiverSchneider._pmd_check_v
    # tall matrix (p ≤ n): uses svd(xᵀx) branch
    Random.seed!(21)
    Xt = randn(60, 40); K = 3
    Vt = cv(Xt, K)
    @test size(Vt) == (40, K)
    Ft = svd(Xt)
    for k in 1:K
        @test isapprox(norm(@view Vt[:, k]), 1.0; atol = tol_ord)     # unit columns
        @test abs(dot(Vt[:, k], Ft.V[:, k])) > 1 - tol_ord           # = right sing. vec (up to sign)
    end
    # wide matrix (p > n): uses svd(xxᵀ) branch + back-projection
    Xw = randn(30, 50)
    Vw = cv(Xw, K)
    @test size(Vw) == (50, K)
    Fw = svd(Xw)
    for k in 1:K
        @test isapprox(norm(@view Vw[:, k]), 1.0; atol = tol_ord)
        @test abs(dot(Vw[:, k], Fw.V[:, k])) > 1 - tol_ord
    end
end

@testset "internal: _pmd_smd! (rank-1 sparse core)" begin
    smd = BigRiverSchneider._pmd_smd!
    cv  = BigRiverSchneider._pmd_check_v
    Random.seed!(31)
    n, p = 50, 40
    X = randn(n, p); Xc = X .- mean(X)
    v0 = cv(Xc, 1)[:, 1]
    u = Vector{Float64}(undef, n); v = Vector{Float64}(undef, p)
    vold = Vector{Float64}(undef, p)
    argu = Vector{Float64}(undef, n); argv = Vector{Float64}(undef, p)
    # at max budget (√n, √p) no penalty binds ⇒ pure power iteration ⇒ rank-1 SVD
    d = smd(Xc, v0, sqrt(n), sqrt(p), 50, u, v, vold, argu, argv)
    F = svd(Xc)
    @test isapprox(norm(u), 1.0; atol = tol_julia)
    @test isapprox(norm(v), 1.0; atol = tol_julia)
    @test abs(dot(u, F.U[:, 1])) > 1 - tol_julia      # u → U₁ (iterative)
    @test abs(dot(v, F.V[:, 1])) > 1 - tol_julia      # v → V₁
    @test isapprox(d, F.S[1]; rtol = tol_julia)       # d → σ₁
    @test d > 0
    # a binding budget makes v genuinely sparse
    u2 = similar(u); v2 = similar(v); vo2 = similar(vold)
    au2 = similar(argu); av2 = similar(argv)
    smd(Xc, v0, 0.4 * sqrt(n), 0.4 * sqrt(p), 50, u2, v2, vo2, au2, av2)
    @test count(!iszero, v2) < p
    @test count(!iszero, u2) < n
end

@testset "matches R PMA::PMD (offline reference fixtures)" begin
    # Reads precomputed R outputs from Test/Data/PMD/ (generated by pmd.R).
    refdir = joinpath(@__DIR__, "Data", "PMD")
    if !isfile(joinpath(refdir, "X.csv"))
        @info "PMD R-reference fixtures not found; skipping. Run pmd.R to create them."
    else
        # provenance: report which PMA version these were generated against
        smfile = joinpath(refdir, "session_meta.csv")
        if isfile(smfile)
            sm = readdlm(smfile, ',', String; skipstart=1)
            row = findfirst(==("PMA_version"), sm[:, 1])
            row !== nothing && @info "PMD fixtures generated against PMA $(sm[row, 2])"
        end

        X   = readdlm(joinpath(refdir, "X.csv"),     ',', Float64; skipstart=1)
        u_r = readdlm(joinpath(refdir, "u_pmd.csv"), ',', Float64; skipstart=1)
        v_r = readdlm(joinpath(refdir, "v_pmd.csv"), ',', Float64; skipstart=1)
        d_r = vec(readdlm(joinpath(refdir, "d_pmd.csv"), ',', Float64; skipstart=1))
        meta = readdlm(joinpath(refdir, "meta.csv"), ',', Float64; skipstart=1)
        K  = Int(meta[3]); su = meta[4]; sv = meta[5]

        # X.csv is already grand-mean-centered by R → run pmd with center=false
        m = pmd(X; sumabsu=su, sumabsv=sv, K=K, center=false)

        for k in 1:K
            # both factors align up to sign (cross-language ⇒ tol_r bar)
            @test abs(dot(m.u[:,k] ./ norm(m.u[:,k]), u_r[:,k] ./ norm(u_r[:,k]))) > 1 - tol_r
            @test abs(dot(m.v[:,k] ./ norm(m.v[:,k]), v_r[:,k] ./ norm(v_r[:,k]))) > 1 - tol_r
            # selected supports match exactly (same features zeroed on both sides)
            @test Set(findall(!iszero, m.u[:,k])) == Set(findall(!iszero, u_r[:,k]))
            @test Set(findall(!iszero, m.v[:,k])) == Set(findall(!iszero, v_r[:,k]))
        end
        # singular values match (cross-language)
        @test isapprox(m.d, d_r; rtol=tol_r)
    end
end