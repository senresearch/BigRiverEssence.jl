# Test/scca_test.jl — formal tests for scca (sparse CCA, Witten PMA::CCA method)
# and its supporting functions.
# Tolerances (tol_ord / tol_julia / tol_r) are defined in runtests.jl.

const BRS = BigRiverSchneider
const scca          = BRS.scca
const sccaStructure = BRS.sccaStructure
const _soft   = BRS._softcca!
const _l2     = BRS._l2n_val
const _l1n    = BRS._l1_of_norm
const _l1ns   = BRS._l1_of_norm_soft
const _l1d    = BRS._l1diff
const _bsearch = BRS._binary_search_opt
const _msqrt  = BRS._matsqrt

@testset "output structure & invariants" begin
    Random.seed!(1)
    dx, dy, n, K = 40, 30, 100, 2
    X = randn(dx, n); Y = randn(dy, n)
    m = scca(X, Y; penaltyx = 0.3, penaltyz = 0.3, K = K)

    @test m isa sccaStructure
    @test size(m.u) == (dx, K)
    @test size(m.v) == (dy, K)
    @test length(m.d) == K
    @test length(m.cors) == K
    @test m.K == K
    @test m.penaltyx == 0.3 && m.penaltyz == 0.3
    # each canonical vector is unit-norm (or zero), correlations in [0,1]
    for k in 1:K
        nu = norm(m.u[:, k]); nv = norm(m.v[:, k])
        @test isapprox(nu, 1.0; atol = tol_ord) || nu == 0.0
        @test isapprox(nv, 1.0; atol = tol_ord) || nv == 0.0
        @test 0 <= m.cors[k] <= 1 + tol_ord
    end
    @test all(isfinite, m.d)
end

@testset "penalty controls sparsity (smaller ⇒ sparser)" begin
    Random.seed!(2)
    X = randn(80, 60); Y = randn(70, 60)
    tight = scca(X, Y; penaltyx = 0.1, penaltyz = 0.1, K = 1)
    loose = scca(X, Y; penaltyx = 0.7, penaltyz = 0.7, K = 1)
    @test count(!iszero, tight.u[:, 1]) <= count(!iszero, loose.u[:, 1])
    @test count(!iszero, tight.v[:, 1]) <= count(!iszero, loose.v[:, 1])
end

@testset "ground truth: selects the planted sparse features" begin
    # first nz features of each view load on a shared latent; rest are noise.
    Random.seed!(42)
    n = 100; p1, p2 = 500, 1000; nz1, nz2 = 25, 40
    lat = randn(n)
    Xr = randn(n, p1); Zr = randn(n, p2)
    Xr[:, 1:nz1] .+= lat * fill(2.0, nz1)'
    Zr[:, 1:nz2] .+= lat * fill(2.0, nz2)'
    m = scca(Matrix(transpose(Xr)), Matrix(transpose(Zr));
             penaltyx = 0.2, penaltyz = 0.2, K = 1)
    sel_x = Set(findall(!iszero, m.u[:, 1]))
    sel_z = Set(findall(!iszero, m.v[:, 1]))
    # high precision: selected features are predominantly the true ones (recovery floors)
    @test length(intersect(sel_x, Set(1:nz1))) / max(1, length(sel_x)) > 0.7
    @test length(intersect(sel_z, Set(1:nz2))) / max(1, length(sel_z)) > 0.7
    @test m.cors[1] > 0.5                          # recovers strong correlation
end

@testset "standardize=false skips scaling" begin
    Random.seed!(3)
    X = randn(20, 80) .* 5 .+ 3; Y = randn(15, 80)
    ms = scca(X, Y; K = 1, standardize = true)
    mn = scca(X, Y; K = 1, standardize = false)
    @test ms isa sccaStructure && mn isa sccaStructure   # both run; paths differ
end

@testset "argument validation" begin
    Random.seed!(0)
    X = randn(10, 50); Y = randn(8, 50)
    @test_throws DimensionMismatch scca(X, randn(8, 40); K = 1)
    @test_throws ArgumentError scca(randn(1, 50), Y; K = 1)            # <2 features
    @test_throws ArgumentError scca(X, Y; penaltyx = 0.0, K = 1)       # penalty ∉ (0,1]
    @test_throws ArgumentError scca(X, Y; penaltyx = 1.5, K = 1)
    @test_throws ArgumentError scca(X, Y; K = 0)                       # K < 1
    @test_throws ArgumentError scca(X, Y; K = 11)                      # K > min(dx,dy)
end

# ----------------------------------------------------------------------------
# internal helpers
# ----------------------------------------------------------------------------

@testset "internal: _softcca! (soft-threshold)" begin
    a = [5.0, -3.0, 1.0, -0.5]; out = similar(a)
    _soft(out, a, 2.0)
    @test out ≈ [3.0, -1.0, 0.0, 0.0]              # shrink by 2, clamp at 0
    @test _soft(similar(a), a, 0.0) ≈ a            # λ=0 identity
    b = randn(50); o = similar(b); λ = 0.6
    _soft(o, b, λ)
    @test o ≈ sign.(b) .* max.(abs.(b) .- λ, 0.0)
end

@testset "internal: _l2n_val (guarded L2 norm)" begin
    @test _l2([3.0, 4.0]) == 5.0
    @test _l2(zeros(5)) == 0.05                    # zero-guard
    a = randn(40); @test _l2(a) ≈ norm(a)
end

@testset "internal: _l1_of_norm & _l1_of_norm_soft" begin
    @test _l1n([3.0, 4.0]) ≈ 7 / 5                 # ‖·‖₁/‖·‖₂
    @test _l1n(ones(9)) ≈ 3.0                      # 9/3 = √9
    a = randn(60); λ = 0.5
    @test _l1ns(a, λ) ≈ _l1n(sign.(a) .* max.(abs.(a) .- λ, 0.0))
    @test _l1n(randn(30)) >= 1 - tol_ord           # ≥1 for nonzero
end

@testset "internal: _l1diff (L1 distance)" begin
    @test _l1d([1.0, 2.0], [1.0, 2.0]) == 0.0
    @test _l1d([1.0, 0.0], [0.0, 1.0]) == 2.0
    a = randn(30); b = randn(30)
    @test _l1d(a, b) ≈ sum(abs, a .- b)
end

@testset "internal: _binary_search_opt (λ for the L1 budget)" begin
    a = ones(4)                                    # l1_of_norm = 2.0
    @test _bsearch(a, 3.0) == 0.0                  # within budget ⇒ 0
    @test _bsearch(zeros(6), 1.0) == 0.0
    Random.seed!(3)
    z = randn(60)
    @test _l1n(z) > 3.0                            # ensure thresholding needed
    λ = _bsearch(z, 3.0)
    @test λ > 0
    @test λ <= maximum(abs, z)
    @test isapprox(_l1ns(z, λ), 3.0; atol = tol_r) # hits the budget (search precision)
    @test _bsearch(z, 2.0) >= _bsearch(z, 4.0)     # tighter ⇒ larger λ
end

@testset "internal: _matsqrt (symmetric matrix square root)" begin
    Random.seed!(5)
    M = randn(8, 8); A = M * M' + I               # SPD
    R = _msqrt(A)
    @test R * R ≈ A                                # R² = A
    @test R ≈ R'                                   # symmetric
end

@testset "internal: _fast_init_v (wide-data SVD initializer)" begin
    _fiv = BRS._fast_init_v
    # x, z are rows=obs, cols=features — the wide case (features > obs)
    Random.seed!(13)
    nobs = 30; p1 = 60; p2 = 80; K = 3
    x = randn(nobs, p1); z = randn(nobs, p2)
    V = _fiv(x, z, K)
    @test size(V) == (p2, K)                       # init for v lives in z's feature space
    # columns are orthonormal (taken from an SVD's U factor)
    @test V' * V ≈ I(K) atol = tol_ord
    # matches the documented construction: U of svd(zᵀ · sqrt(xxᵀ))
    xx_sqrt = BRS._matsqrt(x * transpose(x))
    Vref = svd(transpose(z) * xx_sqrt).U[:, 1:K]
    for k in 1:K
        @test abs(dot(V[:, k], Vref[:, k])) > 1 - tol_ord  # equal up to sign
    end
end

@testset "internal: _sparse_cca_single_opt! (rank-1 sparse core)" begin
    sccacore = BRS._sparse_cca_single_opt!
    # plant a shared latent so the core has real structure to find.
    Random.seed!(17)
    nobs = 100; p1 = 40; p2 = 50
    lat = randn(nobs)
    x = randn(nobs, p1); z = randn(nobs, p2)
    x[:, 1:8]  .+= lat * fill(2.0, 8)'             # first 8 X-features load on lat
    z[:, 1:10] .+= lat * fill(2.0, 10)'            # first 10 Z-features load on lat
    if true                                        # standardize like scca does
        x = (x .- mean(x, dims = 1)) ./ std(x, dims = 1; corrected = true)
        z = (z .- mean(z, dims = 1)) ./ std(z, dims = 1; corrected = true)
    end

    v0 = svd(transpose(x) * z).V[:, 1]             # same init scca uses (n > p case)
    px = pz = 0.3
    # caller-owned buffers, exactly as scca allocates them
    u = Vector{Float64}(undef, p1); v = Vector{Float64}(undef, p2)
    vold = Vector{Float64}(undef, p2)
    zv = Vector{Float64}(undef, nobs); xu = Vector{Float64}(undef, nobs)
    argu = Vector{Float64}(undef, p1); argv = Vector{Float64}(undef, p2)
    su = Vector{Float64}(undef, p1); sv = Vector{Float64}(undef, p2)

    d = sccacore(u, v, x, z, v0, px, pz, 50, vold, zv, xu, argu, argv, su, sv)

    # the canonical vectors come out unit-norm (iterative core ⇒ tol_julia)
    @test isapprox(norm(u), 1.0; atol = tol_julia)
    @test isapprox(norm(v), 1.0; atol = tol_julia)
    # penalty actually induced sparsity (some features zeroed)
    @test count(!iszero, u) < p1
    @test count(!iszero, v) < p2
    # d = uᵀXᵀZv = ⟨Xu, Zv⟩ — exact identity (the function's return contract)
    @test d ≈ dot(x * u, z * v)
    @test d > 0
    # the canonical variates are genuinely correlated (recovery floor)
    @test abs(cor(x * u, z * v)) > 0.5
    # the selected features concentrate on the planted ones (precision floors)
    sel_u = Set(findall(!iszero, u)); sel_v = Set(findall(!iszero, v))
    @test length(intersect(sel_u, Set(1:8)))  / max(1, length(sel_u)) > 0.6
    @test length(intersect(sel_v, Set(1:10))) / max(1, length(sel_v)) > 0.6
end

# ----------------------------------------------------------------------------
# matches PMA::CCA — offline fixtures
# ----------------------------------------------------------------------------

@testset "matches PMA::CCA (offline reference fixtures)" begin
    refdir = joinpath(@__DIR__, "Data", "SCCA")
    if !isfile(joinpath(refdir, "X.csv"))
        @info "sCCA PMA fixtures not found; run scca.R to create them."
    else
        smfile = joinpath(refdir, "session_meta.csv")
        if isfile(smfile)
            sm = readdlm(smfile, ',', String; skipstart = 1)
            row = findfirst(==("PMA_version"), sm[:, 1])
            row !== nothing && @info "sCCA fixtures generated against PMA $(sm[row, 2])"
        end

        rd(f) = readdlm(joinpath(refdir, f), ',', Float64; skipstart = 1)
        X = rd("X.csv"); Z = rd("Z.csv")          # rows=obs (PMA layout)
        ru = rd("u.csv"); rv = rd("v.csv")
        rd_ = vec(rd("d.csv")); rcors = vec(rd("cors.csv"))
        meta = rd("meta.csv")
        K = Int(meta[1]); px = meta[2]; pz = meta[3]; niter = Int(meta[4])

        # scca takes columns=obs ⇒ transpose PMA's row-major matrices
        m = scca(Matrix(transpose(X)), Matrix(transpose(Z));
                 penaltyx = px, penaltyz = pz, K = K, niter = niter)

        for k in 1:K
            # sign-invariant loading agreement (cross-language ⇒ tol_r bar)
            @test abs(cor(m.u[:, k], ru[:, k])) > 1 - tol_r
            @test abs(cor(m.v[:, k], rv[:, k])) > 1 - tol_r
            # selected-feature sets match
            @test Set(findall(!iszero, m.u[:, k])) == Set(findall(!iszero, ru[:, k]))
            @test Set(findall(!iszero, m.v[:, k])) == Set(findall(!iszero, rv[:, k]))
        end
        # d and correlations match (cross-language)
        @test isapprox(m.d, rd_; rtol = tol_r)
        @test isapprox(m.cors, rcors; rtol = tol_r)
    end
end