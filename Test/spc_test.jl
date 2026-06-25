# Test/spc_test.jl — formal tests for SPC (Witten sparse PCA): spc + spc_orth.
# Tolerances (tol_ord / tol_julia / tol_r) are defined in runtests.jl.

@testset "output structure & invariants" begin
    Random.seed!(1)
    n, p, k = 80, 60, 3
    X = randn(n, p)
    m = spc(X; k=k, c=0.6*sqrt(p))

    @test m isa spcStructure
    @test size(m.loadings) == (p, k)
    @test length(m.variances) == k
    @test length(m.propOFvar) == k
    @test length(m.mean) == p
    @test length(m.scale) == p

    # each loading column unit-norm (or zero)
    for j in 1:k
        nv = norm(m.loadings[:, j])
        @test isapprox(nv, 1.0; atol=tol_ord) || nv == 0.0
    end
    @test all(isfinite, m.variances)
    @test all(isfinite, m.propOFvar)
    # cumulative PVE is nondecreasing and within [0, 1]
    @test all(0 .<= m.propOFvar .<= 1 + tol_ord)
    @test all(diff(m.propOFvar) .>= -tol_ord)
    # column means recorded (default standardize=false ⇒ unit scale)
    @test isapprox(m.mean, vec(mean(X, dims=1)); atol=tol_ord)
    @test all(isapprox.(m.scale, 1.0; atol=tol_ord))
end

@testset "max budget reduces to ordinary PCA (theorem anchor)" begin
    # At c = √p the L1 penalty doesn't bind → no soft-thresholding → the loading
    # is the ordinary top PC of the column-centered data, fully dense.
    Random.seed!(7)
    n, p = 60, 40
    X = randn(n, p)
    Xc = X .- mean(X, dims=1)
    m = spc(X; k=1, c=sqrt(p))
    F = svd(Xc)
    @test abs(dot(m.loadings[:, 1], F.V[:, 1])) > 1 - tol_julia   # iterative
    @test count(!iszero, m.loadings[:, 1]) == p          # fully dense
end

@testset "sparsity contract (c controls sparsity)" begin
    Random.seed!(11)
    n, p = 80, 60
    X = randn(n, p)
    dense = spc(X; k=1, c=sqrt(p))
    mid   = spc(X; k=1, c=0.5*sqrt(p))
    tight = spc(X; k=1, c=1.5)
    @test count(!iszero, dense.loadings[:,1]) == p
    # smaller c ⇒ fewer (monotone) nonzeros
    @test count(!iszero, tight.loadings[:,1]) <= count(!iszero, mid.loadings[:,1]) <= p
    @test count(!iszero, tight.loadings[:,1]) < p        # tight actually zeros some
end

@testset "ground-truth: recovers planted sparse loading" begin
    # plant a sparse rank-1 signal; a tight budget gives high PRECISION (every
    # selected variable is real) — recall is budget-limited, so we assert on
    # precision + direction, not recall. Floors are literal recovery bars.
    Random.seed!(42)
    n, p = 200, 300
    u_true = [randn(50); zeros(n-50)]
    v_true = [randn(75); zeros(p-75)]
    X = u_true * v_true' .+ randn(n, p)
    m = spc(X; k=1, c=4.0)
    vh = m.loadings[:, 1]
    est = Set(findall(!iszero, vh)); truth = Set(1:75)
    prec = length(intersect(est, truth)) / length(est)
    @test prec > 0.8                                    # selected vars are real
    @test abs(dot(vh ./ norm(vh), v_true ./ norm(v_true))) > 0.6
    # loosening the budget improves recovery (recall + direction climb)
    m2 = spc(X; k=1, c=8.0)
    rec2 = length(intersect(Set(findall(!iszero, m2.loadings[:,1])), truth)) / 75
    @test rec2 > 0.8
end

@testset "spc_orth: scores are orthonormal" begin
    # the defining property of the orth variant: UᵀU ≈ I. spcStructure doesn't
    # store U, so recompute scores T = Xc·V; for the orthogonal variant the score
    # directions are orthogonal (off-diagonal correlations ≈ 0), unlike deflation.
    Random.seed!(5)
    n, p, k = 100, 50, 4
    X = randn(n, p)
    Xc = X .- mean(X, dims=1)
    mo = spc_orth(X; k=k, c=2.0)
    md = spc(X;      k=k, c=2.0)
    offdiag(C) = maximum(abs(C[i,j]) for i in 1:size(C,1) for j in 1:size(C,2) if i != j)
    Co = cor(Xc * mo.loadings)
    Cd = cor(Xc * md.loadings)
    # orthogonal variant has lower off-diagonal score correlation than deflation
    @test offdiag(Co) <= offdiag(Cd) + tol_ord
end

@testset "multiple components, shapes" begin
    Random.seed!(5)
    n, p, k = 60, 50, 4
    X = randn(n, p)
    m = spc(X; k=k, c=2.0)
    @test size(m.loadings) == (p, k)
    @test length(m.variances) == k
    for j in 1:k
        @test count(!iszero, m.loadings[:,j]) < p
    end
end

@testset "standardize=true scales columns" begin
    Random.seed!(9)
    n, p = 60, 40
    X = randn(n, p) .* (1:p)'            # wildly unequal column scales
    m = spc(X; k=1, c=0.5*sqrt(p), standardize=true)
    @test isapprox(m.scale, vec(std(X, dims=1)); atol=tol_ord)
    md = spc(X; k=1, c=0.5*sqrt(p), standardize=false)
    @test all(isapprox.(md.scale, 1.0; atol=tol_ord))
end

@testset "determinism (SVD init, no random dependence)" begin
    Random.seed!(99)
    n, p = 60, 40
    X = randn(n, p)
    a = spc(X; k=2, c=0.5*sqrt(p))
    b = spc(X; k=2, c=0.5*sqrt(p))
    for j in 1:2
        @test abs(dot(a.loadings[:,j] ./ norm(a.loadings[:,j]),
                      b.loadings[:,j] ./ norm(b.loadings[:,j]))) > 1 - tol_ord
        @test isapprox(a.variances[j], b.variances[j]; rtol=tol_ord)
    end
end

@testset "argument validation" begin
    Random.seed!(0)
    n, p = 100, 16
    X = randn(n, p)
    @test_throws ArgumentError spc(X; k=1, c=0.5)            # c < 1
    @test_throws ArgumentError spc(X; k=1, c=sqrt(p)+1)      # c > √p
    @test_throws ArgumentError spc_orth(X; k=1, c=0.5)
    @test_throws ArgumentError spc_orth(X; k=1, c=sqrt(p)+1)
end

@testset "internal: l1_diff (L1 distance)" begin
    l1d = BigRiverSchneider.l1_diff
    @test l1d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    @test l1d([1.0, 0.0], [0.0, 1.0]) == 2.0
    a = randn(40); b = randn(40)
    @test l1d(a, b) ≈ sum(abs, a .- b)
end

@testset "internal: finding_v! (soft-threshold to the L1 budget)" begin
    fv = BigRiverSchneider.finding_v!
    # contract: returns a UNIT-L2 vector whose L1 norm hits the budget c (when the
    # raw direction exceeds it), signs inherited from z, thresholded entries zeroed.
    p = 60
    Random.seed!(2)
    z = randn(p); v = similar(z); s = similar(z)

    # (1) slack budget: at c = √p no unit vector can exceed it ⇒ returned unchanged
    fv(v, s, z, sqrt(p))
    @test v ≈ z ./ norm(z)
    @test isapprox(norm(v), 1.0; atol = tol_ord)

    # (2) binding budget: result is unit-norm, L1 ≈ c
    for c in (2.0, 4.0, 6.0)
        fv(v, s, z, c)
        @test isapprox(norm(v), 1.0; atol = tol_ord)         # unit L2
        @test isapprox(sum(abs, v), c; atol = tol_r)         # L1 hits the budget (search precision)
        for i in eachindex(v)                                # signs inherited from z
            v[i] != 0 && @test sign(v[i]) == sign(z[i])
        end
    end

    # sparsity is guaranteed only for a TIGHT budget (c well below √p ≈ 7.75)
    fv(v, s, z, 2.0)
    @test count(!iszero, v) < p                              # tight ⇒ some entries zeroed

    # (3) tighter budget ⇒ at least as sparse (monotone)
    fv(v, s, z, 2.0); n_tight = count(!iszero, v)
    fv(v, s, z, 5.0); n_loose = count(!iszero, v)
    @test n_tight <= n_loose

    # (4) matches the explicit soft-threshold-then-normalize construction
    c = 3.0; fv(v, s, z, c)
    δ = let lo = 0.0, hi = maximum(abs, z)          # re-solve δ for the same budget
        for _ in 1:200
            mid = (lo + hi) / 2
            sm = sign.(z) .* max.(abs.(z) .- mid, 0.0)
            (sum(abs, sm) / (norm(sm) + eps()) < c) ? (hi = mid) : (lo = mid)
        end
        (lo + hi) / 2
    end
    ref = sign.(z) .* max.(abs.(z) .- δ, 0.0); ref ./= norm(ref)
    @test abs(dot(v, ref)) > 1 - tol_julia          # same budget-solve, two routes
end

@testset "internal: init_rsv (top-k right singular vectors, both branches)" begin
    irsv = BigRiverSchneider.init_rsv
    # tall (p ≤ n): eigen(XᵀX) branch
    Random.seed!(21)
    Xt = randn(60, 40); k = 3
    Vt = irsv(Xt, k)
    @test size(Vt) == (40, k)
    Ft = svd(Xt)
    for j in 1:k
        @test isapprox(norm(@view Vt[:, j]), 1.0; atol = tol_ord)    # unit columns
        @test abs(dot(Vt[:, j], Ft.V[:, j])) > 1 - tol_ord          # = right sing. vec
    end
    # wide (p > n): eigen(XXᵀ) + back-projection branch
    Xw = randn(30, 50)
    Vw = irsv(Xw, k)
    @test size(Vw) == (50, k)
    Fw = svd(Xw)
    for j in 1:k
        @test isapprox(norm(@view Vw[:, j]), 1.0; atol = tol_ord)
        @test abs(dot(Vw[:, j], Fw.V[:, j])) > 1 - tol_ord
    end
end

@testset "internal: prop_var_explained (trace identity vs explicit projection)" begin
    pve = BigRiverSchneider.prop_var_explained
    Random.seed!(33)
    n, p, K = 50, 40, 4
    Xc = randn(n, p) .- mean(randn(n, p), dims = 1)
    # an arbitrary (NON-orthonormal) sparse-ish V to stress the (VᵀV)⁻¹ term
    V = randn(p, K); V[abs.(V) .< 0.3] .= 0.0
    got = pve(Xc, V)

    # reference: ‖Xc·Vk·(VkᵀVk)⁻¹·Vkᵀ‖²_F / ‖Xc‖²_F  (the displaced form it replaces)
    totsq = sum(abs2, Xc)
    ref = [let Vk = V[:, 1:k]
               sum(abs2, Xc * Vk * inv(Vk' * Vk) * Vk') / totsq
           end for k in 1:K]
    @test got ≈ ref                                # the rewrite is algebraically exact
    @test all(0 .<= got .<= 1 + tol_ord)           # valid proportions
    @test all(diff(got) .>= -tol_ord)              # cumulative ⇒ nondecreasing

    # independent anchor: for ORTHONORMAL V (svd), pve[k] = Σσ₁..ₖ² / Σσ²
    Vo = svd(Xc).V[:, 1:K]
    S  = svd(Xc).S
    @test pve(Xc, Vo) ≈ cumsum(S[1:K].^2) ./ sum(abs2, S)
end

@testset "internal: spca_component! (rank-1 sparse core, deflation)" begin
    sc  = BigRiverSchneider.spca_component!
    irsv = BigRiverSchneider.init_rsv
    Random.seed!(31)
    n, p = 50, 40
    X = randn(n, p); Xc = X .- mean(X, dims = 1)
    v0 = irsv(Xc, 1)[:, 1]
    u = Vector{Float64}(undef, n); Xv = Vector{Float64}(undef, n)
    Xtu = Vector{Float64}(undef, p); s = Vector{Float64}(undef, p)
    vold = Vector{Float64}(undef, p); v = copy(v0)

    # at c = √p no penalty binds ⇒ pure power iteration ⇒ rank-1 SVD of Xc
    d = sc(v, Xc, sqrt(p), u, Xv, Xtu, s, vold; niter = 100)
    F = svd(Xc)
    @test isapprox(norm(u), 1.0; atol = tol_julia)
    @test isapprox(norm(v), 1.0; atol = tol_julia)
    @test abs(dot(v, F.V[:, 1])) > 1 - tol_julia      # v → V₁ (iterative)
    @test abs(dot(u, F.U[:, 1])) > 1 - tol_julia      # u → U₁
    @test isapprox(d, F.S[1]; rtol = tol_julia)       # d → σ₁
    @test d > 0

    # binding budget ⇒ v genuinely sparse
    v2 = copy(v0)
    sc(v2, Xc, 2.0, u, Xv, Xtu, s, vold; niter = 100)
    @test count(!iszero, v2) < p
end

@testset "internal: spca_component_orth! (orthogonal-score core)" begin
    sco = BigRiverSchneider.spca_component_orth!
    irsv = BigRiverSchneider.init_rsv
    Random.seed!(37)
    n, p = 60, 40
    X = randn(n, p); Xc = X .- mean(X, dims = 1)
    Vinit = irsv(Xc, 2)
    u = Vector{Float64}(undef, n); uold = Vector{Float64}(undef, n)
    Xv = Vector{Float64}(undef, n); Xtu = Vector{Float64}(undef, p)
    s = Vector{Float64}(undef, p); vold = Vector{Float64}(undef, p)
    proj = Vector{Float64}(undef, 2)

    # component 1: empty U_prev ⇒ reduces to ordinary core; capture u₁
    U = Matrix{Float64}(undef, n, 2)
    v1 = Vinit[:, 1]
    sco(v1, Xc, sqrt(p), @view(U[:, 1:0]), u, uold, Xv, Xtu, s, vold, proj; niter = 100)
    U[:, 1] .= u
    F = svd(Xc)
    @test abs(dot(U[:, 1], F.U[:, 1])) > 1 - tol_julia    # u₁ → U₁ at max budget (iterative)

    # component 2: U_prev = u₁ ⇒ returned u₂ must be orthogonal to u₁
    v2 = Vinit[:, 2]
    sco(v2, Xc, sqrt(p), @view(U[:, 1:1]), u, uold, Xv, Xtu, s, vold, proj; niter = 100)
    @test isapprox(norm(u), 1.0; atol = tol_julia)
    @test abs(dot(u, U[:, 1])) < tol_ord              # the defining orthogonality property
end

@testset "matches R PMA::SPC (offline reference fixtures)" begin
    refdir = joinpath(@__DIR__, "Data", "SPC")
    if !isfile(joinpath(refdir, "X.csv"))
        @info "SPC R-reference fixtures not found; skipping. Run spc.R to create them."
    else
        smfile = joinpath(refdir, "session_meta.csv")
        if isfile(smfile)
            sm = readdlm(smfile, ',', String; skipstart=1)
            row = findfirst(==("PMA_version"), sm[:, 1])
            row !== nothing && @info "SPC fixtures generated against PMA $(sm[row, 2])"
        end

        X    = readdlm(joinpath(refdir, "X.csv"),          ',', Float64; skipstart=1)
        v_r  = readdlm(joinpath(refdir, "v_spc.csv"),      ',', Float64; skipstart=1)
        d_r  = vec(readdlm(joinpath(refdir, "d_spc.csv"),  ',', Float64; skipstart=1))
        vo_r = readdlm(joinpath(refdir, "v_spc_orth.csv"), ',', Float64; skipstart=1)
        do_r = vec(readdlm(joinpath(refdir, "d_spc_orth.csv"), ',', Float64; skipstart=1))
        meta = readdlm(joinpath(refdir, "meta.csv"), ',', Float64; skipstart=1)
        n = Int(meta[1]); K = Int(meta[3]); sv = meta[4]

        # X.csv is RAW (uncentered); spc column-centers internally to match R.
        m  = spc(X;      k=K, c=sv)
        mo = spc_orth(X; k=K, c=sv)
        d_jl  = sqrt.(max.(m.variances,  0) .* (n - 1))   # spcStructure has no d field
        do_jl = sqrt.(max.(mo.variances, 0) .* (n - 1))

        for k in 1:K
            # orth=FALSE (cross-language ⇒ tol_r bar)
            @test abs(dot(m.loadings[:,k] ./ norm(m.loadings[:,k]),
                          v_r[:,k] ./ norm(v_r[:,k]))) > 1 - tol_r
            @test Set(findall(!iszero, m.loadings[:,k])) == Set(findall(!iszero, v_r[:,k]))
            # orth=TRUE
            @test abs(dot(mo.loadings[:,k] ./ norm(mo.loadings[:,k]),
                          vo_r[:,k] ./ norm(vo_r[:,k]))) > 1 - tol_r
            @test Set(findall(!iszero, mo.loadings[:,k])) == Set(findall(!iszero, vo_r[:,k]))
        end
        @test isapprox(d_jl,  d_r;  rtol=tol_r)
        @test isapprox(do_jl, do_r; rtol=tol_r)
    end
end