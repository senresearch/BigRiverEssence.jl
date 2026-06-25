# Test/jive_test.jl — formal tests for jive (r.jive-matching JIVE) and its
# supporting functions: _jive_rjive_core_opt2, _jive_perm_ranks_opt, safe_svd*.
# Tolerances (tol_ord / tol_julia / tol_r) are defined in runtests.jl.

const BRS  = BigRiverSchneider
const jive = BRS.jive
const jiveStructure = BRS.jiveStructure
const _core  = BRS._jive_rjive_core_opt2
const _perm  = BRS._jive_perm_ranks_opt
const _ssvd  = BRS.safe_svd
const _ssvdv = BRS.safe_svdvals
const _ssvd! = BRS.safe_svd!


# mirror jive's own preprocessing (row-center + r.jive Frobenius scaling)
function _preprocess(Xs; center = true, scale = true)
    k = length(Xs)
    sum_n = sum(size(X, 1) * size(X, 2) for X in Xs)
    Xc = Vector{Matrix{Float64}}(undef, k)
    for i in 1:k
        Xi = center ? Float64.(Xs[i]) .- mean(Xs[i], dims = 2) : Float64.(Xs[i])
        scale && (Xi ./= (norm(Xi) * sqrt(sum_n)))
        Xc[i] = Matrix{Float64}(Xi)
    end
    return Xc
end

# canonical correlations between two subspaces, using their first `d` directions.
# (d is the subspace dimension — e.g. the joint rank — NOT the column count of B,
#  since a stacked-joint basis has many columns but spans only a d-dim subspace.)
function canon(B1, B2, d)
    Q1 = Matrix(qr(B1).Q)[:, 1:d]
    Q2 = Matrix(qr(B2).Q)[:, 1:d]
    svdvals(Q1' * Q2)
end

# shared simulated data with KNOWN structure (matches jive.R's generator)
function make_data(; seed = 2024, noise = 0.3)
    Random.seed!(seed)
    n = 80; rT, r1T, r2T = 2, 3, 3; p1, p2 = 60, 50
    S  = randn(rT, n)
    U1 = randn(p1, rT); U2 = randn(p2, rT)
    S1 = randn(r1T, n); W1 = randn(p1, r1T)
    S2 = randn(r2T, n); W2 = randn(p2, r2T)
    X1 = U1*S + W1*S1 .+ noise .* randn(p1, n)
    X2 = U2*S + W2*S2 .+ noise .* randn(p2, n)
    (; X1, X2, S, n, rT, r1T, r2T, p1, p2)
end

# structured data for the auto-rank tests (strong, well-separated 2 + [3,3])
function make_struct(; seed = 42)
    Random.seed!(seed)
    n = 80; p1, p2 = 60, 50
    Sj = randn(2, n)
    X1 = randn(p1,2)*Sj + randn(p1,3)*randn(3,n)*0.5 + randn(p1,n)*0.1
    X2 = randn(p2,2)*Sj + randn(p2,3)*randn(3,n)*0.5 + randn(p2,n)*0.1
    (; X1, X2, n)
end


@testset "output structure & invariants" begin
    d = make_data()
    m = jive([d.X1, d.X2], d.rT, [d.r1T, d.r2T])

    @test m isa jiveStructure
    @test length(m.J) == 2 && length(m.A) == 2
    @test size(m.J[1]) == (d.p1, d.n) && size(m.J[2]) == (d.p2, d.n)
    @test size(m.A[1]) == (d.p1, d.n) && size(m.A[2]) == (d.p2, d.n)
    @test size(m.S) == (d.rT, d.n)
    @test m.r == d.rT
    @test m.ri == [d.r1T, d.r2T]
    @test all(isfinite, m.S)
    @test m.S * m.S' ≈ I(d.rT) atol = tol_ord     # joint scores have orthonormal rows
end



@testset "joint structure has rank r; individual has rank ri" begin
    d = make_data()
    m = jive([d.X1, d.X2], d.rT, [d.r1T, d.r2T])
    sJ = svdvals(reduce(vcat, m.J))
    @test sJ[d.rT + 1] / sJ[1] < tol_ord          # stacked joint is exactly rank r
    for (i, ri) in enumerate([d.r1T, d.r2T])
        sA = svdvals(m.A[i])
        @test sA[ri + 1] / sA[1] < tol_ord        # each individual block is rank ri
    end
end

@testset "joint ⊥ individual (orthogonality constraint)" begin
    d = make_data()
    m = jive([d.X1, d.X2], d.rT, [d.r1T, d.r2T])
    Jstack = reduce(vcat, m.J)
    for i in 1:2
        @test norm(Jstack * m.A[i]') / (norm(Jstack) * norm(m.A[i])) < 1e-4   # orthogonality floor
    end
    for i in 1:2
        @test norm(m.S * m.Si[i]') / (norm(m.S) * norm(m.Si[i])) < 1e-4
    end
end

@testset "ground truth: recovers joint subspace (noiseless)" begin
    # r.jive's orthIndiv variant does NOT reconstruct X exactly even when noiseless
    # (the orthogonality constraint trades reconstruction for orthogonality), so we
    # test SUBSPACE recovery + orthogonality, NOT ‖X−(J+A)‖≈0.
    Random.seed!(2024)
    n, rT, r1T, r2T, p1, p2 = 80, 2, 3, 3, 60, 50
    S  = randn(rT, n); U1 = randn(p1, rT); U2 = randn(p2, rT)
    S1 = randn(r1T, n); W1 = randn(p1, r1T); S2 = randn(r2T, n); W2 = randn(p2, r2T)
    X1c = U1*S + W1*S1; X2c = U2*S + W2*S2                # clean (noiseless)

    m = jive([X1c, X2c], rT, [r1T, r2T]; scale = false)
    Jstack = reduce(vcat, m.J)
    for i in 1:2
        @test norm(Jstack * m.A[i]') / (norm(Jstack) * norm(m.A[i])) < 1e-4   # orthogonality floor
    end
    Strue = S .- mean(S, dims = 2)
    @test minimum(canon(Strue', m.S', rT)) > 0.99        # subspace recovery floor (noisy)
end

@testset "auto ranks: permutation finds the planted ranks" begin
    s = make_struct()
    Random.seed!(999)
    m = jive([s.X1, s.X2]; nperm = 100)
    @test m.r == 2                                        # joint rank (stable)
    @test m.ri == [3, 3]                                 # individual ranks
end

@testset "positional and keyword signatures agree" begin
    d = make_data()
    a = jive([d.X1, d.X2], d.rT, [d.r1T, d.r2T])
    b = jive([d.X1, d.X2]; r = d.rT, ri = [d.r1T, d.r2T])
    for i in 1:2
        @test a.J[i] ≈ b.J[i]
        @test a.A[i] ≈ b.A[i]
    end
end

@testset "preprocessing: center + r.jive scaling" begin
    d = make_data()
    Xc = _preprocess([d.X1, d.X2])
    @test all(abs(sum(@view Xc[1][r, :])) < tol_ord for r in 1:d.p1)   # rows centered
    @test norm(Xc[1]) < norm(d.X1 .- mean(d.X1, dims = 2))             # scaling shrinks norm
end

@testset "argument validation" begin
    Random.seed!(0)
    X1 = randn(20, 50); X2 = randn(15, 40)               # mismatched #columns
    @test_throws ArgumentError jive([X1, X2], 2, [2, 2])
end

# ----------------------------------------------------------------------------
# internal helpers
# ----------------------------------------------------------------------------

@testset "internal: safe_svd / safe_svdvals / safe_svd!" begin
    Random.seed!(7)
    A = randn(40, 25)
    F = _ssvd(A)
    @test F.U * Diagonal(F.S) * F.Vt ≈ A
    @test _ssvdv(A) ≈ svdvals(A)
    @test F.S ≈ svdvals(A)
    Acopy = copy(A)                                       # safe_svd! mutates its input
    F2 = _ssvd!(Acopy)
    @test F2.U * Diagonal(F2.S) * F2.Vt ≈ A
end

@testset "internal: _jive_rjive_core_opt2 (given-ranks core)" begin
    d = make_data()
    Xc = _preprocess([d.X1, d.X2])
    conv = 1e-6 * norm(reduce(vcat, Xc))
    m = _core(Xc, d.n, d.rT, [d.r1T, d.r2T]; conv = conv, maxiter = 1000)
    @test m isa jiveStructure
    @test m.r == d.rT && m.ri == [d.r1T, d.r2T]
    sJ = svdvals(reduce(vcat, m.J))
    @test sJ[d.rT + 1] / sJ[1] < tol_ord
    Jstack = reduce(vcat, m.J)
    @test norm(Jstack * m.A[1]') / (norm(Jstack) * norm(m.A[1])) < 1e-4   # orthogonality floor
end

@testset "internal: _jive_perm_ranks_opt (rank estimator)" begin
    s = make_struct()
    Xc = _preprocess([s.X1, s.X2])
    conv = 1e-6 * norm(reduce(vcat, Xc))
    Random.seed!(999)
    rJ, rA = _perm(Xc, s.n; nperm = 100, alpha = 0.05, conv = conv, maxiter = 1000)
    @test rJ == 2
    @test rA == [3, 3]
end

# ----------------------------------------------------------------------------
# matches r.jive — offline fixtures
# ----------------------------------------------------------------------------

@testset "matches r.jive (offline reference fixtures, given ranks)" begin
    refdir = joinpath(@__DIR__, "Data", "JIVE")
    if !isfile(joinpath(refdir, "X1.csv"))
        @info "JIVE r.jive fixtures not found; skipping. Run jive.R to create them."
    else
        smfile = joinpath(refdir, "session_meta.csv")
        if isfile(smfile)
            sm = readdlm(smfile, ',', String; skipstart = 1)
            row = findfirst(==("r.jive_version"), sm[:, 1])
            row !== nothing && @info "JIVE fixtures generated against r.jive $(sm[row, 2])"
        end

        rd(f) = readdlm(joinpath(refdir, f), ',', Float64; skipstart = 1)
        X1 = rd("X1.csv"); X2 = rd("X2.csv")
        D1 = rd("D1.csv"); D2 = rd("D2.csv")
        J1 = rd("J1.csv"); J2 = rd("J2.csv")
        A1 = rd("A1.csv"); A2 = rd("A2.csv")
        meta = rd("meta.csv")
        rT = Int(meta[2]); r1T = Int(meta[3]); r2T = Int(meta[4])

        m = jive([X1, X2], rT, [r1T, r2T])

        # preprocessing matches r.jive's scaled input
        Xc = _preprocess([X1, X2])
        @test norm(Xc[1] .- D1) < tol_r
        @test norm(Xc[2] .- D2) < tol_r

        # given-ranks JIVE is deterministic ⇒ decomposition matches r.jive
        @test norm(m.J[1] .- J1) < tol_r
        @test norm(m.J[2] .- J2) < tol_r
        @test norm(m.A[1] .- A1) < tol_r
        @test norm(m.A[2] .- A2) < tol_r
    end
end

@testset "matches r.jive — auto ranks (offline fixtures)" begin
    refdir = joinpath(@__DIR__, "Data", "JIVE")
    if !isfile(joinpath(refdir, "X1s.csv"))
        @info "JIVE auto-rank fixtures not found; re-run jive.R to create them."
    else
        rd(f) = readdlm(joinpath(refdir, f), ',', Float64; skipstart = 1)
        X1s = rd("X1s.csv"); X2s = rd("X2s.csv")
        J1p = rd("J1p.csv"); J2p = rd("J2p.csv")
        ranks = vec(rd("ranks_perm.csv"))
        rJ_r = Int(ranks[1]); rA_r = Int.(ranks[2:end])

        Random.seed!(999)
        m = jive([X1s, X2s]; nperm = 100)

        println("  Julia : joint=$(m.r)  indiv=$(m.ri)")
        println("  r.jive: joint=$rJ_r  indiv=$rA_r")
        println("  match : ", m.r == rJ_r && m.ri == rA_r)

        # both implementations recover the SAME ranks on this well-separated data
        @test m.r == rJ_r
        @test m.ri == rA_r
        # decompositions can't be bit-identical (R/Julia RNG streams differ), so
        # compare the joint SUBSPACE via canonical correlation instead
        if m.r == rJ_r
            cc = canon(reduce(vcat, m.J)', vcat(J1p, J2p)', m.r)
            @test minimum(cc) > 1 - tol_r        # subspace agreement (cross-language)
        end
    end
end