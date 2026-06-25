# Test/splsda_test.jl — formal tests for splsda (sparse PLS-DA) and its
# supporting functions: _center_scale, _unmap, _soft_threshold_L1!, _sqdiff.
# Tolerances (tol_ord / tol_julia / tol_r) are defined in runtests.jl.

const BRS = BigRiverSchneider
const splsda          = BRS.splsda
const splsdaStructure = BRS.splsdaStructure
const _cs    = BRS._center_scale
const _unmap = BRS._unmap
const _soft  = BRS._soft_threshold_L1!
const _sqd   = BRS._sqdiff

@testset "output structure & invariants" begin
    Random.seed!(1)
    n, p, ncomp = 60, 100, 2
    y = repeat(["A", "B", "C"], inner = 20)
    X = randn(n, p)
    m = splsda(X, y, ncomp, [10, 10])

    @test m isa splsdaStructure
    @test size(m.variates_X) == (n, ncomp)
    @test size(m.variates_Y) == (n, ncomp)
    @test size(m.loadings_X) == (p, ncomp)
    @test size(m.loadings_Y) == (3, ncomp)        # 3 classes
    @test m.ncomp == ncomp
    @test m.keepX == [10, 10]
    @test m.classes == ["A", "B", "C"]            # sorted unique
    @test size(m.Y_dummy) == (n, 3)
    # each X-loading column keeps exactly keepX nonzeros, is unit-norm
    for c in 1:ncomp
        @test count(!iszero, m.loadings_X[:, c]) == 10
        @test isapprox(norm(m.loadings_X[:, c]), 1.0; atol = tol_ord)
    end
end

@testset "keepX controls sparsity exactly" begin
    Random.seed!(2)
    y = repeat(["A", "B"], inner = 25)
    X = randn(50, 80)
    for kx in (5, 20, 50)
        m = splsda(X, y, 1, [kx])
        @test count(!iszero, m.loadings_X[:, 1]) == kx
    end
end

@testset "Y_dummy is a valid one-hot encoding" begin
    Random.seed!(3)
    y = repeat(["A", "B", "C"], inner = 15)
    m = splsda(randn(45, 30), y, 1, [5])
    @test all(sum(m.Y_dummy, dims = 2) .== 1.0)   # exactly one 1 per row
    @test all(x -> x == 0.0 || x == 1.0, m.Y_dummy)
    # column c is the indicator for class c
    for (c, cls) in enumerate(m.classes)
        @test m.Y_dummy[:, c] == Float64.(y .== cls)
    end
end

@testset "ground truth: selects discriminative variables & separates classes" begin
    # first 10 variables carry class-specific signal; 11:500 are pure noise.
    Random.seed!(123)
    classes = ["A", "B", "C"]; n_per = 20
    y = repeat(classes, inner = n_per); n = length(y); p = 500
    X = randn(n, p) .* 0.5
    for (ci, cls) in enumerate(classes)
        X[findall(==(cls), y), 1:10] .+= ci * 2.0
    end
    m = splsda(X, y, 2, [10, 10])

    sel = findall(!iszero, m.loadings_X[:, 1])
    @test length(intersect(sel, 1:10)) >= 8       # recovery floor: recovers signal vars
    # component-1 scores separate the classes (between/within ≫ 1)
    sc = m.variates_X[:, 1]; grand = mean(sc)
    between = sum(n_per * (mean(sc[findall(==(c), y)]) - grand)^2 for c in classes)
    within  = sum(sum((sc[i] - mean(sc[findall(==(y[i]), y)]))^2
                      for i in findall(==(c), y)) for c in classes)
    @test between / within > 10                   # separation floor
end

@testset "levels controls class ordering" begin
    Random.seed!(4)
    y = repeat(["A", "B", "C"], inner = 10)
    m1 = splsda(randn(30, 20), y, 1, [5])
    m2 = splsda(randn(30, 20), y, 1, [5]; levels = ["C", "B", "A"])
    @test m1.classes == ["A", "B", "C"]
    @test m2.classes == ["C", "B", "A"]
    # the dummy columns permute accordingly
    @test m2.Y_dummy[:, 1] == Float64.(y .== "C")
end

@testset "argument validation" begin
    Random.seed!(0)
    y = repeat(["A", "B"], inner = 10); X = randn(20, 15)
    @test_throws ArgumentError splsda(X, y, 2, [5])          # keepX length ≠ ncomp
    @test_throws ArgumentError splsda(X, y, 1, [5]; levels = ["A", "B", "C"])  # wrong #levels
    @test_throws ArgumentError splsda(X, y, 1, [5]; levels = ["A", "Z"])       # class not in levels
end

# ----------------------------------------------------------------------------
# internal helpers
# ----------------------------------------------------------------------------

@testset "internal: _center_scale" begin
    Random.seed!(5)
    M = randn(40, 8) .* (1:8)' .+ (1:8)'
    Cs = _cs(M; scale = true)
    @test all(abs.(vec(mean(Cs, dims = 1))) .< tol_ord)         # columns centered
    @test all(isapprox.(vec(std(Cs, dims = 1; corrected = true)), 1.0; atol = tol_ord))
    Cc = _cs(M; scale = false)
    @test all(abs.(vec(mean(Cc, dims = 1))) .< tol_ord)         # centered, not scaled
    @test !all(isapprox.(vec(std(Cc, dims = 1)), 1.0; atol = tol_ord))
    # constant column ⇒ zero std guarded (no NaN/Inf), set to 0
    Mz = hcat(randn(20), fill(3.0, 20))
    Csz = _cs(Mz; scale = true)
    @test all(isfinite, Csz)
    @test all(Csz[:, 2] .== 0.0)
end

@testset "internal: _unmap (one-hot encoder)" begin
    y = ["B", "A", "C", "A", "B"]
    Yd, classes = _unmap(y)
    @test classes == ["A", "B", "C"]                            # sorted
    @test Yd == [0 1 0; 1 0 0; 0 0 1; 1 0 0; 0 1 0]
    # custom levels reorder the columns
    Yd2, classes2 = _unmap(y; levels = ["C", "B", "A"])
    @test classes2 == ["C", "B", "A"]
    @test Yd2[1, :] == [0, 1, 0]                                # "B" → column 2
    @test_throws ArgumentError _unmap(y; levels = ["A", "B"])  # too few levels
    @test_throws ArgumentError _unmap(y; levels = ["A", "B", "Z"])  # missing class
end

@testset "internal: _soft_threshold_L1! (keeps p−nx largest, zeros nx smallest)" begin
    x = [5.0, -3.0, 1.0, -0.5, 2.0]              # |x| = 5,3,1,0.5,2
    out = similar(x)
    absx = similar(x); ord = similar(x, Int); ranks = similar(x, Int)
    # drop the nx=2 smallest-magnitude (|−0.5|, |1|) ⇒ those two become 0
    _soft(out, x, 2, absx, ord, ranks)
    @test out[4] == 0.0 && out[3] == 0.0          # the two smallest zeroed
    @test all(out[[1, 2, 5]] .!= 0.0)             # the three largest survive
    @test sign.(out[[1, 2, 5]]) == sign.(x[[1, 2, 5]])   # signs preserved
    # survivors are soft-thresholded by λ = largest dropped magnitude (=1.0)
    @test out[1] ≈ 5.0 - 1.0
    @test out[5] ≈ 2.0 - 1.0
    @test out[2] ≈ -(3.0 - 1.0)
    # nx ≤ 0 ⇒ no thresholding, copy through
    o2 = similar(x); _soft(o2, x, 0, absx, ord, ranks)
    @test o2 == x
end

@testset "internal: _sqdiff (squared L2 distance)" begin
    @test _sqd([1.0, 2.0], [1.0, 2.0]) == 0.0
    @test _sqd([0.0, 0.0], [3.0, 4.0]) == 25.0
    a = randn(30); b = randn(30)
    @test _sqd(a, b) ≈ sum(abs2, a .- b)
end

# ----------------------------------------------------------------------------
# matches mixOmics — offline fixtures
# ----------------------------------------------------------------------------

@testset "matches mixOmics::splsda (offline reference fixtures)" begin
    refdir = joinpath(@__DIR__, "Data", "SPLSDA")
    if !isfile(joinpath(refdir, "X.csv"))
        @info "sPLS-DA mixOmics fixtures not found; run splsda.R to create them."
    else
        smfile = joinpath(refdir, "session_meta.csv")
        if isfile(smfile)
            sm = readdlm(smfile, ',', String; skipstart = 1)
            row = findfirst(==("mixOmics_version"), sm[:, 1])
            row !== nothing && @info "sPLS-DA fixtures generated against mixOmics $(sm[row, 2])"
        end

        rdf(f) = readdlm(joinpath(refdir, f), ',', Float64; skipstart = 1)
        rds(f) = vec(readdlm(joinpath(refdir, f), ',', String; skipstart = 1))
        X    = rdf("X.csv")
        y    = rds("Y.csv")
        lx   = rdf("lx.csv"); ly = rdf("ly.csv")
        vx   = rdf("vx.csv"); vy = rdf("vy.csv")
        levs = rds("levels.csv")
        meta = rdf("meta.csv")
        ncomp = Int(meta[1]); keepX = [Int(meta[2]), Int(meta[3])]

        m = splsda(X, y, ncomp, keepX; levels = levs)

        for c in 1:ncomp
            # sign-invariant: correlation magnitude ≈ 1 (per-component SVD sign differs)
            @test abs(cor(m.loadings_X[:, c], lx[:, c])) > 1 - tol_r
            @test abs(cor(m.variates_X[:, c], vx[:, c])) > 1 - tol_r
            @test abs(cor(m.loadings_Y[:, c], ly[:, c])) > 1 - tol_r
            @test abs(cor(m.variates_Y[:, c], vy[:, c])) > 1 - tol_r
            # the SET of selected variables matches exactly
            @test Set(findall(!iszero, m.loadings_X[:, c])) == Set(findall(!iszero, lx[:, c]))
        end
    end
end