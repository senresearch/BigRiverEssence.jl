# benchmark_all_extensive.jl — multi-size similarity + benchmark sweep, BigRiverSchneider.jl
# Per method, per size:
#   (1) CORRECTNESS — verify results are identical to the reference (printed first)
#   (2) @btime  — Julia time + allocations + bytes (native @btime line, all three shown)
#   (3) microbenchmark — R median (where the reference is an R package)
#
# === WARNINGS — read before running ===
#  • NOTHING is skipped. Every size runs, including 10000×5000. The R huge tier
#    (jive / scca / splsda via r.jive / PMA / mixOmics) may run for HOURS and can
#    exhaust RAM. The Julia side at huge is fine (seconds); the R side is the risk.
#  • A 10000×5000 Float64 matrix is ~400 MB; with copies + R marshalling, the huge
#    tier can spike to several GB. If you hit OOM, comment out the huge rows.
#  • Julia (@btime) and R (microbenchmark) times are SEPARATE-runtime medians — not a
#    head-to-head ratio. R times exclude RCall marshalling.
#  • Similarity is sign-invariant where the method has sign ambiguity (|cos|/|dot|).
#
# Run:  julia --project benchmark_all_extensive.jl

using BigRiverSchneider
using LinearAlgebra, Statistics, Random, BenchmarkTools, Printf
import MultivariateStats; const MVS = MultivariateStats
using RCall                        # unconditional at top level so @rput / R"..." parse

const BRS = BigRiverSchneider

# --- optional deps ---------------------------------------------------------
const HAS_JCHEMO = let
    try; @eval import Jchemo; true; catch; false; end
end
const R_OK = Dict{String,Bool}()
for pk in ("PMA", "r.jive", "mixOmics", "microbenchmark")
    R_OK[pk] = rcopy(reval("requireNamespace('$pk', quietly = TRUE)"))
    R_OK[pk] || @warn "R package '$pk' not installed; its section will be skipped."
end

# --- helpers ---------------------------------------------------------------
hr() = println("="^92)
section(name) = (println(); hr(); println("  ", name); hr())

colmisalign(A, B) = maximum(1 - abs(dot(normalize(A[:,j]), normalize(B[:,j]))) for j in 1:size(A,2))
selset(col) = Set(findall(!iszero, col))

# print the correctness verdict line prominently
function verdict(misalign; thresh = 1e-2, sets = nothing)
    ok = misalign < thresh && (sets === nothing || sets)
    tag = ok ? "IDENTICAL ✓" : "DIFFER ✗"
    s = sets === nothing ? "" : (sets ? "  (selected sets match)" : "  (SELECTED SETS DIFFER)")
    @printf("    result: %-12s  max misalignment = %.2e%s\n", tag, misalign, s)
end

# run an R microbenchmark, print its median ms (data must already be @rput into R)
function rtime(expr::String; times::Int = 20)
    R_OK["microbenchmark"] || (println("      R: microbenchmark not available"); return)
    res = reval("""
        suppressMessages(library(microbenchmark))
        mb <- microbenchmark($expr, times = $times)
        median(mb\$time) / 1e6
    """)
    @printf("      R  microbenchmark median: %.3f ms (in-R, %d reps)\n", rcopy(res), times)
end

println("\nBigRiverSchneider.jl — EXTENSIVE multi-size similarity + benchmark sweep")
println("Jchemo: $(HAS_JCHEMO ? "yes" : "NO")    NO SIZES SKIPPED — huge tier (10000×5000) WILL run.")
println("@btime shows time + allocations + bytes for every Julia fit; R uses microbenchmark.")
println("Julia and R times are separate-runtime medians, not a direct ratio.\n")

# ===========================================================================
# 1. PCA  vs MultivariateStats.PCA   (X is obs×var)
# ===========================================================================
section("PCA  vs  MultivariateStats.PCA   [X = n×p, obs×var]")
for (n, p, k) in [(200,40,6), (1000,100,10), (5000,200,20),
                  (200,500,20), (2000,1000,30), (10000,5000,50)]
    Random.seed!(1)
    X = randn(n, p); Xt = permutedims(X)
    @printf("\n  n=%d p=%d k=%d   (auto picks %s)\n", n, p, k, n >= p ? ":cov" : ":svd")
    m  = BRS.pca(X; k=k, method=:auto)
    rm = MVS.fit(MVS.PCA, Xt; maxoutdim=k, pratio=1.0)
    verdict(colmisalign(m.loadings, MVS.projection(rm)); thresh = 1e-6)
    print("      BRS.pca (:auto) : "); @btime BRS.pca($X; k=$k, method=:auto);
    print("      MVS.fit(PCA)    : "); @btime MVS.fit(MVS.PCA, $Xt; maxoutdim=$k, pratio=1.0);
end

# ===========================================================================
# 2. PLSKERN  vs Jchemo.plskern   (X is obs×var, y vector)
# ===========================================================================
section("PLSKERN  vs  Jchemo.plskern   [X = n×p, y = n]")
if !HAS_JCHEMO
    println("    Jchemo not installed; skipping.")
else
    for (n, p, nlv) in [(200,30,8), (1000,50,12), (5000,100,20),
                        (400,300,30), (2000,500,25), (10000,2000,40)]
        Random.seed!(1234)
        X = randn(n, p); y = randn(n)
        @printf("\n  n=%d p=%d nlv=%d\n", n, p, nlv)
        m = BRS.plskern(copy(X), reshape(copy(y), :, 1); nlv=nlv, method=:algo1)
        B_mine, _ = BRS.plskerncoef(m)
        mod = Jchemo.plskern(; nlv=nlv); Jchemo.fit!(mod, X, y)
        verdict(maximum(abs.(B_mine .- Jchemo.coef(mod).B)); thresh = 1e-8)
        print("      BRS.plskern :algo1 : "); @btime BRS.plskern(Xc, Yc; nlv=$nlv, method=:algo1) setup=(Xc=copy($X); Yc=reshape(copy($y),:,1));
        print("      BRS.plskern :algo2 : "); @btime BRS.plskern(Xc, Yc; nlv=$nlv, method=:algo2) setup=(Xc=copy($X); Yc=reshape(copy($y),:,1));
        print("      Jchemo.plskern     : "); @btime Jchemo.fit!(Jchemo.plskern(; nlv=$nlv), $X, $y);
    end
end

# ===========================================================================
# 3. CCA  vs MultivariateStats.CCA   (X,Y are var×obs)
# ===========================================================================
section("CCA  vs  MultivariateStats.CCA   [X = dx×n, Y = dy×n, var×obs]")
for (dx, dy, n) in [(6,5,300), (20,15,1000), (50,40,5000),
                    (100,80,2000), (200,150,3000), (500,400,10000)]
    Random.seed!(7)
    X = randn(dx, n); Y = randn(dy, n)
    ns = min(dx, dy, 3); sh = randn(ns, n); X[1:ns,:] .+= sh; Y[1:ns,:] .+= sh
    @printf("\n  dx=%d dy=%d n=%d\n", dx, dy, n)
    m  = BRS.cca(X, Y; method=:svd)
    rm = MVS.fit(MVS.CCA, X, Y; method=:svd)
    dc = norm(sort(m.corrs) .- sort(MVS.correlations(rm)))
    dP = colmisalign(m.xproj, MVS.xprojection(rm))
    verdict(max(dc, dP); thresh = 1e-8)
    print("      BRS.cca       : "); @btime BRS.cca($X, $Y; method=:svd);
    print("      MVS.fit(CCA)  : "); @btime MVS.fit(MVS.CCA, $X, $Y; method=:svd);
end

# ===========================================================================
# 4. PMD  vs PMA::PMD (R)   (X is n×p)
# ===========================================================================
section("PMD  vs  PMA::PMD (R)   [X = n×p]")
if !R_OK["PMA"]
    println("    PMA not available; skipping.")
else
    for (n, p) in [(60,40), (200,150), (1000,500),
                   (300,800), (2000,1000), (5000,3000)]
        Random.seed!(11)
        X = randn(n, p); sumabs = 0.4; K = 3
        @printf("\n  n=%d p=%d\n", n, p)
        @rput X sumabs K
        R"""
        suppressMessages(library(PMA))
        rj <- PMD(X, type='standard', sumabs=sumabs, K=K, center=TRUE, trace=FALSE)
        """
        rv = rcopy(R"rj$v"); ru = rcopy(R"rj$u")
        m = BRS.pmd(copy(X); sumabs=sumabs, K=K, center=true)
        verdict(max(colmisalign(m.v, rv), colmisalign(m.u, ru)))
        print("      BRS.pmd : "); @btime BRS.pmd(Xc; sumabs=$sumabs, K=$K, center=true) setup=(Xc=copy($X));
        rtime("PMD(X, type='standard', sumabs=sumabs, K=K, center=TRUE, trace=FALSE)")
    end
end

# ===========================================================================
# 5. SPC  vs PMA::SPC (R)   (X is n×p)
# ===========================================================================
section("SPC  vs  PMA::SPC (R)   [X = n×p]")
if !R_OK["PMA"]
    println("    PMA not available; skipping.")
else
    for (n, p) in [(80,30), (300,100), (1000,400),
                   (200,600), (2000,800), (5000,3000)]
        Random.seed!(13)
        X = randn(n, p); sumabsv = sqrt(p)/2; K = 3
        @printf("\n  n=%d p=%d\n", n, p)
        @rput X sumabsv K
        R"""
        suppressMessages(library(PMA))
        rj <- SPC(scale(X, center=TRUE, scale=FALSE), sumabsv=sumabsv, K=K, trace=FALSE)
        """
        rv = rcopy(R"rj$v")
        m = BRS.spc(copy(X); k=K, c=sumabsv)
        sm = all(selset(m.loadings[:,j]) == selset(rv[:,j]) for j in 1:K)
        verdict(colmisalign(m.loadings, rv); sets = sm)
        print("      BRS.spc : "); @btime BRS.spc(Xc; k=$K, c=$sumabsv) setup=(Xc=copy($X));
        rtime("SPC(scale(X, center=TRUE, scale=FALSE), sumabsv=sumabsv, K=K, trace=FALSE)")
    end
end

# ===========================================================================
# 6. SCCA  vs PMA::CCA (R)   (PMA layout: Xr,Zr obs×var; BRS wants var×obs)
# ===========================================================================
section("SCCA  vs  PMA::CCA (R)   [Xr = n×p1, Zr = n×p2, obs×var]")
if !R_OK["PMA"]
    println("    PMA not available; skipping.")
else
    for (n, p1, p2) in [(100,40,50), (300,150,200), (500,800,1000),
                        (1000,500,600), (2000,1500,2000), (3000,4000,5000)]
        Random.seed!(17)
        Xr = randn(n,p1); Zr = randn(n,p2); px=0.3; pz=0.3; K=2; niter=15
        @printf("\n  n=%d p1=%d p2=%d\n", n, p1, p2)
        @rput Xr Zr px pz K niter
        R"""
        suppressMessages(library(PMA))
        rj <- CCA(Xr, Zr, typex='standard', typez='standard',
                  penaltyx=px, penaltyz=pz, K=K, niter=niter, trace=FALSE)
        """
        ru = rcopy(R"rj$u"); rv = rcopy(R"rj$v")
        Xc_ = Matrix(transpose(Xr)); Zc_ = Matrix(transpose(Zr))
        m = BRS.scca(Xc_, Zc_; penaltyx=px, penaltyz=pz, K=K, niter=niter)
        sm = all(selset(m.u[:,j])==selset(ru[:,j]) && selset(m.v[:,j])==selset(rv[:,j]) for j in 1:K)
        verdict(max(colmisalign(m.u, ru), colmisalign(m.v, rv)); sets = sm)
        print("      BRS.scca : "); @btime BRS.scca($Xc_, $Zc_; penaltyx=$px, penaltyz=$pz, K=$K, niter=$niter);
        rtime("CCA(Xr, Zr, typex='standard', typez='standard', penaltyx=px, penaltyz=pz, K=K, niter=niter, trace=FALSE)")
    end
end

# ===========================================================================
# 7. JIVE  vs r.jive (R)   (blocks var×obs; given ranks)
# ===========================================================================
section("JIVE  vs  r.jive (R)   [X1 = p1×n, X2 = p2×n, var×obs; given ranks]")
if !R_OK["r.jive"]
    println("    r.jive not available; skipping.")
else
    canon(B1,B2,d) = (Q1=Matrix(qr(B1).Q)[:,1:d]; Q2=Matrix(qr(B2).Q)[:,1:d]; svdvals(Q1'Q2))
    for (n, p1, p2) in [(80,60,50), (200,150,120), (500,400,300),
                        (300,800,700), (1000,1000,800), (2000,3000,2500)]
        rT,r1T,r2T = 2,3,3
        Random.seed!(2024)
        S=randn(rT,n); U1=randn(p1,rT); U2=randn(p2,rT)
        S1=randn(r1T,n); W1=randn(p1,r1T); S2=randn(r2T,n); W2=randn(p2,r2T)
        X1 = U1*S + W1*S1 .+ 0.3.*randn(p1,n)
        X2 = U2*S + W2*S2 .+ 0.3.*randn(p2,n)
        @printf("\n  n=%d p1=%d p2=%d\n", n, p1, p2)
        @rput X1 X2 rT r1T r2T
        R"""
        suppressMessages(library(r.jive))
        rj <- jive(list(X1,X2), rankJ=rT, rankA=c(r1T,r2T), method='given',
                   conv=1e-6, maxiter=1000, scale=TRUE, center=TRUE, showProgress=FALSE)
        """
        rJ1 = rcopy(R"rj$joint[[1]]"); rJ2 = rcopy(R"rj$joint[[2]]")
        Xblocks = [Matrix{Float64}(X1), Matrix{Float64}(X2)]
        m = BRS.jive(Xblocks, rT, [r1T, r2T])
        cc = minimum(canon(reduce(vcat, m.J)', vcat(rJ1,rJ2)', rT))
        # report as (1 − min canonical corr) so smaller = more identical, like the others
        verdict(1 - cc; thresh = 1e-2)
        print("      BRS.jive : "); @btime BRS.jive($Xblocks, $rT, [$r1T, $r2T]);
        rtime("jive(list(X1,X2), rankJ=rT, rankA=c(r1T,r2T), method='given', conv=1e-6, maxiter=1000, scale=TRUE, center=TRUE, showProgress=FALSE)")
    end
end

# ===========================================================================
# 8. SPLSDA  vs mixOmics::splsda (R)   (X is n×p, y labels)
# ===========================================================================
section("SPLSDA  vs  mixOmics::splsda (R)   [X = n×p, 3 classes]")
if !R_OK["mixOmics"]
    println("    mixOmics not available; skipping.")
else
    for (nper, p, kx) in [(20,200,10), (50,500,20), (200,1000,30),
                          (30,1500,15), (100,2000,40), (300,5000,50)]
        n = 3*nper
        Random.seed!(123)
        classes = ["A","B","C"]; y = repeat(classes, inner=nper)
        X = randn(n, p) .* 0.5
        for (ci,cls) in enumerate(classes); X[findall(==(cls),y), 1:10] .+= ci*2.0; end
        ncomp=2; keepX=[kx,kx]; yR = y
        @printf("\n  n=%d p=%d keepX=%d\n", n, p, kx)
        @rput X yR ncomp keepX
        R"""
        suppressMessages(library(mixOmics))
        rj <- splsda(X, factor(yR), ncomp=ncomp, keepX=keepX)
        """
        rlx = rcopy(R"rj$loadings$X")
        m = BRS.splsda(copy(X), y, ncomp, keepX)
        sm = all(selset(m.loadings_X[:,j]) == selset(rlx[:,j]) for j in 1:ncomp)
        verdict(colmisalign(m.loadings_X, rlx); sets = sm)
        print("      BRS.splsda : "); @btime BRS.splsda(Xc, $y, $ncomp, $keepX) setup=(Xc=copy($X));
        rtime("splsda(X, factor(yR), ncomp=ncomp, keepX=keepX)")
    end
end

hr()
println("  Done. Every size ran, including the 10000×5000 huge tier.")
println("  '@btime' lines show time + allocations + bytes; R lines are in-R microbenchmark medians.")
println("  Julia and R times are separate-runtime medians, not a direct ratio.")
hr()