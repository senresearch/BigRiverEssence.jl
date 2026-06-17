using BigRiverSchneider, LinearAlgebra, Statistics, Random
Random.seed!(123)

#  construct data with KNOWN discriminative variables 
n_per = 20                          # samples per class
classes = ["A", "B", "C"]
y = repeat(classes, inner=n_per)    # 60 samples, 3 classes
n = length(y)
p = 500                             # 500 variables (genes)

# the FIRST 10 variables are the "true" discriminative ones:
# each class gets a distinct mean shift on those 10 variables.
X = randn(n, p) .* 0.5              # baseline noise on all 500 variables
true_vars = 1:10
for (ci, cls) in enumerate(classes)
    rows = findall(==(cls), y)  #  which rows (samples) belong to this class. 
    # class ci gets a +signal on a distinct subset of the 10 true variables
    X[rows, true_vars] .+= ci * 2.0   # strong, class-specific shift on vars 1-10
end
# variables 11:500 are pure noise — NOT discriminative

#  fit sPLS-DA, asking it to keep 10 variables on component 1 
res = splsda(X, y, 2, [10, 10])

#  CHECK 1: did it select the TRUE discriminative variables? 
selected_c1 = findall(!iszero, res.loadings_X[:,1])
println("True discriminative vars : ", collect(true_vars))
println("Selected by component 1  : ", sort(selected_c1))
overlap = length(intersect(selected_c1, true_vars))
println("Overlap with truth       : $overlap / 10")

#  CHECK 2: do the classes separate in the scores? 
# compute how well component-1 scores separate the classes (between vs within variance)
sc1 = res.variates_X[:,1]
grand = mean(sc1)
between = sum(n_per * (mean(sc1[findall(==(c),y)]) - grand)^2 for c in classes)
within  = sum(sum((sc1[i] - mean(sc1[findall(==(y[i]),y)]))^2 for i in findall(==(c),y)) for c in classes)
println("Class separation (between/within, higher=better): ", round(between/within, digits=2))










using RCall, BigRiverSchneider
using LinearAlgebra, Statistics, Random



R"""
library(mixOmics)
data(srbct)
# This data set from Khan et al., (2001) gives the expression measure of 2308 genes measured on 63 samples.
X <- srbct$gene[1:60, 1:200]
# data frame with 63 rows and 2308 columns. The expression measure of 2308 genes for the 63 subjects.
Y <- srbct$class[1:60]
# A class vector containing the class tumour of each case (4 classes in total).
res <- splsda(X, Y, ncomp=2, keepX=c(15,15))
vx <- res$variates$X; lx <- res$loadings$X
vy <- res$variates$Y; ly <- res$loadings$Y
levs <- levels(srbct$class)
"""
@rget X Y vx lx vy ly levs
levs = string.(levs)

# fit with mixOmics' class ordering
mine = splsda(Float64.(X), string.(Y), 2, [15,15]; levels=levs)

# verify all five outputs (abs handles arbitrary per-component SVD signs)
for c in 1:2
    sel_match = Set(findall(!iszero, mine.loadings_X[:,c])) == Set(findall(!iszero, lx[:,c]))
    println("comp $c | X-load: ", round(abs(cor(mine.loadings_X[:,c], lx[:,c])), digits=6),
            "  X-var: ", round(abs(cor(mine.variates_X[:,c], vx[:,c])), digits=6),
            "  Y-load: ", round(abs(cor(mine.loadings_Y[:,c], ly[:,c])), digits=6),
            "  Y-var: ", round(abs(cor(mine.variates_Y[:,c], vy[:,c])), digits=6),
            "  sel: ", sel_match)
end



using BenchmarkTools

# --- Julia timing ---
println("="^55); println("BENCHMARK: splsda — mine (Julia) vs mixOmics (R)"); println("="^55)

print("mine (Julia): ")
@btime splsda(Float64.($X), string.($Y), 2, [15,15]; levels=$levs);

# --- mixOmics timing via microbenchmark ---
R"""
library(microbenchmark)
mb <- microbenchmark(
  splsda(X, Y, ncomp=2, keepX=c(15,15)),
  times=20)
cat("mixOmics (R): median", round(median(mb$time)/1e6, 3), "ms\n")
"""


R"""
Xbig <- srbct$gene; Ybig <- srbct$class; levsbig <- levels(srbct$class)
"""
@rget Xbig Ybig levsbig
levsbig = string.(levsbig)

print("mine big (Julia): ")
@btime splsda(Float64.($Xbig), string.($Ybig), 3, [50,50,50]; levels=$levsbig);
R"""
mb2 <- microbenchmark(splsda(Xbig, Ybig, ncomp=3, keepX=c(50,50,50)), times=10)
cat("mixOmics big (R): median", round(median(mb2$time)/1e6, 3), "ms\n")
"""



#=
True discriminative vars : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Selected by component 1  : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Overlap with truth       : 10 / 10
Class separation (between/within, higher=better): 164.56
comp 1 | X-load: 1.0  X-var: 1.0  Y-load: 1.0  Y-var: 1.0  sel: true
comp 2 | X-load: 1.0  X-var: 1.0  Y-load: 1.0  Y-var: 0.999937  sel: true
=======================================================
BENCHMARK: splsda — mine (Julia) vs mixOmics (R)
=======================================================
mine (Julia):   217.583 μs (915 allocations: 1.26 MiB)
mixOmics (R): median 5.09 ms
mine big (Julia):   6.031 ms (1879 allocations: 19.79 MiB)
mixOmics big (R): median 30.216 ms
RObject{NilSxp}
NULL
=#

















# test/TestSplsdaOpt.jl — full verification: splsda_opt vs splsda, + benchmark
using RCall, BigRiverSchneider
using LinearAlgebra, Statistics, Random
using BenchmarkTools

# ---- pull data + reference from R (small and large) ----
R"""
library(mixOmics)
data(srbct)
Xsmall <- srbct$gene[1:60, 1:200]
Ysmall <- srbct$class[1:60]
Xbig   <- srbct$gene
Ybig   <- srbct$class
levs   <- levels(srbct$class)
"""
@rget Xsmall Ysmall Xbig Ybig levs
levs = string.(levs)

# ---- helper: compare two SplsdaResults field by field ----
# loadings/variates have arbitrary per-component SVD signs → compare via abs.
function compare_splsda(a, b; label="")
    println("--- $label ---")
    nc = a.ncomp
    for c in 1:nc
        println("  comp $c:")
        println("    X-load ‖diff‖(abs): ", norm(abs.(a.loadings_X[:,c]) .- abs.(b.loadings_X[:,c])))
        println("    Y-load ‖diff‖(abs): ", norm(abs.(a.loadings_Y[:,c]) .- abs.(b.loadings_Y[:,c])))
        println("    X-var  ‖diff‖(abs): ", norm(abs.(a.variates_X[:,c]) .- abs.(b.variates_X[:,c])))
        println("    Y-var  ‖diff‖(abs): ", norm(abs.(a.variates_Y[:,c]) .- abs.(b.variates_Y[:,c])))
        sel_a = Set(findall(!iszero, a.loadings_X[:,c]))
        sel_b = Set(findall(!iszero, b.loadings_X[:,c]))
        println("    selected-var match: ", sel_a == sel_b)
    end
    println("  Y_dummy ‖diff‖: ", norm(a.Y_dummy .- b.Y_dummy))
    println("  classes match : ", a.classes == b.classes)
    println("  ncomp match   : ", a.ncomp == b.ncomp)
    println("  keepX match   : ", a.keepX == b.keepX)
end

# ============================================================
# CORRECTNESS: splsda_opt vs splsda
# ============================================================
println("="^60)
println("CORRECTNESS: splsda_opt vs splsda")
println("="^60)

# small
ref_s = splsda(Float64.(Xsmall), string.(Ysmall), 2, [15,15]; levels=levs)
opt_s = splsda_opt(Float64.(Xsmall), string.(Ysmall), 2, [15,15]; levels=levs)
compare_splsda(ref_s, opt_s; label="small (60×200), ncomp=2, keepX=[15,15]")

# large
ref_b = splsda(Float64.(Xbig), string.(Ybig), 3, [50,50,50]; levels=levs)
opt_b = splsda_opt(Float64.(Xbig), string.(Ybig), 3, [50,50,50]; levels=levs)
compare_splsda(ref_b, opt_b; label="big (full srbct), ncomp=3, keepX=[50,50,50]")

# ============================================================
# CROSS-CHECK: opt still matches mixOmics (the reference)
# ============================================================
println("\n", "="^60)
println("CROSS-CHECK: splsda_opt vs mixOmics (reference)")
println("="^60)
R"""
res <- splsda(Xsmall, Ysmall, ncomp=2, keepX=c(15,15))
lx <- res$loadings$X; vx <- res$variates$X
"""
@rget lx vx
for c in 1:2
    sel = Set(findall(!iszero, opt_s.loadings_X[:,c])) == Set(findall(!iszero, lx[:,c]))
    println("  comp $c | X-load |cor|: ", round(abs(cor(opt_s.loadings_X[:,c], lx[:,c])), digits=6),
            "  X-var |cor|: ", round(abs(cor(opt_s.variates_X[:,c], vx[:,c])), digits=6),
            "  sel: ", sel)
end

# ============================================================
# BENCHMARK
# ============================================================
println("\n", "="^60)
println("BENCHMARK: time / allocations / memory")
println("="^60)

println("\n[small: 60×200, ncomp=2, keepX=15]")
print("orig: "); @btime splsda($(Float64.(Xsmall)), $(string.(Ysmall)), 2, [15,15]; levels=$levs);
print("opt : "); @btime splsda_opt($(Float64.(Xsmall)), $(string.(Ysmall)), 2, [15,15]; levels=$levs);

println("\n[big: full srbct, ncomp=3, keepX=50]")
print("orig: "); @btime splsda($(Float64.(Xbig)), $(string.(Ybig)), 3, [50,50,50]; levels=$levs);
print("opt : "); @btime splsda_opt($(Float64.(Xbig)), $(string.(Ybig)), 3, [50,50,50]; levels=$levs);

#=
============================================================
CORRECTNESS: splsda_opt vs splsda
============================================================
--- small (60×200), ncomp=2, keepX=[15,15] ---
  comp 1:
    X-load ‖diff‖(abs): 0.0
    Y-load ‖diff‖(abs): 0.0
    X-var  ‖diff‖(abs): 0.0
    Y-var  ‖diff‖(abs): 0.0
    selected-var match: true
  comp 2:
    X-load ‖diff‖(abs): 0.0
    Y-load ‖diff‖(abs): 3.1031676915590914e-17
    X-var  ‖diff‖(abs): 1.1431288060698758e-15
    Y-var  ‖diff‖(abs): 4.726604209672303e-16
    selected-var match: true
  Y_dummy ‖diff‖: 0.0
  classes match : true
  ncomp match   : true
  keepX match   : true
--- big (full srbct), ncomp=3, keepX=[50,50,50] ---
  comp 1:
    X-load ‖diff‖(abs): 0.0
    Y-load ‖diff‖(abs): 0.0
    X-var  ‖diff‖(abs): 0.0
    Y-var  ‖diff‖(abs): 0.0
    selected-var match: true
  comp 2:
    X-load ‖diff‖(abs): 6.133865909644874e-16
    Y-load ‖diff‖(abs): 1.5716250041522723e-16
    X-var  ‖diff‖(abs): 9.654503412841368e-15
    Y-var  ‖diff‖(abs): 1.5468598394511055e-15
    selected-var match: true
  comp 3:
    X-load ‖diff‖(abs): 1.1308830723435884e-15
    Y-load ‖diff‖(abs): 2.298102569334139e-16
    X-var  ‖diff‖(abs): 1.1814598463173429e-14
    Y-var  ‖diff‖(abs): 2.3544025921526978e-15
    selected-var match: true
  Y_dummy ‖diff‖: 0.0
  classes match : true
  ncomp match   : true
  keepX match   : true

============================================================
CROSS-CHECK: splsda_opt vs mixOmics (reference)
============================================================
  comp 1 | X-load |cor|: 1.0  X-var |cor|: 1.0  sel: true
  comp 2 | X-load |cor|: 1.0  X-var |cor|: 1.0  sel: true

============================================================
BENCHMARK: time / allocations / memory
============================================================

[small: 60×200, ncomp=2, keepX=15]
orig:   219.042 μs (910 allocations: 1.17 MiB)
opt :   202.708 μs (209 allocations: 493.52 KiB)

[big: full srbct, ncomp=3, keepX=50]
orig:   5.869 ms (1874 allocations: 18.66 MiB)
opt :   5.189 ms (304 allocations: 6.08 MiB)
=#

