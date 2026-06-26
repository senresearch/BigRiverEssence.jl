# bench_plskern.jl — verify BigRiverSchneider.plskern matches Jchemo.plskern, then
# benchmark. NOTE: plskern is in-place (overwrites X, Y), so we pass copies anywhere
# the original data is reused.
# Run:  julia --project bench_plskern.jl

using BigRiverSchneider
using LinearAlgebra, Statistics, Random
using BenchmarkTools
import Jchemo

const myplskern = BigRiverSchneider.plskern        # alias around the Jchemo name clash
const mycoef    = BigRiverSchneider.plskerncoef
const mypredict = BigRiverSchneider.plskernpredict

function run_case(n, p, nlv; q = 1, standardize = false, seed = 1)
    Random.seed!(seed)
    X = randn(n, p)
    Ymat = q == 1 ? reshape(randn(n), :, 1) : randn(n, q)   # plskern needs a matrix Y

    println("="^70)
    println("n=$n  p=$p  q=$q  nlv=$nlv  standardize=$standardize")
    println("="^70)

    # ---- fit mine on COPIES (plskern overwrites its inputs) ----
    m = myplskern(copy(X), copy(Ymat); nlv = nlv, standardize = standardize, method = :algo1)
    B_mine, _ = mycoef(m)
    ŷ_mine    = mypredict(m, X)            # X here is the untouched original

    # ---- fit Jchemo (it copies internally; pass the originals) ----
    mod = Jchemo.plskern(; nlv = nlv, scal = standardize)
    Jchemo.fit!(mod, X, q == 1 ? vec(Ymat) : Ymat)
    B_jc = Jchemo.coef(mod).B
    ŷ_jc = Jchemo.predict(mod, X).pred
    T_jc = Jchemo.transf(mod, X)

    # ---- identical-results check ----
    dB = maximum(abs.(B_mine .- B_jc))
    dŷ = maximum(abs.(ŷ_mine .- ŷ_jc))
    dT = maximum(1 - abs(dot(normalize(m.T[:, a]), normalize(T_jc[:, a]))) for a in 1:nlv)
    println("  max |Δ B|            : ", dB)
    println("  max |Δ ŷ|            : ", dŷ)
    println("  max scores misalign  : ", dT, "   (1 - |cos|, sign-invariant)")
    identical = dB < 1e-8 && dŷ < 1e-8 && dT < 1e-8
    println("  IDENTICAL vs Jchemo  : ", identical ? "✓ yes" : "✗ NO — timings not comparable")
    println()

    # ---- benchmarks ----
    # mine overwrites X,Y, so hand it FRESH copies each sample via setup= (not timed)
    print("  mine  :algo1 : ")
    @btime myplskern(Xc, Yc; nlv = $nlv, standardize = $standardize, method = :algo1) setup = (Xc = copy($X); Yc = copy($Ymat));
    print("  mine  :algo2 : ")
    @btime myplskern(Xc, Yc; nlv = $nlv, standardize = $standardize, method = :algo2) setup = (Xc = copy($X); Yc = copy($Ymat));
    print("  Jchemo       : ")
    @btime Jchemo.fit!(Jchemo.plskern(; nlv = $nlv, scal = $standardize), $X, $(q == 1 ? vec(Ymat) : Ymat));
    println()
end

run_case(100,  20,  10)
run_case(400,  50,  12)
run_case(2000, 100, 20)
run_case(200,  500, 30)                      # wide: p > n
run_case(5000, 200, 25)
run_case(400,  50,  12; q = 3)               # multi-response
run_case(400,  50,  12; standardize = true)  # exercises the live scaling-division path

#=
======================================================================
n=100  p=20  q=1  nlv=10  standardize=false
======================================================================
  max |Δ B|            : 9.71445146547012e-17
  max |Δ ŷ|            : 5.551115123125783e-16
  max scores misalign  : 7.771561172376096e-16   (1 - |cos|, sign-invariant)
  IDENTICAL vs Jchemo  : ✓ yes

  mine  :algo1 :   10.750 μs (45 allocations: 16.45 KiB)
  mine  :algo2 :   11.542 μs (48 allocations: 20.03 KiB)
  Jchemo       :   77.458 μs (671 allocations: 60.77 KiB)

======================================================================
n=400  p=50  q=1  nlv=12  standardize=false
======================================================================
  max |Δ B|            : 1.1102230246251565e-16
  max |Δ ŷ|            : 8.881784197001252e-16
  max scores misalign  : 6.661338147750939e-16   (1 - |cos|, sign-invariant)
  IDENTICAL vs Jchemo  : ✓ yes

  mine  :algo1 :   106.375 μs (49 allocations: 86.94 KiB)
  mine  :algo2 :   102.750 μs (52 allocations: 107.02 KiB)
  Jchemo       :   206.208 μs (859 allocations: 293.38 KiB)

======================================================================
n=2000  p=100  q=1  nlv=20  standardize=false
======================================================================
  max |Δ B|            : 5.551115123125783e-17
  max |Δ ŷ|            : 9.43689570931383e-16
  max scores misalign  : 1.9984014443252818e-15   (1 - |cos|, sign-invariant)
  IDENTICAL vs Jchemo  : ✓ yes

  mine  :algo1 :   1.564 ms (49 allocations: 391.12 KiB)
  mine  :algo2 :   1.186 ms (52 allocations: 487.20 KiB)
  Jchemo       :   1.975 ms (1820 allocations: 2.05 MiB)

======================================================================
n=200  p=500  q=1  nlv=30  standardize=false
======================================================================
  max |Δ B|            : 1.4085954624931674e-15
  max |Δ ŷ|            : 3.552713678800501e-15
  max scores misalign  : 7.771561172376096e-16   (1 - |cos|, sign-invariant)
  IDENTICAL vs Jchemo  : ✓ yes

  mine  :algo1 :   1.288 ms (54 allocations: 475.70 KiB)
  mine  :algo2 :   2.722 ms (57 allocations: 2.40 MiB)
  Jchemo       :   1.556 ms (3561 allocations: 1.40 MiB)

======================================================================
n=5000  p=200  q=1  nlv=25  standardize=false
======================================================================
  max |Δ B|            : 6.938893903907228e-17
  max |Δ ŷ|            : 1.1657341758564144e-15
  max scores misalign  : 2.9976021664879227e-15   (1 - |cos|, sign-invariant)
  IDENTICAL vs Jchemo  : ✓ yes

  mine  :algo1 :   3.459 ms (49 allocations: 1.23 MiB)
  mine  :algo2 :   3.828 ms (52 allocations: 1.54 MiB)
  Jchemo       :   6.389 ms (2615 allocations: 9.24 MiB)

======================================================================
n=400  p=50  q=3  nlv=12  standardize=false
======================================================================
  max |Δ B|            : 1.942890293094024e-16
  max |Δ ŷ|            : 1.1102230246251565e-15
  max scores misalign  : 6.661338147750939e-16   (1 - |cos|, sign-invariant)
  IDENTICAL vs Jchemo  : ✓ yes

  mine  :algo1 :   149.833 μs (217 allocations: 151.16 KiB)
  mine  :algo2 :   144.292 μs (220 allocations: 171.23 KiB)
  Jchemo       :   251.333 μs (1027 allocations: 364.38 KiB)

======================================================================
n=400  p=50  q=1  nlv=12  standardize=true
======================================================================
  max |Δ B|            : 6.938893903907228e-17
  max |Δ ŷ|            : 7.771561172376096e-16
  max scores misalign  : 5.551115123125783e-16   (1 - |cos|, sign-invariant)
  IDENTICAL vs Jchemo  : ✓ yes

  mine  :algo1 :   112.083 μs (67 allocations: 88.38 KiB)
  mine  :algo2 :   108.958 μs (70 allocations: 108.45 KiB)
  Jchemo       :   264.208 μs (881 allocations: 295.31 KiB)
=#
