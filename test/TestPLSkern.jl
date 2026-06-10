
using BigRiverSchneider
using LinearAlgebra, Statistics, Random
Random.seed!(1234)

# data with real structure so PLS has a clear signal to find
n, p, q = 200, 20, 1
Xlatent = randn(n, 3)  # 3 latent variables
X = Xlatent * randn(3, p) .+ 0.1 .* randn(n, p)  # same as PCA etc..
ytrue = Xlatent * randn(3) .+ 0.05 .* randn(n)      # y depends on the same latent signals

println("X is $n × $p,  q = $q\n")

# ---------------------------------------------------------------------------
# TEST 1 — Full-rank PLS equals Ordinary Least Squares.
# With nlv = p components, PLS uses all the latent directions, so its
# predictions must match the OLS fit exactly. This is the key correctness
# anchor: a full PLS model IS the least-squares regression.
# ---------------------------------------------------------------------------
nlv_full = p
m  = BigRiverSchneider.plskern(X, ytrue; nlv = nlv_full, method = :algo1)
B, intercept = BigRiverSchneider.plskerncoef(m)

# OLS reference: fit y on centered X directly (closed form)
Xc   = X .- mean(X, dims = 1)
yc   = ytrue .- mean(ytrue)
B_ols = Xc \ yc                                     # least-squares coefficients
ŷ_pls = vec(BigRiverSchneider.plskernpredict(m, X))
ŷ_ols = mean(ytrue) .+ Xc * B_ols

println("TEST 1  full-rank PLS vs OLS")
println("  max |prediction diff|     : ", round(maximum(abs.(ŷ_pls .- ŷ_ols)), digits = 10))
println("  (want ≈ 0 — full PLS = OLS)\n")

# ---------------------------------------------------------------------------
# TEST 2 — algo1 and algo2 give the SAME model.
# The paper proves all kernel variants are equivalent, so the coefficients
# must match to numerical precision.
# ---------------------------------------------------------------------------
m1 = BigRiverSchneider.plskern(X, ytrue; nlv = 10, method = :algo1)
m2 = BigRiverSchneider.plskern(X, ytrue; nlv = 10, method = :algo2)
B1, _ = BigRiverSchneider.plskerncoef(m1)
B2, _ = BigRiverSchneider.plskerncoef(m2)
println("TEST 2  algo1 vs algo2")
println("  max |coef diff|           : ", round(maximum(abs.(B1 .- B2)), digits = 12))
println("  (want ≈ 0 — same model)\n")

# ---------------------------------------------------------------------------
# TEST 3 — predict reproduces fitted values, and transform gives right shape.
# Sanity that the coef/predict/transform plumbing is consistent.
# ---------------------------------------------------------------------------
m3 = BigRiverSchneider.plskern(X, ytrue; nlv = 5, method = :algo1)
scores = BigRiverSchneider.plskerntransform(m3, X)
println("TEST 3  shapes & plumbing  (nlv = 5)")
println("  scores size               : ", size(scores), "   (want ($n, 5))")
println("  coef B size               : ", size(BigRiverSchneider.plskerncoef(m3)[1]), "   (want ($p, $q))")

# more components ⇒ better training fit (RMSE should decrease)
my_rmse(a, b) = sqrt(mean(abs2, a .- b))
errs = [my_rmse(vec(BigRiverSchneider.plskernpredict(BigRiverSchneider.plskern(X, ytrue; nlv = k), X)), ytrue) for k in 1:6]
println("  train RMSE by nlv (1..6)  : ", round.(errs, digits = 4))
println("  (want monotonically decreasing)\n")

# ---------------------------------------------------------------------------
# TEST 4 — multiple-Y works (exercises the SVD weight branch).
# ---------------------------------------------------------------------------
Y2 = hcat(ytrue, Xlatent * randn(3) .+ 0.05 .* randn(n))   # 2 responses
mY = BigRiverSchneider.plskern(X, Y2; nlv = 5, method = :algo1)
println("TEST 4  multiple Y (q = 2)")
println("  coef B size               : ", size(BigRiverSchneider.plskerncoef(mY)[1]), "   (want ($p, 2))")
println("  ran without error         : true")