# test/TestJiveRjive.jl — thorough validation of jive_rjive (given + auto ranks)
using BigRiverSchneider, RCall, BenchmarkTools
using LinearAlgebra, Statistics, Random
Random.seed!(1234)

# ============================================================
# PART A — GIVEN RANKS: must still match r.jive bit-for-bit
# (confirms the refactor didn't break the validated path)
# ============================================================
R"""
library(r.jive); data(BRCA_data)
X1<-Data[[1]]; X2<-Data[[2]]; X3<-Data[[3]]
"""
@rget X1 X2 X3
nm = ["Expression","Methylation","miRNA"]

res = jive_rjive([X1,X2,X3], 2, [27,26,25])      # positional → given ranks

@rput X1 X2 X3
R"""
fit <- jive(list(X1,X2,X3), rankJ=2, rankA=c(27,26,25),
            method="given", scale=TRUE, center=TRUE, est=TRUE, orthIndiv=TRUE,
            showProgress=FALSE)
"""
R"J1<-fit$joint[[1]]; J2<-fit$joint[[2]]; J3<-fit$joint[[3]]"
R"A1<-fit$individual[[1]]; A2<-fit$individual[[2]]; A3<-fit$individual[[3]]"
R"d1<-fit$data[[1]]; d2<-fit$data[[2]]; d3<-fit$data[[3]]"
@rget J1 J2 J3 A1 A2 A3 d1 d2 d3
Jr=[J1,J2,J3]; Ar=[A1,A2,A3]; Dr=[d1,d2,d3]

println("="^64)
println("PART A — GIVEN RANKS: jive_rjive vs r.jive (should be bit-identical)")
println("="^64)
println("\nJ / A matrix differences (want 0.0):")
for i in 1:3
    println("  $(nm[i]): ‖J diff‖=", round(norm(res.J[i].-Jr[i]),digits=8),
            "  ‖A diff‖=", round(norm(res.A[i].-Ar[i]),digits=8))
end
ve(J,A,D)=(norm(J)^2/norm(D)^2, norm(A)^2/norm(D)^2, norm(D.-J.-A)^2/norm(D)^2)
println("\nvariance explained (yours vs r.jive):")
for i in 1:3
    println("  $(nm[i]): yours ", round.(ve(res.J[i],res.A[i],Dr[i]),digits=4),
            "  r.jive ", round.(ve(Jr[i],Ar[i],Dr[i]),digits=4))
end
jb(b...) = Matrix(qr(svd(vcat(b...)).Vt[1:2,:]').Q)[:,1:2]
println("\njoint subspace canonical corr: ", round.(svd(jb(res.J...)'*jb(Jr...)).S, digits=6))

# ============================================================
# PART B — GROUND TRUTH: synthetic data with known structure
# (does jive_rjive recover planted structure?)
# ============================================================
Random.seed!(7)
nG=100; rG,r1G,r2G = 2,3,3
SG=randn(rG,nG); U1=randn(40,rG); U2=randn(30,rG)
S1=randn(r1G,nG); W1=randn(40,r1G); S2=randn(r2G,nG); W2=randn(30,r2G)
G1=U1*SG+W1*S1; G2=U2*SG+W2*S2
resG = jive_rjive([G1,G2], rG, [r1G,r2G]; scale=false)
# scaled data jive_rjive used (scale=false → just centered)
Gc = [G .- mean(G,dims=2) for G in (G1,G2)]
println("\n", "="^64)
println("PART B — GROUND TRUTH (noiseless synthetic, given true ranks)")
println("="^64)
recon = sum(norm(Gc[i].-resG.J[i].-resG.A[i])^2 for i in 1:2)
println("  reconstruction ‖X−(J+A)‖² : ", round(recon, digits=10), "  (want ≈0)")
ortho = norm(vcat(resG.J...)*resG.A[1]') + norm(vcat(resG.J...)*resG.A[2]')
println("  orthogonality ‖J·Aᵀ‖      : ", round(ortho, digits=10), "  (want ≈0)")
Strue = SG .- mean(SG,dims=2)
Qt = Matrix(qr(Strue').Q)[:,1:rG]
Qm = Matrix(qr(svd(vcat(resG.J...)).Vt[1:rG,:]').Q)[:,1:rG]
println("  joint subspace vs truth   : ", round.(svd(Qm'*Qt).S, digits=6), "  (want ≈1)")

# ============================================================
# PART C — AUTO RANKS: permutation test estimates ranks
# (does the perm path find the vignette's ranks 2, [27,26,25]?)
# ============================================================
println("\n", "="^64)
println("PART C — AUTO RANKS via permutation (r.jive method=\"perm\")")
println("="^64)
Random.seed!(1234)
# NOTE: this is SLOW on full BRCA. Use fewer perms for a quick check, or subset.
res_auto = jive_rjive([X1,X2,X3]; nperm=50)      # ranks estimated
println("  estimated joint rank : ", res_auto.r)
println("  estimated indiv ranks: ", res_auto.ri)
println("  (vignette/given ranks were: 2, [27,26,25])")

# compare auto vs r.jive's own permutation estimate
R"""
set.seed(1234)
fitperm <- jive(list(X1,X2,X3), method="perm", est=TRUE, orthIndiv=TRUE,
                showProgress=FALSE)
rJ_r <- fitperm$rankJ; rA_r <- fitperm$rankA
"""
@rget rJ_r rA_r
println("  r.jive perm estimate : joint ", Int(rJ_r), ", indiv ", Int.(rA_r))

# ============================================================
# PART D — BENCHMARKING (given-ranks path, the fast one)
# ============================================================
println("\n", "="^64)
println("PART D — TIMING (given ranks)")
println("="^64)
print("  jive_rjive (Julia): ")
@btime jive_rjive([$X1,$X2,$X3], 2, [27,26,25]);
R"""
library(microbenchmark)
mb <- microbenchmark(
  jive(list(X1,X2,X3), rankJ=2, rankA=c(27,26,25), method="given",
       scale=TRUE, center=TRUE, est=TRUE, orthIndiv=TRUE, showProgress=FALSE),
  times=10)
cat("  r.jive (R)        :  median", round(median(mb$time)/1e6,2), "ms\n")
"""