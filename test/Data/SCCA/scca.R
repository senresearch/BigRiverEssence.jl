# generate_scca_reference.R
# PMA::CCA (sparse CCA) reference fixtures for the Julia test suite.
# RUN AS A FILE:  cd Test/Data/SCCA && Rscript scca.R
# Requires: install.packages("PMA")

suppressMessages(library(PMA))
set.seed(3189)

# planted sparse structure (matches your reference script)
uu <- matrix(c(rep(1, 25), rep(0, 75)), ncol = 1)
v1 <- matrix(c(rep(1, 50), rep(0, 450)), ncol = 1)
v2 <- matrix(c(rep(0, 50), rep(1, 50), rep(0, 900)), ncol = 1)
X <- uu %*% t(v1) + matrix(rnorm(100 * 500),  ncol = 500)
Z <- uu %*% t(v2) + matrix(rnorm(100 * 1000), ncol = 1000)

K <- 3; px <- 0.3; pz <- 0.3; niter <- 15
out <- CCA(X, Z, typex = "standard", typez = "standard", K = K,
           penaltyx = px, penaltyz = pz, niter = niter, trace = FALSE)

# X.csv / Z.csv are rows=obs (PMA layout); Julia transposes to columns=obs.
write.csv(X,         "X.csv",    row.names = FALSE)
write.csv(Z,         "Z.csv",    row.names = FALSE)
write.csv(out$u,     "u.csv",    row.names = FALSE)   # 500 × K
write.csv(out$v,     "v.csv",    row.names = FALSE)   # 1000 × K
write.csv(out$d,     "d.csv",    row.names = FALSE)
write.csv(out$cors,  "cors.csv", row.names = FALSE)
write.csv(data.frame(K = K, penaltyx = px, penaltyz = pz, niter = niter),
          "meta.csv", row.names = FALSE)

# provenance
writeLines(capture.output(sessionInfo()), "sessionInfo.txt")
env <- data.frame(
  field = c("R_version", "platform", "PMA_version", "generated"),
  value = c(R.version$version.string, sessionInfo()$platform,
            as.character(packageVersion("PMA")), as.character(Sys.time())))
write.csv(env, "session_meta.csv", row.names = FALSE)

cat("Done. Wrote fixtures to", getwd(), "| PMA",
    as.character(packageVersion("PMA")), "\n")