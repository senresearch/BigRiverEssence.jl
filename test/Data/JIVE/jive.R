# generate_jive_reference.R
# Generates r.jive reference fixtures for the Julia JIVE test suite.
# RUN AS A FILE (so output lands beside it):
#   cd Test/Data/JIVE && Rscript jive.R     — or — setwd(".../Test/Data/JIVE"); source("jive.R")
#
# Requires: install.packages("r.jive")

suppressMessages(library(r.jive))
set.seed(12345678)

#  simulated data with KNOWN structure 
n  <- 80
rT <- 2; r1T <- 3; r2T <- 3
p1 <- 60; p2 <- 50
S  <- matrix(rnorm(rT * n),  rT,  n)
U1 <- matrix(rnorm(p1 * rT), p1, rT); U2 <- matrix(rnorm(p2 * rT), p2, rT)
S1 <- matrix(rnorm(r1T * n), r1T, n); W1 <- matrix(rnorm(p1 * r1T), p1, r1T)
S2 <- matrix(rnorm(r2T * n), r2T, n); W2 <- matrix(rnorm(p2 * r2T), p2, r2T)
X1 <- U1 %*% S + W1 %*% S1 + 0.3 * matrix(rnorm(p1 * n), p1, n)
X2 <- U2 %*% S + W2 %*% S2 + 0.3 * matrix(rnorm(p2 * n), p2, n)

#  GIVEN-ranks fit (deterministic; the bit-identical reference) --
fit <- jive(list(X1, X2), rankJ = rT, rankA = c(r1T, r2T), method = "given",
            scale = TRUE, center = TRUE, est = TRUE, orthIndiv = TRUE,
            showProgress = FALSE)

# write the RAW inputs 
write.csv(X1, "X1.csv", row.names = FALSE)
write.csv(X2, "X2.csv", row.names = FALSE)
# ... the scaled data r.jive actually decomposed (to verify preprocessing match) ...
write.csv(fit$data[[1]],       "D1.csv", row.names = FALSE)
write.csv(fit$data[[2]],       "D2.csv", row.names = FALSE)
# ... and the decomposition.
write.csv(fit$joint[[1]],      "J1.csv", row.names = FALSE)
write.csv(fit$joint[[2]],      "J2.csv", row.names = FALSE)
write.csv(fit$individual[[1]], "A1.csv", row.names = FALSE)
write.csv(fit$individual[[2]], "A2.csv", row.names = FALSE)

# --- AUTO-RANK (permutation) reference on STRUCTURED data ---
# Strong, well-separated structure: joint rank 2, individual rank 3 each.
set.seed(99)
ns <- 80; ps1 <- 60; ps2 <- 50
Sj  <- matrix(rnorm(2 * ns), 2, ns)
i1  <- matrix(rnorm(ps1 * 3), ps1, 3) %*% matrix(rnorm(3 * ns), 3, ns)
i2  <- matrix(rnorm(ps2 * 3), ps2, 3) %*% matrix(rnorm(3 * ns), 3, ns)
X1s <- matrix(rnorm(ps1 * 2), ps1, 2) %*% Sj + 0.5 * i1 + 0.1 * matrix(rnorm(ps1 * ns), ps1, ns)
X2s <- matrix(rnorm(ps2 * 2), ps2, 2) %*% Sj + 0.5 * i2 + 0.1 * matrix(rnorm(ps2 * ns), ps2, ns)

set.seed(99)
fitp <- jive(list(X1s, X2s), method = "perm", scale = TRUE, center = TRUE,
             orthIndiv = TRUE, showProgress = FALSE)

# store r.jive's structured INPUTS (Julia reads THESE, so both decompose the same X),
# its estimated ranks, and its joint matrices (for a subspace comparison).
write.csv(X1s, "X1s.csv", row.names = FALSE)
write.csv(X2s, "X2s.csv", row.names = FALSE)
write.csv(fitp$joint[[1]], "J1p.csv", row.names = FALSE)
write.csv(fitp$joint[[2]], "J2p.csv", row.names = FALSE)
write.csv(data.frame(rankJ = fitp$rankJ,
                     rankA1 = fitp$rankA[1], rankA2 = fitp$rankA[2]),
          "ranks_perm.csv", row.names = FALSE)
cat("  perm fit ranks: joint", fitp$rankJ, "indiv", fitp$rankA, "\n")

write.csv(data.frame(n = n, rT = rT, r1T = r1T, r2T = r2T, p1 = p1, p2 = p2),
          "meta.csv", row.names = FALSE)

# provenance
writeLines(capture.output(sessionInfo()), "sessionInfo.txt")
env <- data.frame(
  field = c("R_version", "platform", "r.jive_version", "generated"),
  value = c(R.version$version.string, sessionInfo()$platform,
            as.character(packageVersion("r.jive")), as.character(Sys.time())))
write.csv(env, "session_meta.csv", row.names = FALSE)

cat("Done. Wrote fixtures to", getwd(), "| r.jive",
    as.character(packageVersion("r.jive")), "\n")