# generate_pmd_reference.R
# Generates the PMD reference fixtures used by the Julia test suite.
# RUN THIS AS A FILE so it can locate itself — do NOT paste it line-by-line.
#   From a shell:   Rscript pmd.R
#   From R:         source("Test/Data/PMD/pmd.R")
#
# Requires: install.packages("PMA")





suppressMessages(library(PMA))
set.seed(1234)

#  input data 
n <- 50
p <- 40
X <- matrix(rnorm(n * p), nrow = n, ncol = p)
Xc <- X - mean(X)                 # PMD centers by the GRAND mean

#  PMD
K  <- 3
su <- 0.5 * sqrt(n)
sv <- 0.5 * sqrt(p)
out <- PMD(Xc, type = "standard", sumabsu = su, sumabsv = sv, K = K,
           center = FALSE, trace = FALSE)

#  write fixtures 
write.csv(Xc,    file = "X.csv",     row.names = FALSE)
write.csv(out$u, file = "u_pmd.csv", row.names = FALSE)
write.csv(out$v, file = "v_pmd.csv", row.names = FALSE)
write.csv(out$d, file = "d_pmd.csv", row.names = FALSE)
write.csv(data.frame(n = n, p = p, K = K, sumabsu = su, sumabsv = sv),
          file = "meta.csv", row.names = FALSE)

#  provenance 
writeLines(capture.output(sessionInfo()), "sessionInfo.txt")
env <- data.frame(
  field = c("R_version", "platform", "PMA_version", "generated"),
  value = c(R.version$version.string,
            sessionInfo()$platform,
            as.character(packageVersion("PMA")),
            as.character(Sys.time()))
)
write.csv(env, "session_meta.csv", row.names = FALSE)

cat("Done. Wrote 7 files to", getwd(), "| PMA",
    as.character(packageVersion("PMA")), "\n")