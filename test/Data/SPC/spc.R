# generate_spc_reference.R
# Generates the SPC (Witten sparse PCA) reference fixtures for the Julia suite.
# RUN AS A FILE so it can locate itself — do NOT paste line-by-line.
#   From a shell:   Rscript spc.R
#   From R:         source("Test/Data/SPC/spc.R")
#
# Requires: install.packages("PMA")




suppressMessages(library(PMA))
set.seed(1234)

# input data
n <- 50
p <- 40
X <- matrix(rnorm(n * p), nrow = n, ncol = p)

# SPC column-centers to match Julia's spc (per-column means, NOT grand mean).
# We pre-center here and call SPC(center=FALSE); the Julia side runs spc on the
# RAW X and lets spc do the identical column-centering internally.
Xc <- scale(X, center = TRUE, scale = FALSE)

# SPC: L1 penalty on v only (u unpenalized). c == sumabsv, in [1, sqrt(p)].
K  <- 3
sv <- 0.6 * sqrt(p)

out_f <- SPC(Xc, sumabsv = sv, K = K, orth = FALSE, center = FALSE, trace = FALSE)
out_t <- SPC(Xc, sumabsv = sv, K = K, orth = TRUE,  center = FALSE, trace = FALSE)

# write fixtures — RAW X (Julia centers it), plus both orth variants
write.csv(X,        file = "X.csv",          row.names = FALSE)
write.csv(out_f$v,  file = "v_spc.csv",      row.names = FALSE)
write.csv(out_f$u,  file = "u_spc.csv",      row.names = FALSE)
write.csv(out_f$d,  file = "d_spc.csv",      row.names = FALSE)
write.csv(out_t$v,  file = "v_spc_orth.csv", row.names = FALSE)
write.csv(out_t$u,  file = "u_spc_orth.csv", row.names = FALSE)
write.csv(out_t$d,  file = "d_spc_orth.csv", row.names = FALSE)
write.csv(data.frame(n = n, p = p, K = K, sumabsv = sv),
          file = "meta.csv", row.names = FALSE)

# provenance
writeLines(capture.output(sessionInfo()), "sessionInfo.txt")
env <- data.frame(
  field = c("R_version", "platform", "PMA_version", "generated"),
  value = c(R.version$version.string,
            sessionInfo()$platform,
            as.character(packageVersion("PMA")),
            as.character(Sys.time()))
)
write.csv(env, "session_meta.csv", row.names = FALSE)

cat("Done. Wrote 9 files to", getwd(), "| PMA",
    as.character(packageVersion("PMA")), "\n")