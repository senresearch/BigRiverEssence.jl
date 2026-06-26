# generate_splsda_reference.R
# mixOmics sPLS-DA reference fixtures for the Julia test suite.
# RUN AS A FILE:  cd Test/Data/SPLSDA && Rscript splsda.R
# Requires: BiocManager::install("mixOmics")

suppressMessages(library(mixOmics))
set.seed(12345678)
data(srbct)

# subset: 60 samples, 200 genes, ncomp=2, keepX=15 each 
X <- srbct$gene[1:60, 1:200]
Y <- srbct$class[1:60]
ncomp <- 2; keepX <- c(15, 15)

res <- splsda(X, Y, ncomp = ncomp, keepX = keepX)

write.csv(X,                 "X.csv",   row.names = FALSE)
write.csv(data.frame(class = as.character(Y)), "Y.csv", row.names = FALSE)
write.csv(res$loadings$X,    "lx.csv",  row.names = FALSE)
write.csv(res$loadings$Y,    "ly.csv",  row.names = FALSE)
write.csv(res$variates$X,    "vx.csv",  row.names = FALSE)
write.csv(res$variates$Y,    "vy.csv",  row.names = FALSE)
write.csv(data.frame(level = levels(srbct$class)), "levels.csv", row.names = FALSE)
write.csv(data.frame(ncomp = ncomp, keepX1 = keepX[1], keepX2 = keepX[2]),
          "meta.csv", row.names = FALSE)

# provenance
writeLines(capture.output(sessionInfo()), "sessionInfo.txt")
env <- data.frame(
  field = c("R_version", "platform", "mixOmics_version", "generated"),
  value = c(R.version$version.string, sessionInfo()$platform,
            as.character(packageVersion("mixOmics")), as.character(Sys.time())))
write.csv(env, "session_meta.csv", row.names = FALSE)

cat("Done. Wrote fixtures to", getwd(), "| mixOmics",
    as.character(packageVersion("mixOmics")), "\n")