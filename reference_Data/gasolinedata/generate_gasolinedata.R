# generate_gasolinedata.R
# One-time: extract the gasoline NIR dataset from the pls package and save as CSVs.
# Run once locally where R and pls are available; commit the resulting CSVs.

suppressMessages(library(pls))
data(gasoline)

octane <- as.numeric(gasoline$octane)   # 60 responses
NIR    <- unclass(gasoline$NIR)         # 60 × 401 spectra matrix

outdir <- "reference_Data/gasolinedata"


write.table(NIR, file = file.path(outdir, "NIR.csv"),
            sep = ",", row.names = FALSE, col.names = FALSE)
write.table(octane, file = file.path(outdir, "octane.csv"),
            sep = ",", row.names = FALSE, col.names = FALSE)

cat("Saved gasoline CSVs to", outdir, "\n")
cat("NIR:   ", nrow(NIR), "x", ncol(NIR), "\n")
cat("octane:", length(octane), "\n")