#!/usr/bin/env Rscript
# Install scater package for MNN analysis
# Usage: Rscript install_scater.R

cat("Installing scater package...\n")

# Ensure BiocManager is available
if (!requireNamespace("BiocManager", quietly = TRUE)) {
    cat("Installing BiocManager first...\n")
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
}

# Install scater
cat("Installing scater from Bioconductor...\n")
BiocManager::install("scater", update = FALSE, ask = FALSE)

# Verify installation
if (requireNamespace("scater", quietly = TRUE)) {
    cat("\n✓ scater successfully installed!\n")
    cat("Version:", as.character(packageVersion("scater")), "\n")
} else {
    cat("\n✗ Failed to install scater\n")
    quit(status = 1)
}
