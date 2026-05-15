#!/usr/bin/env Rscript

# Install zellkonverter for converting between AnnData and SingleCellExperiment
# This package enables interoperability between Python (scanpy) and R (Bioconductor)

cat("========================================\n")
cat("Installing zellkonverter...\n")
cat("========================================\n\n")

# Ensure BiocManager is installed
if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
}

# Install zellkonverter and dependencies
BiocManager::install("zellkonverter", update = FALSE, ask = FALSE)

cat("\n========================================\n")
cat("Validating installation...\n")
cat("========================================\n\n")

# Validate installation
if (requireNamespace("zellkonverter", quietly = TRUE)) {
    library(zellkonverter)
    cat(sprintf("✓ zellkonverter: %s\n", packageVersion("zellkonverter")))
    
    # Check key functions
    cat("\nKey functions available:\n")
    cat("  - readH5AD(): Read AnnData .h5ad files into R\n")
    cat("  - writeH5AD(): Write SingleCellExperiment to .h5ad format\n")
    cat("  - AnnData2SCE(): Convert AnnData to SingleCellExperiment\n")
    cat("  - SCE2AnnData(): Convert SingleCellExperiment to AnnData\n")
    
    cat("\n========================================\n")
    cat("✓ Installation complete!\n")
    cat("========================================\n\n")
    
    cat("Usage in R:\n")
    cat("  library(zellkonverter)\n")
    cat("  sce <- readH5AD('data.h5ad')\n")
    cat("  writeH5AD(sce, 'output.h5ad')\n")
} else {
    cat("✗ Installation failed!\n")
    quit(status = 1)
}
