#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(zellkonverter)
  library(SingleCellExperiment)
  library(batchelor)
  library(BiocNeighbors)
  library(BiocParallel)
  library(BiocSingular)
})

args <- commandArgs(trailingOnly=TRUE)

# ------------------------------------------------------------------
# NEW calling convention (Python appends seed as the FINAL argument):
#   run_fastmnn.R <input_h5ad> <batch_key> <out_prefix> <d> <k>
#                 <cos_norm TRUE/FALSE> <ndist> <assay_type X|logcounts>
#                 [subset_genes_csv] <seed>
#
# So total args:
#   - without subset_genes_csv: 9  (8 fixed + seed)
#   - with subset_genes_csv   : 10 (8 fixed + subset + seed)
# ------------------------------------------------------------------
if (length(args) < 9) {
  cat("Usage:\n")
  cat("  run_fastmnn.R <input_h5ad> <batch_key> <out_prefix> <d> <k> <cos_norm TRUE/FALSE> <ndist> <assay_type X|logcounts> [subset_genes_csv] <seed>\n")
  quit(status = 2)
}

# seed is ALWAYS last
seed <- suppressWarnings(as.integer(args[[length(args)]]))
if (is.na(seed)) {
  stop("Last argument must be an integer seed. Received: ", args[[length(args)]])
}
set.seed(seed)

# Parse fixed args from the front
input_h5ad <- args[[1]]
batch_key  <- args[[2]]
out_prefix <- args[[3]]
d          <- as.integer(args[[4]])
k          <- as.integer(args[[5]])
cos_norm   <- as.logical(args[[6]])
ndist      <- as.numeric(args[[7]])
assay_type <- args[[8]]

# Optional subset CSV is present only if we have 10 args (8 fixed + subset + seed)
subset_csv <- NA
if (length(args) == 10) {
  subset_csv <- args[[9]]
}

cat("fastMNN run_fastmnn.R\n")
cat("  input_h5ad:", input_h5ad, "\n")
cat("  batch_key :", batch_key, "\n")
cat("  out_prefix:", out_prefix, "\n")
cat("  d/k      :", d, "/", k, "\n")
cat("  cos_norm :", cos_norm, "\n")
cat("  ndist    :", ndist, "\n")
cat("  assay    :", assay_type, "\n")
cat("  subset   :", ifelse(is.na(subset_csv), "(none)", subset_csv), "\n")
cat("  seed     :", seed, "\n\n")

# ------------------------------------------------------------------
# Read input
# ------------------------------------------------------------------
cat("Reading h5ad:", input_h5ad, "\n")
sce <- zellkonverter::readH5AD(input_h5ad)

# ---- validate batch_key in colData ----
cd <- colData(sce)
if (!(batch_key %in% colnames(cd))) {
  stop(paste0(
    "batch_key '", batch_key, "' not found in colData(sce). Available: ",
    paste(colnames(cd), collapse = ", ")
  ))
}

# Ensure batch is factor (as expected)
batch_vec <- factor(colData(sce)[[batch_key]])
if (nlevels(batch_vec) < 2) {
  stop("batch_key column has <2 unique batches. fastMNN requires multiple batches.")
}

# ---- choose assay ----
available_assays <- assayNames(sce)
if (!(assay_type %in% available_assays)) {
  stop(paste0(
    "Requested assay_type='", assay_type, "' not present. Available assays: ",
    paste(available_assays, collapse = ", "),
    ".\nIf your log-normalized matrix is AnnData .X, use assay_type='X'."
  ))
}

# ------------------------------------------------------------------
# Optional gene subsetting
# ------------------------------------------------------------------
subset.row <- NULL
if (!is.na(subset_csv) && nzchar(subset_csv) && file.exists(subset_csv)) {
  g <- read.csv(subset_csv, header = FALSE, stringsAsFactors = FALSE)[[1]]
  subset.row <- intersect(rownames(sce), g)

  cat("Gene subset CSV:", subset_csv, "-> requested", length(g), "genes; using", length(subset.row), "overlap\n")

  if (length(subset.row) == 0) {
    stop("subset_genes_csv produced 0 overlapping genes with rownames(sce). Check that HVG names match var_names.")
  }
} else {
  cat("No gene subset provided; using all genes\n")
}

# ------------------------------------------------------------------
# Run fastMNN
# ------------------------------------------------------------------
cat("Running fastMNN with:\n")
cat("  assay_type =", assay_type, "\n")
cat("  d =", d, "k =", k, "cos.norm =", cos_norm, "ndist =", ndist, "\n")

# Reproducibility:
# - SerialParam avoids parallel RNG non-determinism
bp <- BiocParallel::SerialParam()

res <- batchelor::fastMNN(
  sce,
  batch = batch_vec,
  k = k,
  d = d,
  cos.norm = cos_norm,
  ndist = ndist,
  subset.row = subset.row,
  assay.type = assay_type,
  BSPARAM = BiocSingular::IrlbaParam(),
  BNPARAM = BiocNeighbors::KmknnParam(),
  BPPARAM = bp,
  deferred = TRUE
)

# ------------------------------------------------------------------
# Extract corrected coordinates
# ------------------------------------------------------------------
rdn <- reducedDimNames(res)
cat("reducedDimNames:", paste(rdn, collapse = ", "), "\n")

mat <- NULL
if ("corrected" %in% rdn) {
  mat <- reducedDim(res, "corrected")
} else if (length(rdn) >= 1) {
  mat <- reducedDim(res, rdn[[1]])
}

if (is.null(mat)) {
  stop("Could not extract corrected reduced dimensions from fastMNN result.")
}

# Ensure rownames are cell IDs and match input colnames(sce)
# zellkonverter typically preserves colnames -> rownames(mat), but enforce.
if (is.null(rownames(mat)) || any(rownames(mat) == "")) {
  rownames(mat) <- colnames(sce)
}

# Publication safety: if rownames still don't match colnames(sce), fix if possible
if (!identical(rownames(mat), colnames(sce))) {
  # Try to reorder by matching names
  if (all(colnames(sce) %in% rownames(mat))) {
    mat <- mat[colnames(sce), , drop = FALSE]
  } else {
    stop(
      "Corrected embedding rownames do not match input cell IDs (colnames(sce)). ",
      "Cannot safely align. Ensure input H5AD has stable cell IDs and zellkonverter preserves them."
    )
  }
}

# ------------------------------------------------------------------
# Write outputs
# ------------------------------------------------------------------
rds_out   <- paste0(out_prefix, "_fastmnn.rds")
csv_out   <- paste0(out_prefix, "_fastmnn_embedding.csv")
merge_out <- paste0(out_prefix, "_fastmnn_mergeinfo.rds")

saveRDS(res, file = rds_out)

# IMPORTANT: write.csv will write rownames as first column when row.names=TRUE
write.csv(mat, file = csv_out, quote = FALSE, row.names = TRUE)

mi <- metadata(res)$merge.info
if (!is.null(mi)) saveRDS(mi, file = merge_out)

cat("Saved:\n")
cat("  ", rds_out, "\n")
cat("  ", csv_out, "\n")
cat("  ", merge_out, "\n")
cat("Done.\n")

