#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(zellkonverter)
  library(SingleCellExperiment)
  library(SummarizedExperiment)
  library(batchelor)
  library(BiocNeighbors)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 10) {
  stop(
    paste(
      "Usage:",
      "Rscript run_fastmnn.R <input_h5ad> <batch_key> <out_prefix>",
      "<d> <k> <cos_norm> <ndist> <assay> <subset_csv_or_NA> <seed>"
    )
  )
}

input_h5ad <- args[[1]]
batch_key  <- args[[2]]
out_prefix <- args[[3]]
d          <- as.integer(args[[4]])
k          <- as.integer(args[[5]])
cos_norm   <- tolower(args[[6]]) %in% c("true", "1", "t", "yes", "y")
ndist      <- as.numeric(args[[7]])
assay_name <- args[[8]]
subset_csv <- args[[9]]
seed       <- as.integer(args[[10]])

cat("fastMNN run_fastmnn.R\n")
cat("  input_h5ad:", input_h5ad, "\n")
cat("  batch_key :", batch_key, "\n")
cat("  out_prefix:", out_prefix, "\n")
cat("  d/k      :", d, "/", k, "\n")
cat("  cos_norm :", cos_norm, "\n")
cat("  ndist    :", ndist, "\n")
cat("  assay    :", assay_name, "\n")
cat("  subset   :", subset_csv, "\n")
cat("  seed     :", seed, "\n\n")

set.seed(seed)

if (!file.exists(input_h5ad)) {
  stop("Input H5AD not found: ", input_h5ad)
}

cat("Reading h5ad:", input_h5ad, "\n")
sce <- zellkonverter::readH5AD(input_h5ad)

if (!(batch_key %in% colnames(colData(sce)))) {
  stop("batch_key not found in colData(sce): ", batch_key)
}

if (!(assay_name %in% assayNames(sce))) {
  stop(
    "Requested assay '", assay_name, "' not found. Available assays: ",
    paste(assayNames(sce), collapse = ", ")
  )
}

colData(sce)[[batch_key]] <- droplevels(as.factor(colData(sce)[[batch_key]]))
batch_vec <- colData(sce)[[batch_key]]
assay_to_use <- assay_name

subset_genes <- NULL
if (!is.na(subset_csv) && subset_csv != "NA" && nzchar(subset_csv)) {
  if (!file.exists(subset_csv)) {
    stop("Subset CSV not found: ", subset_csv)
  }

  raw_genes <- read.csv(subset_csv, header = FALSE, stringsAsFactors = FALSE)[[1]]
  raw_genes <- unique(raw_genes)
  overlap <- intersect(raw_genes, rownames(sce))

  cat(
    "Gene subset CSV:", subset_csv,
    "-> requested", length(raw_genes),
    "genes; using", length(overlap), "overlap\n"
  )

  if (length(overlap) == 0) {
    stop("No overlap between subset genes and rownames(sce)")
  }

  subset_genes <- overlap
} else {
  cat("No gene subset provided; using all genes\n")
}

x <- assay(sce, assay_to_use)

if (inherits(x, "sparseMatrix")) {
  if (length(x@x) > 0 && any(!is.finite(x@x))) {
    stop("Non-finite values (NA/Inf) detected in non-zero entries of assay '", assay_to_use, "'")
  }
  gene_vars <- Matrix::rowSums(x^2) / ncol(x) - (Matrix::rowSums(x) / ncol(x))^2
} else {
  if (any(!is.finite(x))) {
    stop("Non-finite values (NA/Inf) detected in assay '", assay_to_use, "'")
  }
  gene_vars <- apply(x, 1, var)
}

gene_vars[is.na(gene_vars)] <- 0

if (!is.null(subset_genes)) {
  keep <- subset_genes[gene_vars[subset_genes] > 0]
  cat("Subset genes with variance > 0:", length(keep), "\n")
} else {
  keep <- rownames(sce)[gene_vars > 0]
  cat("All genes with variance > 0:", length(keep), "\n")
}

if (length(keep) == 0) {
  stop("No genes with non-zero variance in assay '", assay_to_use, "'")
}

if (!is.null(subset_genes) && length(keep) < 10) {
  stop(
    "Too few variable genes remain after filtering subset: ", length(keep),
    ". Check assay choice and subset CSV."
  )
}

x_keep <- x[keep, , drop = FALSE]
if (inherits(x_keep, "sparseMatrix")) {
  cell_l2sq <- Matrix::colSums(x_keep^2)
} else {
  cell_l2sq <- colSums(x_keep^2)
}

if (any(is.na(cell_l2sq) | !is.finite(cell_l2sq))) {
  stop("Detected non-finite per-cell squared norms on selected genes")
}

zero_norm <- which(cell_l2sq <= 0)
if (length(zero_norm) > 0) {
  cat("WARNING:", length(zero_norm), "cells have zero norm on selected genes\n")
  cat("First few zero-norm cells:", paste(head(colnames(sce)[zero_norm], 10), collapse = ", "), "\n")
}

if (cos_norm && length(zero_norm) > 0) {
  cat("Disabling cos.norm because zero-norm cells were detected on selected genes\n")
  cos_norm <- FALSE
}

batch_levels <- levels(batch_vec)
if (length(batch_levels) < 2) {
  stop("fastMNN requires at least 2 batches; found: ", length(batch_levels))
}

sce_list <- lapply(batch_levels, function(b) {
  cur <- sce[, batch_vec == b, drop = FALSE]
  if (ncol(cur) == 0) {
    stop("Batch '", b, "' has zero cells after filtering")
  }
  cur
})
names(sce_list) <- batch_levels

cat("Batches:", paste(names(sce_list), collapse = ", "), "\n")
cat("Cells per batch:", paste(vapply(sce_list, ncol, integer(1)), collapse = ", "), "\n")

# More robust settings for large, imbalanced atlas batches
prop_k <- 0.01
k_use <- max(k, 50L)

cat("Running fastMNN with:\n")
cat("  assay_type =", assay_to_use, "\n")
cat("  d =", d, "k =", k_use, "prop.k =", prop_k,
    "cos.norm =", cos_norm, "ndist =", ndist, "\n")
cat("  auto.merge = TRUE\n")
cat("  BNPARAM = BiocNeighbors::VptreeParam()\n\n")

res <- do.call(
  batchelor::fastMNN,
  c(
    sce_list,
    list(
      subset.row = keep,
      assay.type = assay_to_use,
      d = d,
      k = k_use,
      prop.k = prop_k,
      cos.norm = cos_norm,
      ndist = ndist,
      auto.merge = TRUE,
      BNPARAM = BiocNeighbors::VptreeParam()
    )
  )
)

if (!("corrected" %in% reducedDimNames(res))) {
  stop("fastMNN result does not contain reducedDim 'corrected'")
}

emb <- reducedDim(res, "corrected")
if (is.null(rownames(emb))) {
  rownames(emb) <- colnames(res)
}

emb_df <- as.data.frame(emb)
rownames(emb_df) <- rownames(emb)

out_csv <- paste0(out_prefix, "_fastmnn.csv")
write.csv(emb_df, out_csv, quote = FALSE)

cat("Saved corrected embedding to:", out_csv, "\n")
cat("Done.\n")