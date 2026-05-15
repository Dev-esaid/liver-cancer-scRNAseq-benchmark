#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(zellkonverter)
  library(Matrix)
  library(rliger)
  if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    library(RhpcBLASctl)
  }
})
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 11) {
  cat("Usage:\n")
  cat("  run_liger.R <input_h5ad> <batch_key> <out_prefix> <k_factors> <lambda> <n_iters> <subset_genes_txt|NA> <seed> <align_method> <n_cores> <save_rds>\n")
  quit(status = 1)
}

input_h5ad      <- args[[1]]
batch_key       <- args[[2]]
out_prefix      <- args[[3]]
k_factors       <- as.integer(args[[4]])
lambda_reg      <- as.numeric(args[[5]])
n_iters         <- as.integer(args[[6]])
subset_genes    <- args[[7]]
seed            <- as.integer(args[[8]])
align_method    <- args[[9]]
n_cores         <- as.integer(args[[10]])
save_rds_flag   <- as.integer(args[[11]]) == 1L

set.seed(seed)


if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
  RhpcBLASctl::blas_set_num_threads(1)
  RhpcBLASctl::omp_set_num_threads(1)
  message("[LIGER R] Forced BLAS/OMP threads to 1 via RhpcBLASctl")
} else {
  message("[LIGER R] RhpcBLASctl not available; relying on environment variables only")
}
message("[LIGER R] Reading H5AD: ", input_h5ad)
sce <- zellkonverter::readH5AD(input_h5ad)

if (!(batch_key %in% colnames(SummarizedExperiment::colData(sce)))) {
  stop(
    "[LIGER R] batch_key '", batch_key, "' not found. Available: ",
    paste(colnames(SummarizedExperiment::colData(sce)), collapse = ", ")
  )
}

batches <- as.factor(SummarizedExperiment::colData(sce)[[batch_key]])
if (length(batches) != ncol(sce)) {
  stop("[LIGER R] batch vector length does not match number of cells.")
}
if (any(is.na(batches))) {
  stop("[LIGER R] batch assignments contain NA values.")
}

assays_avail <- SummarizedExperiment::assayNames(sce)
message("[LIGER R] assays: ", paste(assays_avail, collapse = ", "))

expr_assay <- if ("counts" %in% assays_avail) "counts" else "X"
if (!(expr_assay %in% assays_avail)) {
  stop("[LIGER R] Neither 'counts' nor 'X' assay found in H5AD.")
}
message("[LIGER R] Using assay: ", expr_assay)

expr_mat <- SummarizedExperiment::assay(sce, expr_assay)
if (!inherits(expr_mat, "dgCMatrix")) {
  expr_mat <- as(expr_mat, "dgCMatrix")
}

if (is.null(rownames(expr_mat)) || anyDuplicated(rownames(expr_mat))) {
  stop("[LIGER R] Gene names are missing or duplicated.")
}
if (is.null(colnames(expr_mat)) || anyDuplicated(colnames(expr_mat))) {
  stop("[LIGER R] Cell names are missing or duplicated.")
}

if (length(expr_mat@x) > 0) {
  min_val <- min(expr_mat@x)
  if (min_val < 0) {
    stop("[LIGER R] Negative values found in raw counts (min=", signif(min_val, 6), ").")
  }
} else {
  min_val <- 0
}
message("[LIGER R] Raw counts OK (min=", signif(min_val, 6), ")")

cell_sums <- Matrix::colSums(expr_mat)
if (any(cell_sums <= 0)) {
  bad_n <- sum(cell_sums <= 0)
  bad_cells <- head(colnames(expr_mat)[cell_sums <= 0], 10)
  stop(
    "[LIGER R] Found ", bad_n, " cells with zero total counts after export. ",
    "Examples: ", paste(bad_cells, collapse = ", ")
  )
}

message(
  "[LIGER R] Full matrix: genes=", nrow(expr_mat),
  " cells=", ncol(expr_mat),
  " sparse=", inherits(expr_mat, "dgCMatrix")
)

features_use <- NULL
if (!identical(subset_genes, "NA") && nzchar(subset_genes)) {
  genes_input <- readLines(subset_genes, warn = FALSE, encoding = "UTF-8")
  genes_input <- unique(trimws(genes_input))
  genes_input <- genes_input[nzchar(genes_input)]
  genes_present <- intersect(rownames(expr_mat), genes_input)

  message(
    "[LIGER R] External HVGs: input=", length(genes_input),
    " present=", length(genes_present)
  )

  if (length(genes_present) == 0) {
    stop("[LIGER R] None of the supplied HVGs were found in the matrix.")
  }
  features_use <- genes_present
} else {
  stop("[LIGER R] This wrapper expects externally supplied HVGs; got none.")
}

message("[LIGER R] Creating liger object from full raw matrix ...")
liger_obj <- rliger::as.liger(expr_mat, datasetVar = batches)

message("[LIGER R] Setting external HVGs as varFeatures (n=", length(features_use), ")")
rliger::varFeatures(liger_obj) <- features_use

message("[LIGER R] normalize() ...")
liger_obj <- rliger::normalize(liger_obj)

message("[LIGER R] scaleNotCenter() on varFeatures only ...")
liger_obj <- rliger::scaleNotCenter(liger_obj)

message(
  "[LIGER R] runIntegration(method='iNMF', k=", k_factors,
  ", lambda=", lambda_reg,
  ", nIteration=", n_iters,
  ", nCores=", n_cores, ") ..."
)
liger_obj <- rliger::runIntegration(
  liger_obj,
  method = "iNMF",
  k = k_factors,
  lambda = lambda_reg,
  nIteration = n_iters,
  seed = seed,
  nRandomStarts = 0,
  nCores = n_cores
)

message("[LIGER R] alignFactors(method='", align_method, "') ...")
liger_obj <- rliger::alignFactors(liger_obj, method = align_method)

message("[LIGER R] Exporting aligned factors ...")
hnorm <- rliger::getMatrix(liger_obj, "H.norm")
if (is.null(hnorm)) {
  stop("[LIGER R] H.norm not available after alignment.")
}

h_out <- paste0(out_prefix, "_liger_factors.csv")
# Verify rownames match original cell names before export
original_cells <- colnames(expr_mat)
hnorm_cells <- rownames(hnorm)

if (!all(hnorm_cells %in% original_cells)) {
  # rliger may have prepended batch names — try to strip them
  # by matching the suffix after the first underscore
  message("[LIGER R] WARNING: H.norm rownames do not directly match input cell IDs.")
  message("[LIGER R] Attempting to restore original cell IDs from colnames(expr_mat)...")
  
  if (length(hnorm_cells) == length(original_cells)) {
    rownames(hnorm) <- original_cells
    message("[LIGER R] Cell IDs restored by position (same count, assumed same order).")
  } else {
    stop(
      "[LIGER R] H.norm has ", nrow(hnorm), " rows but input has ", 
      length(original_cells), " cells. Cannot align."
    )
  }
}

write.csv(hnorm, file = h_out, quote = FALSE)
message(
  "[LIGER R] Saved: ", h_out,
  " (", nrow(hnorm), " cells x ", ncol(hnorm), " factors)"
)

if (isTRUE(save_rds_flag)) {
  rds_out <- paste0(out_prefix, "_liger_integrated.rds")
  saveRDS(liger_obj, file = rds_out)
  message("[LIGER R] Saved: ", rds_out)
}

message("[LIGER R] DONE")
