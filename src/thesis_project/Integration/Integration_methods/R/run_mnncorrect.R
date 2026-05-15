#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(zellkonverter)
  library(SingleCellExperiment)
  library(SummarizedExperiment)
  library(batchelor)
  library(BiocNeighbors)
  library(BiocParallel)
  library(BiocSingular)
})

log_msg <- function(...) {
  message(...)
  flush.console()
}

parse_bool <- function(x) {
  if (is.logical(x)) return(x)
  x <- toupper(as.character(x))
  if (x %in% c("TRUE", "T", "1", "YES", "Y")) return(TRUE)
  if (x %in% c("FALSE", "F", "0", "NO", "N")) return(FALSE)
  stop("Could not parse logical value: ", x)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 15) {
  stop(
    paste0(
      "Expected 15+ arguments: input_h5ad batch_key out_prefix k subset_genes_csv n_pcs seed sigma cos_norm_in cos_norm_out svd_dim var_adj auto_merge nn_method bp_workers [restrict_n_per_batch]. ",
      "Received ", length(args), "."
    )
  )
}

input_h5ad         <- args[[1]]
batch_key          <- args[[2]]
out_prefix         <- args[[3]]
k                  <- as.integer(args[[4]])
subset_genes_csv   <- args[[5]]
n_pcs              <- as.integer(args[[6]])
seed               <- as.integer(args[[7]])
sigma              <- as.numeric(args[[8]])
cos_norm_in        <- parse_bool(args[[9]])
cos_norm_out       <- parse_bool(args[[10]])
svd_dim            <- as.integer(args[[11]])
var_adj            <- parse_bool(args[[12]])
auto_merge         <- parse_bool(args[[13]])
nn_method          <- tolower(args[[14]])
bp_workers         <- as.integer(args[[15]])
restrict_n_per_batch <- if (length(args) >= 16 && !identical(args[[16]], "NA")) as.integer(args[[16]]) else NA_integer_

if (!file.exists(input_h5ad)) stop("Input H5AD not found: ", input_h5ad)
if (is.na(k) || k <= 0L) stop("k must be a positive integer.")
if (is.na(n_pcs) || n_pcs <= 0L) stop("n_pcs must be a positive integer.")
if (is.na(seed)) seed <- 0L
if (is.na(sigma) || sigma <= 0) stop("sigma must be positive.")
if (is.na(svd_dim) || svd_dim < 0L) stop("svd_dim must be non-negative.")
if (!(nn_method %in% c("kmknn", "hnsw", "annoy"))) stop("nn_method must be one of 'kmknn', 'hnsw', 'annoy'.")
if (is.na(bp_workers) || bp_workers <= 0L) stop("bp_workers must be positive.")
if (!is.na(restrict_n_per_batch) && restrict_n_per_batch <= 0L) stop("restrict_n_per_batch must be positive when provided.")

set.seed(seed)
start_time <- Sys.time()
log_msg("=== MNN START ===")
log_msg("Start time: ", start_time)
log_msg("Seed: ", seed)

# readH5AD defaults to use_hdf5=FALSE, i.e., assays are realized in memory.
log_msg("Reading H5AD: ", input_h5ad)
sce <- zellkonverter::readH5AD(input_h5ad)
log_msg("Cells: ", ncol(sce), " | Genes: ", nrow(sce))

if (!(batch_key %in% colnames(SummarizedExperiment::colData(sce)))) {
  stop("batch_key '", batch_key, "' not found in colData(sce).")
}

available_assays <- SummarizedExperiment::assayNames(sce)
if ("logcounts" %in% available_assays) {
  expr_assay <- "logcounts"
} else if ("X" %in% available_assays) {
  expr_assay <- "X"
} else if (length(available_assays) >= 1L) {
  expr_assay <- available_assays[[1]]
} else {
  stop("No assays available in the loaded H5AD object.")
}
log_msg("Using assay: ", expr_assay)

batches <- factor(SummarizedExperiment::colData(sce)[[batch_key]])
if (any(is.na(batches))) stop("batch_key column contains NA values; please sanitize batch labels before running mnnCorrect.")
if (length(levels(batches)) < 2L) stop("mnnCorrect requires at least two batches.")

batch_sizes <- table(batches)
log_msg("Number of batches: ", length(levels(batches)))
log_msg("Batch sizes: ", paste(names(batch_sizes), batch_sizes, sep = "=", collapse = ", "))
if (any(batch_sizes < 2L)) stop("At least one batch has fewer than 2 cells; mnnCorrect cannot run.")
if (any(batch_sizes <= k)) {
  log_msg("Warning: one or more batches have <= k cells (k=", k, "). mnnCorrect may be unstable for those batches.")
}

ads <- lapply(levels(batches), function(b) sce[, batches == b, drop = FALSE])
log_msg("Split complete.")

subset_row <- NULL
if (!identical(subset_genes_csv, "NA")) {
  if (!file.exists(subset_genes_csv)) stop("subset_genes_csv not found: ", subset_genes_csv)
  subset_row <- unique(readLines(subset_genes_csv, warn = FALSE))
  subset_row <- subset_row[nzchar(subset_row)]
  if (!length(subset_row)) stop("subset_genes_csv is empty after removing blank lines.")

  common_genes <- Reduce(intersect, lapply(ads, rownames))
  subset_row <- intersect(subset_row, common_genes)
  if (!length(subset_row)) stop("No overlap between subset_genes_csv and the genes present in all batches.")
  log_msg("Using subset.row with ", length(subset_row), " genes.")
} else {
  log_msg("No subset.row supplied; using all genes in the input object.")
}

restrict_list <- NULL
if (!is.na(restrict_n_per_batch)) {
  restrict_list <- lapply(ads, function(x) {
    nc <- ncol(x)
    if (nc <= restrict_n_per_batch) {
      seq_len(nc)
    } else {
      sample.int(nc, restrict_n_per_batch, replace = FALSE)
    }
  })
  log_msg("Using restrict= with up to ", restrict_n_per_batch, " cells per batch for MNN matching.")
}

nn_param <- switch(
  nn_method,
  kmknn = BiocNeighbors::KmknnParam(),
  hnsw  = BiocNeighbors::HnswParam(),
  annoy = BiocNeighbors::AnnoyParam()
)
log_msg("Neighbor search method: ", class(nn_param)[1])

bp_param <- if (.Platform$OS.type == "unix" && bp_workers > 1L) {
  BiocParallel::MulticoreParam(workers = bp_workers, progressbar = TRUE)
} else {
  BiocParallel::SerialParam(progressbar = TRUE)
}
log_msg("Parallel backend: ", class(bp_param)[1], " | workers=", bp_workers)

log_msg("Running mnnCorrect...")
mnn_start <- Sys.time()

mnn_res <- do.call(
  batchelor::mnnCorrect,
  c(
    ads,
    list(
      assay.type = expr_assay,
      k = k,
      sigma = sigma,
      cos.norm.in = cos_norm_in,
      cos.norm.out = cos_norm_out,
      svd.dim = svd_dim,
      var.adj = var_adj,
      subset.row = subset_row,
      correct.all = FALSE,
      merge.order = NULL,
      auto.merge = auto_merge,
      restrict = restrict_list,
      BNPARAM = nn_param,
      BPPARAM = bp_param
    )
  )
)

mnn_end <- Sys.time()
log_msg("mnnCorrect runtime: ", difftime(mnn_end, mnn_start, units = "mins"), " minutes")

assay_names_out <- SummarizedExperiment::assayNames(mnn_res)
if (!("corrected" %in% assay_names_out)) {
  stop("Expected a 'corrected' assay in the mnnCorrect output, found: ", paste(assay_names_out, collapse = ", "))
}
corrected <- SummarizedExperiment::assay(mnn_res, "corrected")
log_msg("Corrected assay dim: ", paste(dim(corrected), collapse = " x "))

if (nrow(corrected) < 2L || ncol(corrected) < 2L) {
  stop("Corrected assay is too small for PCA: ", paste(dim(corrected), collapse = " x "))
}

max_rank <- min(as.integer(n_pcs), nrow(corrected), ncol(corrected) - 1L)
if (max_rank < 2L) {
  stop("Requested n_pcs is not feasible after correction. Requested=", n_pcs, ", corrected dim=", paste(dim(corrected), collapse = " x "))
}
if (max_rank < n_pcs) {
  log_msg("Reducing n_pcs from ", n_pcs, " to feasible rank ", max_rank, ".")
}

log_msg("Running approximate PCA on corrected assay...")
pca_start <- Sys.time()
pca <- BiocSingular::runPCA(
  t(corrected),
  rank = max_rank,
  center = TRUE,
  scale = FALSE,
  get.rotation = FALSE,
  get.pcs = TRUE,
  BSPARAM = BiocSingular::IrlbaParam(),
  BPPARAM = bp_param
)
pcs <- pca$x
if (nrow(pcs) != ncol(corrected) && ncol(pcs) == ncol(corrected)) {
  pcs <- t(pcs)
}
if (nrow(pcs) != ncol(corrected)) {
  stop(
    "Unexpected PCA score matrix shape ", paste(dim(pcs), collapse = " x "),
    "; expected one row per cell (", ncol(corrected), ")."
  )
}
if (is.null(rownames(pcs))) rownames(pcs) <- colnames(mnn_res)
pca_end <- Sys.time()

log_msg("PCA runtime: ", difftime(pca_end, pca_start, units = "mins"), " minutes")
log_msg("PCA embedding dim: ", paste(dim(pcs), collapse = " x "))

out_csv <- paste0(out_prefix, "_mnn_pca.csv")
write.csv(pcs, out_csv)
log_msg("Wrote PCA CSV: ", out_csv)

end_time <- Sys.time()
log_msg("Total runtime: ", difftime(end_time, start_time, units = "mins"), " minutes")
log_msg("=== MNN DONE ===")
