# #!/usr/bin/env Rscript
# library(future)
# plan("sequential")
# options(future.globals.maxSize = 50 * 1024^3)

# suppressPackageStartupMessages({
#   library(zellkonverter)
#   library(SingleCellExperiment)
#   library(Matrix)
#   library(Seurat)
# })

# args <- commandArgs(trailingOnly = TRUE)

# if (length(args) < 4) {
#   cat("Usage:\n")
#   cat("  run_seurat_integration.R <input_h5ad> <batch_key> <out_prefix> <mode:cca|rpca> [k_anchor] [dims] [n_pcs] [subset_genes_csv] [seed]\n")
#   quit(status = 1)
# }

# input_h5ad <- args[[1]]
# batch_key  <- args[[2]]
# out_prefix <- args[[3]]
# mode       <- tolower(args[[4]])

# k_anchor <- if (length(args) >= 5) as.integer(args[[5]]) else 5L
# dims     <- if (length(args) >= 6) as.integer(args[[6]]) else 30L
# n_pcs    <- if (length(args) >= 7) as.integer(args[[7]]) else 50L

# subset_genes_csv <- if (length(args) >= 8) args[[8]] else NA
# seed <- if (length(args) >= 9) as.integer(args[[9]]) else 0L

# if (!(mode %in% c("cca", "rpca"))) {
#   stop("mode must be 'cca' or 'rpca'")
# }

# set.seed(seed)

# message("Reading H5AD: ", input_h5ad)
# sce <- zellkonverter::readH5AD(input_h5ad)

# # ----------------------------
# # Validate batch key
# # ----------------------------
# if (!(batch_key %in% colnames(colData(sce)))) {
#   stop(paste0("batch_key '", batch_key, "' not found in colData(sce). Available: ",
#               paste(colnames(colData(sce)), collapse = ", ")))
# }

# batches <- factor(colData(sce)[[batch_key]])
# levels_batches <- levels(batches)
# message("Batches found: ", paste(levels_batches, collapse = ", "))

# # ----------------------------
# # Get counts matrix for Seurat
# # ----------------------------
# # Prefer X assay (log-normalized) for HVG-subset data to avoid double normalization
# assays_avail <- assayNames(sce)
# message("Assays in SCE: ", paste(assays_avail, collapse = ", "))

# counts_mat <- NULL
# counts_assay <- NULL
# is_normalized <- FALSE

# # For HVG-subset data, prefer X (already log-normalized)
# # Priority: X > logcounts > counts
# if ("X" %in% assays_avail) {
#   counts_assay <- "X"
#   counts_mat <- assay(sce, "X")
#   is_normalized <- TRUE  # X is typically log-normalized
#   message("Using 'X' assay (assumed log-normalized, will skip NormalizeData)")
# } else if ("logcounts" %in% assays_avail) {
#   counts_assay <- "logcounts"
#   counts_mat <- assay(sce, "logcounts")
#   is_normalized <- TRUE
#   message("Using 'logcounts' assay (already normalized, will skip NormalizeData)")
# } else if ("counts" %in% assays_avail) {
#   counts_assay <- "counts"
#   counts_mat <- assay(sce, "counts")
#   is_normalized <- FALSE
#   message("Using 'counts' assay (raw counts, will apply NormalizeData)")
# } else {
#   stop("No suitable assay found. Expected 'X', 'logcounts', or 'counts'.")
# }

# # Ensure sparse dgCMatrix where possible
# if (!inherits(counts_mat, "dgCMatrix")) {
#   counts_mat <- as(as.matrix(counts_mat), "dgCMatrix")
# }

# message("Using counts assay: ", counts_assay, " | dims = ", nrow(counts_mat), " genes x ", ncol(counts_mat), " cells")

# # ----------------------------
# # Build Seurat object + metadata
# # ----------------------------
# obj <- CreateSeuratObject(counts = counts_mat, assay = "RNA", project = "SeuratIntegration")

# meta <- as.data.frame(colData(sce))
# # ensure rownames align to cell names
# if (!is.null(colnames(sce))) {
#   rownames(meta) <- colnames(sce)
# }
# obj <- AddMetaData(obj, metadata = meta)

# # ----------------------------
# # Optional: restrict to gene list (HVG list from Python)
# # ----------------------------
# features_use <- NULL
# if (!is.na(subset_genes_csv) && nzchar(subset_genes_csv) && subset_genes_csv != "NA") {
#   gd <- read.csv(subset_genes_csv, header = FALSE, stringsAsFactors = FALSE)
#   genes_input <- unique(as.character(gd[[1]]))
#   genes_present <- intersect(genes_input, rownames(obj))
#   message("Subset genes requested: ", length(genes_input), " | present: ", length(genes_present))
#   if (length(genes_present) < 50) {
#     message("WARNING: very few subset genes present (<50). Integration may fail or be unstable.")
#   }
#   features_use <- genes_present
# }

# # ----------------------------
# # Split objects and preprocess
# # ----------------------------
# obj_list <- SplitObject(obj, split.by = batch_key)

# # Seurat integration best practice (LogNormalize workflow)
# for (i in seq_along(obj_list)) {
#   # Only normalize if data is not already normalized
#   if (!is_normalized) {
#     obj_list[[i]] <- NormalizeData(obj_list[[i]], normalization.method = "LogNormalize", verbose = FALSE)
#   } else {
#     message("Skipping NormalizeData (data already normalized)")
#     # For Seurat 5+, we need to create the 'data' layer manually when using pre-normalized data
#     # Copy the counts layer to data layer (since counts already contains normalized values)
#     obj_list[[i]][["RNA"]]$data <- obj_list[[i]][["RNA"]]$counts
#   }

#   if (is.null(features_use)) {
#     obj_list[[i]] <- FindVariableFeatures(obj_list[[i]], selection.method = "vst", nfeatures = 2000, verbose = FALSE)
#   } else {
#     # force variable features to the provided set (Seurat expects VariableFeatures to exist)
#     VariableFeatures(obj_list[[i]]) <- features_use
#   }

#   # Always run ScaleData to properly center and scale the data for PCA
#   obj_list[[i]] <- ScaleData(obj_list[[i]], features = VariableFeatures(obj_list[[i]]), verbose = FALSE)

#   # PCA per batch object
#   obj_list[[i]] <- RunPCA(obj_list[[i]], features = VariableFeatures(obj_list[[i]]), npcs = max(dims, n_pcs), verbose = FALSE)
# }

# # ----------------------------
# # Select integration features / anchors
# # ----------------------------
# features <- NULL
# if (is.null(features_use)) {
#   features <- SelectIntegrationFeatures(object.list = obj_list, nfeatures = 2000)
# } else {
#   features <- features_use
# }

# # For RPCA we also run:
# # - ScaleData + RunPCA already done above per object
# reduction_use <- if (mode == "rpca") "rpca" else "cca"

# message("Running FindIntegrationAnchors(mode=", mode, ", reduction=", reduction_use, ", dims=1:", dims, ", k.anchor=", k_anchor, ")")

# anchors <- FindIntegrationAnchors(
#   object.list = obj_list,
#   anchor.features = features,
#   reduction = reduction_use,
#   dims = 1:dims,
#   k.anchor = k_anchor,
#   verbose = FALSE
# )

# message("Running IntegrateData(dims=1:", dims, ")")
# obj_int <- IntegrateData(anchorset = anchors, dims = 1:dims, verbose = FALSE)

# DefaultAssay(obj_int) <- "integrated"

# # Standard: scale + PCA on integrated
# obj_int <- ScaleData(obj_int, verbose = FALSE)
# obj_int <- RunPCA(obj_int, npcs = max(dims, n_pcs), verbose = FALSE)

# # ----------------------------
# # Export embeddings
# # ----------------------------
# emb <- Embeddings(obj_int, reduction = "pca")
# if (!is.null(n_pcs) && n_pcs > 0 && ncol(emb) > n_pcs) {
#   emb <- emb[, 1:n_pcs, drop = FALSE]
# }

# pca_out <- paste0(out_prefix, "_seurat_pca.csv")
# write.csv(emb, file = pca_out, quote = FALSE)
# message("Saved PCA embeddings CSV: ", pca_out)

# # Variance explained (optional)
# stdev <- Stdev(obj_int[["pca"]])
# var <- stdev^2
# var_ratio <- var / sum(var)
# var_df <- data.frame(
#   PC = seq_along(var),
#   variance = var,
#   variance_ratio = var_ratio
# )
# if (!is.null(n_pcs) && n_pcs > 0 && nrow(var_df) > n_pcs) {
#   var_df <- var_df[1:n_pcs, , drop = FALSE]
# }
# pca_var_out <- paste0(out_prefix, "_seurat_pca_variance.csv")
# write.csv(var_df, file = pca_var_out, row.names = FALSE, quote = FALSE)
# message("Saved PCA variance CSV: ", pca_var_out)

# # Save full Seurat object for reproducibility
# rds_out <- paste0(out_prefix, "_seurat_integrated.rds")
# saveRDS(obj_int, file = rds_out)
# message("Saved Seurat RDS: ", rds_out)

# message("DONE")
#!/usr/bin/env Rscript
#!/usr/bin/env Rscript

#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(future)
  library(zellkonverter)
  library(SingleCellExperiment)
  library(Matrix)
  library(Seurat)
})

plan("sequential")
options(future.globals.maxSize = 50 * 1024^3)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  cat("Usage:\n")
  cat("  run_seurat_integration.R <input_h5ad> <batch_key> <out_prefix> <mode:cca|rpca> [k_anchor] [dims] [n_pcs] [subset_genes_csv] [seed]\n")
  quit(status = 1)
}

input_h5ad <- args[[1]]
batch_key  <- args[[2]]
out_prefix <- args[[3]]
mode       <- tolower(args[[4]])

k_anchor <- if (length(args) >= 5) as.integer(args[[5]]) else 5L
dims     <- if (length(args) >= 6) as.integer(args[[6]]) else 30L
n_pcs    <- if (length(args) >= 7) as.integer(args[[7]]) else 50L
subset_genes_csv <- if (length(args) >= 8) args[[8]] else NA
seed <- if (length(args) >= 9) as.integer(args[[9]]) else 0L

if (!(mode %in% c("cca", "rpca"))) stop("mode must be 'cca' or 'rpca'")
set.seed(seed)

message("Reading H5AD: ", input_h5ad)
sce <- zellkonverter::readH5AD(input_h5ad)

if (!(batch_key %in% colnames(colData(sce)))) {
  stop(paste0("batch_key '", batch_key, "' not found in colData(sce). Available: ",
              paste(colnames(colData(sce)), collapse = ", ")))
}

batches <- factor(colData(sce)[[batch_key]])
message("Batches found: ", paste(levels(batches), collapse = ", "))

assays_avail <- assayNames(sce)
message("Assays in SCE: ", paste(assays_avail, collapse = ", "))

mat <- NULL
assay_name <- NULL
is_normalized <- FALSE

if ("X" %in% assays_avail) {
  assay_name <- "X"
  mat <- assay(sce, "X")
  is_normalized <- TRUE
  message("Using 'X' assay (assumed log-normalized, will skip NormalizeData)")
} else if ("logcounts" %in% assays_avail) {
  assay_name <- "logcounts"
  mat <- assay(sce, "logcounts")
  is_normalized <- TRUE
  message("Using 'logcounts' assay (already normalized, will skip NormalizeData)")
} else if ("counts" %in% assays_avail) {
  assay_name <- "counts"
  mat <- assay(sce, "counts")
  is_normalized <- FALSE
  message("Using 'counts' assay (raw counts, will apply NormalizeData)")
} else {
  stop("No suitable assay found. Expected 'X', 'logcounts', or 'counts'.")
}

if (!inherits(mat, "dgCMatrix")) {
  mat <- as(as.matrix(mat), "dgCMatrix")
}
message("Using assay: ", assay_name, " | dims = ", nrow(mat), " genes x ", ncol(mat), " cells")

# ----------------------------
# Build full Seurat object
# ----------------------------
obj <- CreateSeuratObject(counts = mat, assay = "RNA", project = "SeuratIntegration")

meta <- as.data.frame(colData(sce))
if (!is.null(colnames(sce))) rownames(meta) <- colnames(sce)
obj <- AddMetaData(obj, metadata = meta)

# ----------------------------
# Optional: restrict to provided gene list (HVG list)
# ----------------------------
features_use <- NULL
if (!is.na(subset_genes_csv) && nzchar(subset_genes_csv) && subset_genes_csv != "NA") {
  gd <- read.csv(subset_genes_csv, header = FALSE, stringsAsFactors = FALSE)
  genes_input <- unique(as.character(gd[[1]]))
  genes_present <- intersect(genes_input, rownames(obj))
  message("Subset genes requested: ", length(genes_input), " | present: ", length(genes_present))
  features_use <- genes_present
}

# ----------------------------
# Ensure a valid 'data' layer for Seurat v5 when using pre-normalized matrices
# ----------------------------
if (is_normalized) {
  message("Skipping NormalizeData (data already normalized) [FULL OBJ]")
  if (inherits(obj[["RNA"]], "Assay5")) {
    x_counts <- tryCatch(GetAssayData(obj, assay = "RNA", layer = "counts"), error = function(e) NULL)
    if (is.null(x_counts)) stop("Assay5: could not retrieve layer='counts' after CreateSeuratObject().")
    obj <- SetAssayData(obj, assay = "RNA", layer = "data", new.data = x_counts)
  } else {
    if (ncol(obj[["RNA"]]@data) == 0) obj[["RNA"]]@data <- obj[["RNA"]]@counts
  }
} else {
  obj <- NormalizeData(obj, normalization.method = "LogNormalize", verbose = FALSE)
}

# ----------------------------
# Build a FULL PCA reduction (237,144 cells) to satisfy IntegrateEmbeddings
# ----------------------------
if (is.null(features_use)) {
  obj <- FindVariableFeatures(obj, selection.method = "vst", nfeatures = 2000, verbose = FALSE)
} else {
  VariableFeatures(obj) <- features_use
}
obj <- ScaleData(obj, features = VariableFeatures(obj), verbose = FALSE)
obj <- RunPCA(obj, features = VariableFeatures(obj), npcs = max(dims, n_pcs), verbose = FALSE)

# ----------------------------
# Split objects for anchors workflow
# ----------------------------
obj_list <- SplitObject(obj, split.by = batch_key)

# Per-batch preprocessing (needed for RPCA anchor workflow)
for (i in seq_along(obj_list)) {
  # data already prepared in full obj; still ensure data exists for each split
  if (inherits(obj_list[[i]][["RNA"]], "Assay5")) {
    x_counts <- tryCatch(GetAssayData(obj_list[[i]], assay = "RNA", layer = "counts"), error = function(e) NULL)
    if (!is.null(x_counts)) {
      obj_list[[i]] <- SetAssayData(obj_list[[i]], assay = "RNA", layer = "data", new.data = x_counts)
    }
  } else {
    if (ncol(obj_list[[i]][["RNA"]]@data) == 0) {
      obj_list[[i]][["RNA"]]@data <- obj_list[[i]][["RNA"]]@counts
    }
  }

  if (is.null(features_use)) {
    obj_list[[i]] <- FindVariableFeatures(obj_list[[i]], selection.method = "vst", nfeatures = 2000, verbose = FALSE)
  } else {
    VariableFeatures(obj_list[[i]]) <- features_use
  }

  obj_list[[i]] <- ScaleData(obj_list[[i]], features = VariableFeatures(obj_list[[i]]), verbose = FALSE)
  obj_list[[i]] <- RunPCA(obj_list[[i]], features = VariableFeatures(obj_list[[i]]), npcs = max(dims, n_pcs), verbose = FALSE)
}

features <- if (is.null(features_use)) {
  SelectIntegrationFeatures(object.list = obj_list, nfeatures = 2000)
} else {
  features_use
}

reduction_use <- if (mode == "rpca") "rpca" else "cca"
message("Running FindIntegrationAnchors(mode=", mode, ", reduction=", reduction_use,
        ", dims=1:", dims, ", k.anchor=", k_anchor, ")")

anchors <- FindIntegrationAnchors(
  object.list = obj_list,
  anchor.features = features,
  reduction = reduction_use,
  dims = 1:dims,
  k.anchor = k_anchor,
  verbose = FALSE
)

# ------------------------------------------------------------------
# Atlas-scale safe: integrate EMBEDDINGS using the FULL PCA reduction
# (must match AnchorSet cell count)
# ------------------------------------------------------------------
message("Running IntegrateEmbeddings(dims=1:", dims, ") [atlas-scale safe]")

obj_int <- IntegrateEmbeddings(
  anchorset = anchors,
  reductions = obj[["pca"]],           # ✅ FULL object PCA: 237,144 cells
  new.reduction.name = "integrated.rpca",
  dims = 1:dims
)
emb <- Embeddings(obj_int, reduction = "integrated.rpca")
if (!is.null(n_pcs) && n_pcs > 0 && ncol(emb) > n_pcs) {
  emb <- emb[, 1:n_pcs, drop = FALSE]
}

emb_out <- paste0(out_prefix, "_seurat_integrated_rpca.csv")
write.csv(emb, file = emb_out, quote = FALSE)
message("Saved integrated embedding CSV: ", emb_out)

pca_legacy_out <- paste0(out_prefix, "_seurat_pca.csv")
write.csv(emb, file = pca_legacy_out, quote = FALSE)
message("Saved legacy PCA CSV (compatibility): ", pca_legacy_out)

rds_out <- paste0(out_prefix, "_seurat_integrated_embedding_only.rds")
saveRDS(obj_int, file = rds_out)
message("Saved Seurat RDS: ", rds_out)

message("DONE")
