#!/usr/bin/env Rscript
# slingshot_run.R — Robust Slingshot runner for TI benchmarking
#
# Inputs (CLI args):
#   --embedding    embedding CSV (cells x dims, rownames = cell IDs)
#   --meta         metadata CSV  (columns: cell_id, <cluster_key>)
#   --cluster_key  column name for cluster labels in meta
#   --root_cluster cluster label to use as trajectory start
#   --seed         integer random seed
#   --out_dir      output directory
#
# Outputs written to out_dir/:
#   pseudotime.csv            columns: cell_id, pseudotime
#   pseudotime_lineages.csv   columns: cell_id, Lineage1, Lineage2, ...
#   edges.csv                 columns: source, target, weight, directed
#
# Key fixes:
#   (A) Avoid pkg::setter<- assignment (colData<- / reducedDim<-). Use library()
#       + set colData in SingleCellExperiment() constructor.
#   (B) Use slingPseudotime(..., na = FALSE) to avoid all-NA pseudotime
#       (na=TRUE yields NA for unassigned cells; na=FALSE returns arclength). :contentReference[oaicite:2]{index=2}
#   (C) Handle slingMST() returning igraph OR adjacency matrix. :contentReference[oaicite:3]{index=3}

suppressPackageStartupMessages({
  library(slingshot)
  library(SingleCellExperiment)
  library(S4Vectors)
  library(igraph)
})

# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------
parse_args <- function(x) {
  args <- list()
  i <- 1L
  while (i <= length(x)) {
    key <- x[[i]]
    if (startsWith(key, "--")) {
      if (i == length(x)) stop(paste("Missing value for argument:", key))
      args[[substring(key, 3L)]] <- x[[i + 1L]]
      i <- i + 2L
    } else {
      i <- i + 1L
    }
  }
  args
}

args         <- parse_args(commandArgs(trailingOnly = TRUE))
emb_path     <- args[["embedding"]]
meta_path    <- args[["meta"]]
cluster_key  <- args[["cluster_key"]]
root_cluster <- args[["root_cluster"]]
seed         <- suppressWarnings(as.integer(args[["seed"]]))
out_dir      <- args[["out_dir"]]

if (is.null(emb_path) || is.null(meta_path) || is.null(cluster_key) || is.null(out_dir)) {
  stop("Required args: --embedding --meta --cluster_key --out_dir [--root_cluster] --seed")
}
if (is.na(seed)) seed <- 0L

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(seed)

message("slingshot version: ", as.character(utils::packageVersion("slingshot")))
message("SingleCellExperiment version: ", as.character(utils::packageVersion("SingleCellExperiment")))
message("igraph version: ", as.character(utils::packageVersion("igraph")))
message("seed: ", seed)

# ---------------------------------------------------------------------------
# Load & align data
# ---------------------------------------------------------------------------
emb <- read.csv(emb_path, row.names = 1L, check.names = FALSE)
emb <- as.matrix(emb)
storage.mode(emb) <- "double"

meta <- read.csv(meta_path, stringsAsFactors = FALSE)
if (!("cell_id" %in% colnames(meta)))
  stop("meta CSV must contain a 'cell_id' column.")
if (!(cluster_key %in% colnames(meta)))
  stop(paste("meta CSV is missing cluster_key column:", cluster_key))

cell_ids <- rownames(emb)
message("n_cells: ", length(cell_ids))

rownames(meta) <- meta$cell_id
missing_meta <- setdiff(cell_ids, rownames(meta))
if (length(missing_meta) > 0) {
  stop("meta is missing ", length(missing_meta), " cells present in embedding (first few): ",
       paste(head(missing_meta, 5), collapse=", "))
}
meta <- meta[cell_ids, , drop = FALSE]

labs <- trimws(as.character(meta[[cluster_key]]))
bad <- is.na(labs) | labs == "" | labs == "NA"
if (any(bad)) {
  message("Dropping ", sum(bad), " cells with missing cluster labels in meta[", cluster_key, "].")
  keep <- !bad
  emb <- emb[keep, , drop = FALSE]
  meta <- meta[keep, , drop = FALSE]
  labs <- labs[keep]
  cell_ids <- rownames(emb)
}

cl <- factor(labs)
message("n_clusters: ", nlevels(cl))

if (!is.null(root_cluster) && nchar(trimws(root_cluster)) > 0L) {
  root_cluster <- trimws(as.character(root_cluster))
  message("root_cluster: ", root_cluster)
  if (!(root_cluster %in% levels(cl))) {
    stop("root_cluster '", root_cluster, "' not found in meta[", cluster_key, "]. Levels (first 30): ",
         paste(head(levels(cl), 30), collapse=" | "))
  }
} else {
  root_cluster <- NULL
  message("root_cluster: <none>")
}

# ---------------------------------------------------------------------------
# Build minimal SingleCellExperiment
# - IMPORTANT: set colnames (cell IDs), otherwise downstream mapping can break.
# - Set colData in constructor to avoid colData<- replacement issues.
# ---------------------------------------------------------------------------
counts <- matrix(0L, nrow = 1L, ncol = nrow(emb))
rownames(counts) <- "dummy_gene"
colnames(counts) <- cell_ids

sce <- SingleCellExperiment(
  assays  = list(counts = counts),
  colData = S4Vectors::DataFrame(cluster = cl, row.names = cell_ids)
)

# Store embedding as reducedDim
reducedDim(sce, "TI") <- emb

# ---------------------------------------------------------------------------
# Run Slingshot
# ---------------------------------------------------------------------------
if (!is.null(root_cluster)) {
  sds <- slingshot(
    sce,
    clusterLabels = "cluster",
    reducedDim    = "TI",
    start.clus    = root_cluster
  )
} else {
  sds <- slingshot(sce, clusterLabels = "cluster", reducedDim = "TI")
}

# ---------------------------------------------------------------------------
# Pseudotime extraction (robust)
# - Use na = FALSE so every cell gets an arclength per lineage. :contentReference[oaicite:4]{index=4}
# - Primary consensus: slingAvgPseudotime()
# - Fallback: min across lineages (row-min) from na=FALSE matrix
# ---------------------------------------------------------------------------
pt_mat <- slingPseudotime(sds, na = FALSE)

# Ensure rownames are cell_ids
if (is.null(rownames(pt_mat))) rownames(pt_mat) <- colnames(sce)
if (is.null(colnames(pt_mat))) colnames(pt_mat) <- paste0("Lineage", seq_len(ncol(pt_mat)))

pt_avg <- tryCatch(slingAvgPseudotime(sds), error = function(e) NULL)

pt_cons <- rep(NA_real_, nrow(pt_mat))
names(pt_cons) <- rownames(pt_mat)

if (!is.null(pt_avg)) {
  pt_avg <- as.numeric(pt_avg)
  names(pt_avg) <- rownames(pt_mat)
  pt_cons <- pt_avg
}

# Fill any non-finite with row-min of pt_mat
need <- !is.finite(pt_cons)
if (any(need)) {
  row_min <- apply(pt_mat, 1L, function(x) {
    x <- x[is.finite(x)]
    if (length(x) == 0L) NA_real_ else min(x)
  })
  pt_cons[need] <- row_min[need]
}

# Final fallback: if anything is still NA, set to 0 (should be extremely rare)
need2 <- !is.finite(pt_cons)
if (any(need2)) {
  message("WARNING: ", sum(need2), " cells still have non-finite pseudotime after fallbacks; setting to 0.")
  pt_cons[need2] <- 0.0
}

# Shift so min = 0 (common convention; helps metrics/plots)
finite_vals <- pt_cons[is.finite(pt_cons)]
if (length(finite_vals) > 0) {
  pt_cons <- pt_cons - min(finite_vals)
}

pt_df <- data.frame(
  cell_id    = names(pt_cons),
  pseudotime = as.numeric(pt_cons),
  stringsAsFactors = FALSE
)
write.csv(pt_df, file = file.path(out_dir, "pseudotime.csv"), row.names = FALSE)

# Per-lineage pseudotime (na=FALSE matrix)
pt_lin <- as.data.frame(pt_mat)
pt_lin$cell_id <- rownames(pt_mat)
# reorder: cell_id first
pt_lin <- pt_lin[, c("cell_id", setdiff(colnames(pt_lin), "cell_id")), drop = FALSE]
write.csv(pt_lin, file = file.path(out_dir, "pseudotime_lineages.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# MST edges between clusters (robust return-type handling) :contentReference[oaicite:5]{index=5}
# ---------------------------------------------------------------------------
mst_obj <- slingMST(sds)

edges_df <- NULL

if (igraph::is_igraph(mst_obj)) {
  el <- igraph::as_edgelist(mst_obj)
  el <- matrix(el, ncol = 2L)
  w  <- igraph::E(mst_obj)$weight
  if (is.null(w)) w <- rep(1.0, nrow(el))

  edges_df <- data.frame(
    source   = as.character(el[, 1L]),
    target   = as.character(el[, 2L]),
    weight   = as.numeric(w),
    directed = FALSE,
    stringsAsFactors = FALSE
  )

} else if (is.matrix(mst_obj) || inherits(mst_obj, "Matrix")) {
  m <- as.matrix(mst_obj)
  if (is.null(rownames(m)) || is.null(colnames(m))) {
    stop("slingMST returned an adjacency matrix without row/col names; cannot label edges.")
  }
  m[!is.finite(m)] <- 0
  idx <- which(upper.tri(m) & m != 0, arr.ind = TRUE)

  edges_df <- data.frame(
    source   = as.character(rownames(m)[idx[, 1L]]),
    target   = as.character(colnames(m)[idx[, 2L]]),
    weight   = as.numeric(m[idx]),
    directed = FALSE,
    stringsAsFactors = FALSE
  )

} else {
  stop(
    "Unsupported slingMST() return type: ",
    paste(class(mst_obj), collapse = ", "),
    ". Expected igraph or adjacency matrix."
  )
}

write.csv(edges_df, file = file.path(out_dir, "edges.csv"), row.names = FALSE)

message("Slingshot completed successfully. Outputs written to: ", out_dir)