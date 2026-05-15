#!/usr/bin/env Rscript
# tscan_run.R — Robust TSCAN runner compatible with TSCAN 1.40.0+
#
# Outputs:
#   pseudotime.csv  columns: cell_id, pseudotime
#   edges.csv       columns: source, target, weight, directed

suppressPackageStartupMessages({
  if (!requireNamespace("TSCAN", quietly = TRUE))
    stop("Package 'TSCAN' is required.")
  if (!requireNamespace("igraph", quietly = TRUE))
    stop("Package 'igraph' is required.")
  # Usually installed as a dependency of TSCAN; used for the cleanest extraction.
  # We'll still have fallbacks if it's missing.
})

# -------------------------
# CLI parser
# -------------------------
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

if (is.null(emb_path) || is.null(meta_path) || is.null(cluster_key) ||
    is.null(root_cluster) || is.null(seed) || is.null(out_dir)) {
  stop("Required: --embedding --meta --cluster_key --root_cluster --seed --out_dir")
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(seed)

message("TSCAN version: ", as.character(utils::packageVersion("TSCAN")))
message("igraph version: ", as.character(utils::packageVersion("igraph")))
message("seed: ", seed)

# -------------------------
# Load data
# -------------------------
X_df <- read.csv(emb_path, row.names = 1L, check.names = FALSE)
X <- as.matrix(X_df)
storage.mode(X) <- "double"

meta <- read.csv(meta_path, stringsAsFactors = FALSE)
if (!("cell_id" %in% colnames(meta))) stop("meta CSV must contain column 'cell_id'")
if (!(cluster_key %in% colnames(meta))) stop(paste0("meta missing cluster_key column: ", cluster_key))

rownames(meta) <- meta$cell_id

# Align meta to embedding rows
missing_meta <- setdiff(rownames(X), rownames(meta))
if (length(missing_meta) > 0) {
  stop("meta is missing ", length(missing_meta), " cells present in embedding (first few): ",
       paste(head(missing_meta, 5), collapse = ", "))
}
meta <- meta[rownames(X), , drop = FALSE]

labs <- trimws(as.character(meta[[cluster_key]]))
bad <- is.na(labs) | labs == "" | labs == "NA"
if (any(bad)) {
  message("Dropping ", sum(bad), " cells with missing cluster labels in meta[", cluster_key, "].")
  keep <- !bad
  X <- X[keep, , drop = FALSE]
  meta <- meta[keep, , drop = FALSE]
  labs <- labs[keep]
}

cl <- factor(labs)
if (nlevels(cl) < 2) stop("Need >=2 clusters for TSCAN MST; got n_levels=", nlevels(cl))

root_cluster <- trimws(as.character(root_cluster))
if (!(root_cluster %in% levels(cl))) {
  stop("root_cluster '", root_cluster, "' not found in meta[", cluster_key, "]. Levels (first 30): ",
       paste(head(levels(cl), 30), collapse = " | "))
}

message("root_cluster label: ", root_cluster)
message("n_clusters: ", nlevels(cl))

cell_ids <- rownames(X)

# -------------------------
# Build cluster MST
# -------------------------
cMST <- NULL

# Preferred in modern TSCAN: clusters=
cMST <- tryCatch(
  TSCAN::createClusterMST(X, clusters = cl),
  error = function(e) {
    message("createClusterMST(X, clusters=cl) failed: ", conditionMessage(e))
    NULL
  }
)

# Fallback: positional second argument
if (is.null(cMST)) {
  cMST <- tryCatch(
    TSCAN::createClusterMST(X, cl),
    error = function(e) {
      stop("createClusterMST failed with known signatures. Last error: ", conditionMessage(e))
    }
  )
}

# Resolve MST igraph object
mst <- NULL
if (igraph::is_igraph(cMST)) {
  mst <- cMST
} else if (!is.null(cMST$MSTtree) && igraph::is_igraph(cMST$MSTtree)) {
  mst <- cMST$MSTtree
} else if (!is.null(cMST$MST) && igraph::is_igraph(cMST$MST)) {
  mst <- cMST$MST
} else {
  stop("Cannot find an igraph MST in createClusterMST() output. Names: ", paste(names(cMST), collapse = ", "))
}

vnames <- igraph::V(mst)$name
if (!is.null(vnames)) {
  message("MST vertex names (first 20): ", paste(head(vnames, 20), collapse = ", "))
}

# -------------------------
# IMPORTANT FIX:
# mapping MUST come from mapCellsToEdges()
# (It contains left/right cluster + distances needed by orderCells.)
# -------------------------
if (!exists("mapCellsToEdges", where = asNamespace("TSCAN"), inherits = FALSE)) {
  stop("TSCAN::mapCellsToEdges not found in this TSCAN version.")
}

mapping <- tryCatch(
  TSCAN::mapCellsToEdges(X, mst, clusters = cl),
  error = function(e) {
    stop("mapCellsToEdges() failed: ", conditionMessage(e))
  }
)

# Sanity check: required columns exist
req_cols <- c("left.cluster", "right.cluster", "left.distance", "right.distance")
missing_cols <- setdiff(req_cols, colnames(as.data.frame(mapping)))
if (length(missing_cols) > 0) {
  stop("mapCellsToEdges() returned mapping missing required fields: ",
       paste(missing_cols, collapse = ", "),
       ". Found: ", paste(colnames(as.data.frame(mapping)), collapse = ", "))
}

# -------------------------
# Handle multi-component MST: choose one start per component
# -------------------------
comp <- igraph::components(mst)
n_comp <- comp$no
message("MST connected components: ", n_comp)

start_vec <- character(0)
for (k in seq_len(n_comp)) {
  verts_k <- igraph::V(mst)$name[which(comp$membership == k)]
  verts_k <- as.character(verts_k)
  if (root_cluster %in% verts_k) {
    start_vec <- c(start_vec, root_cluster)
  } else {
    start_vec <- c(start_vec, verts_k[1])
  }
}
start_vec <- unique(start_vec)
message("Start clusters used (per component): ", paste(start_vec, collapse = " | "))

# -------------------------
# orderCells(mapping, mst, start=...)
# -------------------------
if (!exists("orderCells", where = asNamespace("TSCAN"), inherits = FALSE)) {
  stop("TSCAN::orderCells not found in this TSCAN version.")
}

ord <- tryCatch(
  TSCAN::orderCells(mapping, mst, start = start_vec),
  error = function(e) {
    stop("orderCells() failed: ", conditionMessage(e))
  }
)

message("orderCells return class: ", paste(class(ord), collapse = ", "))

# -------------------------
# Extract pseudotime matrix and unify to 1 value per cell
# unified <- rowMeans(pathStat(ord), na.rm=TRUE)
# -------------------------
pt_mat <- NULL

# Best path: TrajectoryUtils::pathStat()
if (requireNamespace("TrajectoryUtils", quietly = TRUE)) {
  pt_mat <- tryCatch(TrajectoryUtils::pathStat(ord), error = function(e) NULL)
}

# Fallback 1: SummarizedExperiment assay (if available)
if (is.null(pt_mat) && requireNamespace("SummarizedExperiment", quietly = TRUE)) {
  pt_mat <- tryCatch(SummarizedExperiment::assay(ord), error = function(e) NULL)
}

# Fallback 2: direct slot access
if (is.null(pt_mat)) {
  pt_mat <- tryCatch(ord@assays@data[[1]], error = function(e) NULL)
}

if (is.null(pt_mat)) {
  stop("Could not extract pseudotime matrix from orderCells() output via pathStat(), assay(), or S4 slots.")
}

pt_mat <- as.matrix(pt_mat)
storage.mode(pt_mat) <- "double"
message("pseudotime matrix dim: ", paste(dim(pt_mat), collapse = "x"))

# Unify to a single pseudotime per cell:
pt_vec <- rowMeans(pt_mat, na.rm = TRUE)
pt_vec[!is.finite(pt_vec)] <- NA_real_

# Align to input cell order using rownames if possible
rn <- rownames(pt_mat)
if (!is.null(rn) && length(rn) == length(pt_vec)) {
  names(pt_vec) <- as.character(rn)
  pt_out <- pt_vec[as.character(cell_ids)]
  # If this fails due to name mismatch, fall back to row order:
  if (sum(is.finite(pt_out)) == 0 && sum(is.finite(pt_vec)) > 0) {
    message("WARNING: rownames(pt_mat) did not match input cell_ids; falling back to row order.")
    pt_out <- pt_vec
    if (length(pt_out) != length(cell_ids)) stop("Row-order fallback length mismatch.")
    names(pt_out) <- as.character(cell_ids)
  }
} else {
  # No rownames → assume row order matches X / mapping input order.
  pt_out <- pt_vec
  if (length(pt_out) != length(cell_ids)) stop("Pseudotime length mismatch with input cells.")
  names(pt_out) <- as.character(cell_ids)
}

n_finite <- sum(is.finite(pt_out))
message("Finite pseudotime values: ", n_finite, " / ", length(pt_out))
if (n_finite == 0) {
  stop("Extracted pseudotime has no finite values (all NA/NaN). This indicates mapping/orderCells failed to assign cells.")
}

# Write pseudotime.csv
pt_df <- data.frame(
  cell_id    = as.character(cell_ids),
  pseudotime = as.numeric(pt_out),
  stringsAsFactors = FALSE
)
write.csv(pt_df, file = file.path(out_dir, "pseudotime.csv"), row.names = FALSE)

# -------------------------
# Write edges.csv (cluster-level MST)
# -------------------------
raw_edges <- igraph::as_edgelist(mst)
raw_edges <- matrix(raw_edges, ncol = 2L)

w <- igraph::E(mst)$weight
if (is.null(w)) w <- rep(1.0, nrow(raw_edges))
w <- as.numeric(w)

edges_df <- data.frame(
  source   = as.character(raw_edges[, 1L]),
  target   = as.character(raw_edges[, 2L]),
  weight   = w,
  directed = FALSE,
  stringsAsFactors = FALSE
)
write.csv(edges_df, file = file.path(out_dir, "edges.csv"), row.names = FALSE)

message("TSCAN completed successfully. Outputs written to: ", out_dir)