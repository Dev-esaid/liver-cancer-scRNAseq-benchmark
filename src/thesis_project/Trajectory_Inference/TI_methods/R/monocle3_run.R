#!/usr/bin/env Rscript
# monocle3_run.R
# Monocle3 trajectory inference for TI benchmarking.
#
# Outputs written to out_dir/:
#   pseudotime.csv   columns: cell_id, pseudotime
#   edges.csv        columns: source, target, weight, directed
#
# Fix history
# -----------
# Fix 1 (original): Always use the provided (shared) UMAP coordinates via
#   SingleCellExperiment::reducedDims and re-inject them after cluster_cells /
#   learn_graph (defensive).
#
# Fix 2 (original — superseded by Fix 5): Robust group-level edge extraction.
#   Originally used ALL labels present on each principal-graph vertex to avoid
#   empty edge lists. Superseded by Fix 5 (majority-label strategy) which
#   solves the empty-edge problem via NA fallback while preventing the
#   over-connectivity that ALL-labels caused.
#
# Fix 3 (2026-03-18 — close_loop): Set close_loop = FALSE in learn_graph().
#   Rationale: All benchmarked trajectories represent experimentally validated
#   directed acyclic differentiation hierarchies (monocyte-to-TAM, T cell
#   exhaustion, CD4 differentiation, CD8 exhaustion, NK maturation). The
#   default close_loop = TRUE performs a post-hoc graph augmentation that adds
#   edges between principal graph tip nodes whose pairwise Euclidean distance
#   ratio is below euclidean_distance_ratio (default 1.0) AND whose geodesic
#   distance ratio exceeds geodesic_distance_ratio (default 1/3). For compact
#   intra-lineage UMAP embeddings — where all populations belong to a single
#   cell type family — these thresholds are systematically over-permissive,
#   causing Monocle3 to add loops between biologically unrelated terminal
#   states. Across all six tasks this produced near-complete graphs with
#   2.5–3.3x the expected number of edges, inconsistent with the acyclic tree
#   topology independently recovered by all other methods on identical data.
#   Setting close_loop = FALSE retains only the primary spanning tree and
#   places Monocle3 on equal topological footing with all other benchmarked
#   methods, which are acyclic by construction (Slingshot: simultaneous
#   principal curves; TSCAN: MST; ElPiGraph: elastic principal tree;
#   SCORPIUS: principal curve; VIA/PAGA: spanning trees).
#
# Fix 4 (2026-03-18 — parallel edge deduplication): Enforce strict canonical
#   deduplication of group-level edges before writing edges.csv.
#   Rationale: build_group_edges() used a nested loop over all label pairs for
#   each principal graph edge. The aggregate() call that followed collapsed
#   most but not all duplicates depending on the directed column type.
#   Fix 4 enforces alphabetical canonical ordering of source/target before
#   aggregation and converts directed to integer for stable aggregation,
#   guaranteeing each group pair appears at most once.
#
# Fix 5 (2026-03-18 — majority-label vertex assignment): Change
#   build_group_edges() from ALL-labels to MAJORITY-label per vertex.
#   Rationale: The ALL-labels strategy (Fix 2) assigned every unique cluster
#   label present among a vertex's cells to that vertex, then created a
#   group-level edge for every label pair on adjacent vertices. Because
#   Monocle3 fits ~50-100 internal principal graph waypoints and TME cluster
#   boundaries overlap in UMAP space, a single boundary vertex often receives
#   cells from 3-4 clusters simultaneously. This caused the nested label-pair
#   loop to generate C(4,2)=6 group edges per junction vertex — far exceeding
#   the n_groups-1 = 6 edges expected for the entire spanning tree. With 7
#   clusters and multiple boundary vertices, this produced 13-15 group edges,
#   i.e. 2-3x over-connection even after close_loop=FALSE and deduplication.
#
#   Fix 5 replaces ALL-labels with MAJORITY-label: each vertex is assigned
#   the single cluster label held by the plurality of its projected cells
#   (ties broken by which.max, which is deterministic under set.seed).
#   A group-level edge is then added only when two ADJACENT vertices hold
#   DIFFERENT majority labels — i.e. only at true cluster boundary crossings
#   of the principal graph. This produces exactly one edge per boundary
#   crossing, yielding a sparse group-level topology comparable in density
#   to the spanning trees produced by Slingshot, TSCAN, PAGA, and VIA.
#
#   The ALL-labels approach (Fix 2) was originally introduced to prevent
#   empty edge lists when vertices have very few cells. Fix 5 preserves
#   this robustness via a graceful NA fallback: vertices with zero assigned
#   cells are assigned NA and skipped in the edge-building loop.

suppressPackageStartupMessages({
  pkgs <- c("monocle3", "Matrix", "igraph", "SingleCellExperiment",
            "SummarizedExperiment", "S4Vectors")
  for (p in pkgs) {
    if (!requireNamespace(p, quietly = TRUE)) {
      stop("Package '", p, "' is required but not installed.")
    }
  }
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

# -------------------------
# Inject UMAP into cds (SingleCellExperiment reducedDims)
# -------------------------
inject_umap <- function(cds, umap_mat) {
  cell_ids <- colnames(cds)
  if (is.null(rownames(umap_mat))) rownames(umap_mat) <- cell_ids
  umap_mat <- umap_mat[cell_ids, , drop = FALSE]
  rd <- SingleCellExperiment::reducedDims(cds)
  if (is.null(rd)) rd <- S4Vectors::SimpleList()
  rd$UMAP <- umap_mat
  SingleCellExperiment::reducedDims(cds) <- rd
  cds
}

# -------------------------
# Coerce pr_graph_cell_proj_closest_vertex to named character vector
# -------------------------
coerce_closest_vertex <- function(closest_raw, g) {
  vnames <- igraph::V(g)$name

  if (is.data.frame(closest_raw) || is.matrix(closest_raw)) {
    vals     <- closest_raw[, 1L]
    cell_ids <- rownames(closest_raw)
  } else {
    vals     <- closest_raw
    cell_ids <- names(closest_raw)
  }

  if (is.null(cell_ids)) {
    stop("closest_vertex object has no rownames/names; cannot align to cells.")
  }

  vals_chr        <- as.character(vals)
  names(vals_chr) <- cell_ids

  # Map numeric indices to vertex names if needed
  if (!all(is.na(vals_chr)) && !all(vals_chr %in% vnames)) {
    suppressWarnings(idx <- as.integer(vals_chr))
    if (all(!is.na(idx)) && all(idx >= 1L & idx <= length(vnames))) {
      vals_chr        <- vnames[idx]
      names(vals_chr) <- cell_ids
    }
  }

  vals_chr
}

# -------------------------
# Build group-level edges from principal graph
# Fix 5 applied: majority-label per vertex (replaces ALL-labels from Fix 2)
# Fix 4 applied: canonical deduplication before returning
# -------------------------
build_group_edges <- function(cds, group_key, reduction_method = "UMAP") {

  empty_df <- data.frame(
    source   = character(),
    target   = character(),
    weight   = numeric(),
    directed = logical(),
    stringsAsFactors = FALSE
  )

  g <- monocle3::principal_graph(cds)[[reduction_method]]
  if (is.null(g) || !igraph::is_igraph(g)) {
    stop("principal_graph(cds)[['", reduction_method,
         "']] is missing or not an igraph object.")
  }

  ed <- igraph::as_edgelist(g, names = TRUE)
  if (is.null(ed) || nrow(ed) == 0L) return(empty_df)

  aux <- monocle3::principal_graph_aux(cds)[[reduction_method]]
  if (is.null(aux) || is.null(aux$pr_graph_cell_proj_closest_vertex)) {
    stop("principal_graph_aux '$pr_graph_cell_proj_closest_vertex' is missing.")
  }

  closest_v <- coerce_closest_vertex(aux$pr_graph_cell_proj_closest_vertex, g)

  cell_data <- SummarizedExperiment::colData(cds)
  if (!(group_key %in% colnames(cell_data))) {
    stop("group_key column not found in colData(cds): ", group_key)
  }
  grp        <- as.character(cell_data[[group_key]])
  names(grp) <- rownames(cell_data)
  grp        <- grp[names(closest_v)]   # align to closest_v cell order

  # ── Fix 5: MAJORITY label per vertex ────────────────────────────────────────
  # Assign each principal graph vertex the single cluster label held by the
  # plurality of its projected cells.  Vertices with no assigned cells receive
  # NA and are skipped in the edge-building loop below.
  #
  # Rationale: the ALL-labels approach (original Fix 2) generated one group
  # edge for every label pair touching each principal graph edge.  At UMAP
  # cluster boundaries a single vertex can hold cells from 3-4 clusters,
  # yielding C(4,2)=6 edges per junction — producing 2-3x over-connection.
  # Majority-label guarantees at most ONE edge per principal graph edge
  # (fired only when adjacent vertices belong to DIFFERENT clusters), giving
  # a sparse group topology comparable to the spanning trees of other methods.
  vnames <- igraph::V(g)$name

  vert_to_label <- vapply(
    vnames,
    function(v) {
      cell_mask <- closest_v == v
      if (!any(cell_mask)) return(NA_character_)
      labels_here <- grp[cell_mask]
      labels_here <- labels_here[!is.na(labels_here) & nzchar(labels_here)]
      if (length(labels_here) == 0L) return(NA_character_)
      tb <- table(labels_here)
      names(which.max(tb))          # majority label; ties broken deterministically
    },
    character(1L),
    USE.NAMES = TRUE
  )

  # Log vertex label summary for diagnostics
  n_labeled <- sum(!is.na(vert_to_label))
  n_total   <- length(vert_to_label)
  message(sprintf(
    "build_group_edges [Fix 5]: %d / %d principal vertices assigned a majority label.",
    n_labeled, n_total
  ))

  # ── Accumulate group-level edges ─────────────────────────────────────────────
  # An edge A→B is added when adjacent vertices hold DIFFERENT majority labels.
  # The weight accumulates the number of principal graph edges crossing that
  # cluster boundary (useful for weighting later).
  edge_counts <- new.env(parent = emptyenv())

  inc_edge <- function(a, b) {
    if (is.na(a) || is.na(b) || !nzchar(a) || !nzchar(b) || a == b) return()
    pair <- sort(c(a, b))                       # canonical alphabetical order
    key  <- paste(pair, collapse = "\x1f")      # unit-separator delimiter
    cur  <- edge_counts[[key]]
    if (is.null(cur)) cur <- 0L
    edge_counts[[key]] <- cur + 1L
  }

  for (i in seq_len(nrow(ed))) {
    v1 <- as.character(ed[i, 1L])
    v2 <- as.character(ed[i, 2L])
    a  <- vert_to_label[[v1]]
    b  <- vert_to_label[[v2]]
    inc_edge(a, b)                              # no-op if NA or same label
  }

  keys <- ls(edge_counts)
  if (length(keys) == 0L) {
    message("build_group_edges [Fix 5]: no cross-cluster edges found. ",
            "Check that group_key labels are present in the projected cells.")
    return(empty_df)
  }

  parts  <- strsplit(keys, "\x1f", fixed = TRUE)
  source <- vapply(parts, `[[`, character(1L), 1L)
  target <- vapply(parts, `[[`, character(1L), 2L)
  weight <- as.numeric(
    vapply(keys, function(k) edge_counts[[k]], integer(1L))
  )

  df <- data.frame(
    source   = source,
    target   = target,
    weight   = weight,
    directed = FALSE,
    stringsAsFactors = FALSE
  )

  # ── Fix 4: canonical deduplication (defensive — keys already canonical) ─────
  df <- df[!is.na(df$source) & !is.na(df$target), , drop = FALSE]
  df <- df[df$source != df$target, , drop = FALSE]

  if (nrow(df) == 0L) return(empty_df)

  # Enforce alphabetical canonical ordering one final time
  swap            <- df$source > df$target
  tmp             <- df$source[swap]
  df$source[swap] <- df$target[swap]
  df$target[swap] <- tmp

  # Stable aggregation: convert directed to integer, aggregate, convert back
  df$directed_int <- as.integer(df$directed)
  df <- aggregate(
    weight ~ source + target + directed_int,
    data = df,
    FUN  = sum
  )
  df$directed     <- as.logical(df$directed_int)
  df$directed_int <- NULL

  message(sprintf(
    "build_group_edges [Fix 5]: %d group-level edges written (expected ~%d for spanning tree).",
    nrow(df), length(unique(c(df$source, df$target))) - 1L
  ))

  df[, c("source", "target", "weight", "directed")]
}

# ============================================================
# Main
# ============================================================
args         <- parse_args(commandArgs(trailingOnly = TRUE))
expr_mtx     <- args[["expr_mtx"]]
genes_csv    <- args[["genes_csv"]]
cells_csv    <- args[["cells_csv"]]
meta_csv     <- args[["meta_csv"]]
umap_csv     <- args[["umap_csv"]]
group_key    <- args[["group_key"]]
root_cell_id <- args[["root_cell_id"]]
use_partition <- identical(toupper(trimws(args[["use_partition"]])), "TRUE")
seed         <- as.integer(args[["seed"]])
out_dir      <- args[["out_dir"]]

required_args <- c("expr_mtx", "genes_csv", "cells_csv", "meta_csv",
                   "umap_csv", "group_key", "root_cell_id",
                   "use_partition", "seed", "out_dir")
missing_args <- required_args[
  vapply(required_args, function(a) is.null(args[[a]]), logical(1L))
]
if (length(missing_args) > 0L) {
  stop("Missing required arguments: ", paste(missing_args, collapse = ", "))
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(seed)

# Log versions and key parameters for reproducibility
message("monocle3 version            : ",
        as.character(utils::packageVersion("monocle3")))
message("SingleCellExperiment version: ",
        as.character(utils::packageVersion("SingleCellExperiment")))
message("SummarizedExperiment version: ",
        as.character(utils::packageVersion("SummarizedExperiment")))
message("igraph version              : ",
        as.character(utils::packageVersion("igraph")))
message("seed                        : ", seed)
message("use_partition               : ", use_partition)
message("close_loop                  : FALSE  [Fix 3 — acyclic biology]")
message("build_group_edges           : majority-label per vertex [Fix 5]")

# ── Load expression matrix (genes x cells) ──────────────────────────────────
expr  <- Matrix::readMM(expr_mtx)
expr  <- methods::as(expr, "dgCMatrix")
genes <- read.csv(genes_csv, header = FALSE, stringsAsFactors = FALSE)$V1
cells <- read.csv(cells_csv, header = FALSE, stringsAsFactors = FALSE)$V1
rownames(expr) <- as.character(genes)
colnames(expr) <- as.character(cells)

# ── Load metadata ────────────────────────────────────────────────────────────
meta <- read.csv(meta_csv, stringsAsFactors = FALSE)
if (!("cell_id" %in% colnames(meta)))
  stop("meta.csv must include column 'cell_id'")
rownames(meta) <- meta$cell_id
meta <- meta[colnames(expr), , drop = FALSE]
if (!(group_key %in% colnames(meta)))
  stop("meta.csv is missing group_key column: ", group_key)

# ── Load shared UMAP ─────────────────────────────────────────────────────────
umap_df <- read.csv(umap_csv, row.names = 1L, check.names = FALSE)
umap_df <- umap_df[colnames(expr), , drop = FALSE]

if (all(c("UMAP1", "UMAP2") %in% colnames(umap_df))) {
  umap_mat <- as.matrix(umap_df[, c("UMAP1", "UMAP2"), drop = FALSE])
} else if (ncol(umap_df) >= 2L) {
  umap_mat <- as.matrix(umap_df[, 1:2, drop = FALSE])
  colnames(umap_mat) <- c("UMAP1", "UMAP2")
} else {
  stop("umap_csv must contain at least 2 columns (preferably UMAP1, UMAP2).")
}

# ── Build cell_data_set ───────────────────────────────────────────────────────
gene_meta <- data.frame(
  gene_short_name = rownames(expr),
  row.names       = rownames(expr),
  stringsAsFactors = FALSE
)

cds <- monocle3::new_cell_data_set(
  expr,
  cell_metadata = meta,
  gene_metadata = gene_meta
)

# Inject shared UMAP; keep a fixed copy for defensive re-injection
cds             <- inject_umap(cds, umap_mat)
umap_mat_fixed  <- SingleCellExperiment::reducedDims(cds)$UMAP

# ── cluster_cells ─────────────────────────────────────────────────────────────
# All clustering parameters at official defaults:
#   k=20, cluster_method="leiden", resolution=1e-5, num_iter=2,
#   partition_qval=0.05, weight=FALSE
# Only reduction_method and random_seed are set explicitly.
cc_formals <- names(formals(monocle3::cluster_cells))
if ("random_seed" %in% cc_formals) {
  cds <- monocle3::cluster_cells(
    cds,
    reduction_method = "UMAP",
    random_seed      = seed
  )
} else {
  set.seed(seed)
  cds <- monocle3::cluster_cells(cds, reduction_method = "UMAP")
}

# Defensive re-injection (cluster_cells may overwrite reducedDims)
cds <- inject_umap(cds, umap_mat_fixed)

# ── learn_graph ───────────────────────────────────────────────────────────────
# Fix 3: close_loop = FALSE (see header rationale).
# All other learn_graph parameters at official defaults:
#   euclidean_distance_ratio=1, geodesic_distance_ratio=1/3,
#   minimal_branch_len=10, prune_graph=TRUE, orthogonal_proj_tip=FALSE,
#   nn.k=25, ncenter=NULL (auto), maxiter=10, L1.gamma=0.5, L1.sigma=0.01
cds <- monocle3::learn_graph(
  cds,
  use_partition = use_partition,
  close_loop    = FALSE          # Fix 3
)

# Defensive re-injection (learn_graph may overwrite reducedDims)
cds <- inject_umap(cds, umap_mat_fixed)

# ── order_cells ───────────────────────────────────────────────────────────────
if (!(root_cell_id %in% colnames(cds))) {
  stop("root_cell_id not found among cell names: ", root_cell_id)
}
cds <- monocle3::order_cells(cds, root_cells = root_cell_id)

# ── Write pseudotime ──────────────────────────────────────────────────────────
pt    <- monocle3::pseudotime(cds)
pt_df <- data.frame(
  cell_id    = names(pt),
  pseudotime = as.numeric(pt),
  stringsAsFactors = FALSE
)
write.csv(pt_df, file = file.path(out_dir, "pseudotime.csv"), row.names = FALSE)

# ── Write topology edges (group-level) ───────────────────────────────────────
# Fix 5 (majority-label) + Fix 4 (deduplication) applied inside
# build_group_edges() — produces a sparse spanning tree topology.
edges_group <- build_group_edges(
  cds,
  group_key        = group_key,
  reduction_method = "UMAP"
)

write.csv(
  edges_group,
  file      = file.path(out_dir, "edges.csv"),
  row.names = FALSE
)

message("Monocle3 completed. Outputs written to: ", out_dir)