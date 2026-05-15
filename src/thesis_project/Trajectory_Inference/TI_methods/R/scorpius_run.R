#!/usr/bin/env Rscript
# scorpius_run.R
# SCORPIUS trajectory inference runner for TI benchmarking.
#
# Fix history
# -----------
# 2026-03-17 (root-orientation fix):
#   - Added optional --root_cell_id argument.
#   - After computing traj$time, if root_cell_id is provided and
#     pt[root_cell_id] > 0.5, flip: pt = max(pt) - pt
#   - All edge builders use corrected pt so group ordering is consistent.
#   - Note: the primary orientation correction is also in the Python adapter
#     (scorpius_adapter.py :: _orient_pseudotime_to_root).  This R-level
#     step is an independent redundant safety net.
#
# Required inputs:
#   --embedding  CSV (cells x dims), row.names = cell IDs
#   --seed       integer
#   --out_dir    directory
#
# Optional:
#   --meta_csv     CSV with at least: cell_id + group_key column
#   --group_key    column name in meta_csv
#   --root_cell_id barcode of the designated root cell (for orientation check)
#
# Optional (edge construction control):
#   --edge_mode      group|waypoints|cells
#   --n_waypoints    integer (default: 50)
#   --weight_mode    unit|pseudotime|pseudotime_scaled|euclidean
#   --require_groups 0|1
#   --directed       0|1
#
# Outputs:
#   pseudotime.csv : cell_id, pseudotime  (root-anchored)
#   edges.csv      : source, target, weight, directed

suppressPackageStartupMessages({
  if (!requireNamespace("SCORPIUS", quietly = TRUE))
    stop("Package 'SCORPIUS' is required.")
  if (!requireNamespace("TSP", quietly = TRUE))
    stop("Package 'TSP' is required.")
})

# ── TSP %||% patch ─────────────────────────────────────────────────────────
patch_TSP_percent_or_or <- function(verbose = TRUE) {
  ns     <- asNamespace("TSP")
  fnames <- ls(ns, all.names = TRUE)
  def    <- quote(`%||%` <- function(x, y) if (!is.null(x)) x else y)
  patched <- character(0)
  for (nm in fnames) {
    obj <- get(nm, envir = ns, inherits = FALSE)
    if (!is.function(obj)) next
    b    <- body(obj)
    if (is.null(b)) next
    btxt <- paste(deparse(b), collapse = "\n")
    if (!grepl("%\\|\\|%", btxt, perl = TRUE)) next
    if (grepl("`%\\|\\|%`\\s*<-\\s*function", btxt, perl = TRUE)) next
    obj2 <- obj
    if (is.call(b) && identical(b[[1]], as.name("{"))) {
      body(obj2) <- as.call(c(as.name("{"), def, as.list(b)[-1]))
    } else {
      body(obj2) <- as.call(list(as.name("{"), def, b))
    }
    utils::assignInNamespace(nm, obj2, ns = "TSP")
    patched <- c(patched, nm)
  }
  if (verbose) {
    if (length(patched) == 0) {
      message("TSP patch: no functions referencing %||% found (or already patched).")
    } else {
      message("TSP patch: injected local %||% into ", length(patched), " function(s).")
    }
  }
  invisible(TRUE)
}
patch_TSP_percent_or_or(verbose = TRUE)

# ── CLI parser ─────────────────────────────────────────────────────────────
parse_args <- function(x) {
  args <- list()
  i <- 1L
  while (i <= length(x)) {
    key <- x[[i]]
    if (startsWith(key, "--")) {
      if (i == length(x)) stop(paste("Missing value for argument:", key))
      k <- substring(key, 3L)
      k <- gsub("-", "_", k, fixed = FALSE)
      args[[k]] <- x[[i + 1L]]
      i <- i + 2L
    } else {
      i <- i + 1L
    }
  }
  args
}

args <- parse_args(commandArgs(trailingOnly = TRUE))

# Required
emb_path <- args[["embedding"]]
seed     <- suppressWarnings(as.integer(args[["seed"]]))
out_dir  <- args[["out_dir"]]

# Optional
meta_csv     <- args[["meta_csv"]]
group_key    <- args[["group_key"]]
root_cell_id <- args[["root_cell_id"]]   # NEW

edge_mode      <- args[["edge_mode"]]
n_waypoints    <- suppressWarnings(as.integer(args[["n_waypoints"]]))
weight_mode    <- args[["weight_mode"]]
require_groups <- args[["require_groups"]]
directed_flag  <- suppressWarnings(as.integer(args[["directed"]]))

if (is.null(emb_path) || is.null(out_dir) || is.null(seed) || is.na(seed))
  stop("Required arguments: --embedding --seed --out_dir")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(seed)

if (is.null(n_waypoints) || is.na(n_waypoints) || n_waypoints < 2L) n_waypoints <- 50L
if (is.null(weight_mode)  || !nzchar(weight_mode))  weight_mode  <- "unit"
if (is.null(edge_mode)    || !nzchar(edge_mode))    edge_mode    <- NA_character_
if (is.na(directed_flag)) directed_flag <- 0L
directed_flag <- ifelse(directed_flag != 0L, 1L, 0L)

has_meta <- !is.null(meta_csv) && nzchar(meta_csv) && file.exists(meta_csv) &&
  !is.null(group_key) && nzchar(group_key)

has_root <- !is.null(root_cell_id) && nzchar(root_cell_id)

if (is.null(require_groups) || !nzchar(require_groups))
  require_groups <- if (has_meta) "1" else "0"
require_groups <- require_groups %in% c("1", "TRUE", "True", "true")

if (is.na(edge_mode))
  edge_mode <- if (has_meta) "group" else "waypoints"
edge_mode <- tolower(edge_mode)

if (!edge_mode %in% c("group", "waypoints", "cells"))
  stop("--edge_mode must be: group | waypoints | cells")
if (!weight_mode %in% c("unit", "pseudotime", "pseudotime_scaled", "euclidean"))
  stop("--weight_mode must be: unit | pseudotime | pseudotime_scaled | euclidean")

# ── Load embedding ─────────────────────────────────────────────────────────
X     <- read.csv(emb_path, row.names = 1L, check.names = FALSE)
if (nrow(X) == 0L) stop("Embedding CSV is empty.")
space <- as.matrix(X)

# ── Run SCORPIUS ───────────────────────────────────────────────────────────
traj <- SCORPIUS::infer_trajectory(space)
if (!("time" %in% names(traj)))
  stop("SCORPIUS::infer_trajectory result missing 'time'. Check installation.")

pt <- as.numeric(traj$time)
if (is.null(names(pt)) || all(!nzchar(names(pt))))
  names(pt) <- rownames(space)
if (length(pt) != nrow(space))
  stop("Length mismatch: traj$time (", length(pt),
       ") vs nrow(embedding) (", nrow(space), ")")

# ── Root-based orientation (R-level safety net) ────────────────────────────
# Primary correction is in Python adapter; this is a redundant check.
# Flip if: root_cell_id provided AND pt[root_cell_id] > 0.5
pt_was_flipped <- FALSE

if (has_root) {
  root_str <- as.character(root_cell_id)
  if (root_str %in% names(pt)) {
    root_val <- as.numeric(pt[root_str])
    if (is.finite(root_val) && root_val > 0.5) {
      message(sprintf(
        "SCORPIUS R-orientation: root '%s' at pt=%.4f (> 0.5) — FLIPPING.",
        root_str, root_val
      ))
      pt <- max(pt) - pt
      pt_was_flipped <- TRUE
      message(sprintf(
        "After flip: root '%s' at pt=%.4f | range [%.4f, %.4f]",
        root_str, as.numeric(pt[root_str]), min(pt), max(pt)
      ))
    } else if (is.finite(root_val)) {
      message(sprintf(
        "SCORPIUS R-orientation: root '%s' at pt=%.4f — no flip needed.",
        root_str, root_val
      ))
    } else {
      message(sprintf(
        "SCORPIUS R-orientation: root '%s' has non-finite pt — skipping.",
        root_str
      ))
    }
  } else {
    message(sprintf(
      "SCORPIUS R-orientation: root_cell_id '%s' not in pt names — skipping.",
      root_str
    ))
  }
} else {
  message("SCORPIUS R-orientation: no root_cell_id provided — skipping.")
}

# ── Write pseudotime ───────────────────────────────────────────────────────
pt_df <- data.frame(
  cell_id    = names(pt),
  pseudotime = as.numeric(pt),
  stringsAsFactors = FALSE
)
write.csv(pt_df, file = file.path(out_dir, "pseudotime.csv"), row.names = FALSE)

# ── Edge builders ──────────────────────────────────────────────────────────
# All builders receive the CORRECTED pt (already flipped if needed).
# Group edges sort by ascending median pt → source=root-side, target=terminal-side.

build_group_edges <- function(meta_csv, group_key, pt_named,
                               weight_mode = "unit", directed_flag = 0L) {
  meta <- read.csv(meta_csv, stringsAsFactors = FALSE)
  if (!("cell_id" %in% colnames(meta)))
    stop("meta_csv must include column 'cell_id'")
  if (!(group_key %in% colnames(meta)))
    stop("meta_csv missing group_key: ", group_key)

  meta$cell_id <- as.character(meta$cell_id)
  grp          <- meta[[group_key]]
  names(grp)   <- meta$cell_id

  common <- intersect(names(pt_named), names(grp))
  if (length(common) < 2L)
    stop("Not enough overlap between meta_csv and pseudotime cells.")

  grp2 <- as.character(grp[common])
  pt2  <- as.numeric(pt_named[common])
  ok   <- !is.na(grp2) & nzchar(grp2) & is.finite(pt2)
  grp2 <- grp2[ok]; pt2 <- pt2[ok]

  med <- tapply(pt2, grp2, stats::median, na.rm = TRUE)
  med <- med[is.finite(med)]
  if (length(med) < 2L)
    stop("Need >= 2 groups with finite median pseudotime.")

  ord <- names(sort(med, decreasing = FALSE))   # low pt → high pt
  src <- ord[-length(ord)]
  tgt <- ord[-1L]

  w <- rep(1, length(src))
  if (weight_mode == "pseudotime") {
    w <- as.numeric(med[tgt] - med[src])
  } else if (weight_mode == "pseudotime_scaled") {
    w <- as.numeric((med[tgt] - med[src]) * (length(ord) - 1L))
  }

  data.frame(source = src, target = tgt, weight = as.numeric(w),
             directed = as.integer(directed_flag), stringsAsFactors = FALSE)
}

build_waypoint_edges <- function(traj_path, pt_named, n_waypoints = 50L,
                                  weight_mode = "unit", directed_flag = 0L) {
  path <- as.matrix(traj_path)
  if (is.null(path) || !is.matrix(path) || nrow(path) < 2L)
    return(data.frame(source = character(0), target = character(0),
                      weight = numeric(0), directed = integer(0)))

  ord_cells <- names(sort(pt_named, decreasing = FALSE))
  if (length(ord_cells) == nrow(path)) rownames(path) <- ord_cells

  K   <- min(as.integer(n_waypoints), nrow(path))
  idx <- unique(round(seq(1, nrow(path), length.out = K)))
  idx <- idx[idx >= 1 & idx <= nrow(path)]
  if (length(idx) < 2L)
    return(data.frame(source = character(0), target = character(0),
                      weight = numeric(0), directed = integer(0)))

  K2 <- length(idx); node_ids <- paste0("W", seq_len(K2))
  w  <- rep(1, K2 - 1L)
  if (weight_mode == "euclidean") {
    diffs <- path[idx[-1L],, drop = FALSE] - path[idx[-K2],, drop = FALSE]
    w <- sqrt(rowSums(diffs^2))
  } else if (weight_mode == "pseudotime") {
    pts <- sort(pt_named, decreasing = FALSE); pts_sel <- pts[idx]
    w   <- as.numeric(pts_sel[-1L] - pts_sel[-K2])
  } else if (weight_mode == "pseudotime_scaled") {
    pts <- sort(pt_named, decreasing = FALSE); pts_sel <- pts[idx]
    w   <- as.numeric((pts_sel[-1L] - pts_sel[-K2]) * (K2 - 1L))
  }
  data.frame(source = node_ids[-K2], target = node_ids[-1L],
             weight = as.numeric(w), directed = as.integer(directed_flag),
             stringsAsFactors = FALSE)
}

build_cell_edges <- function(pt_named, weight_mode = "unit", directed_flag = 0L) {
  ord_cells <- names(sort(pt_named, decreasing = FALSE))
  if (length(ord_cells) < 2L)
    return(data.frame(source = character(0), target = character(0),
                      weight = numeric(0), directed = integer(0)))
  src <- ord_cells[-length(ord_cells)]; tgt <- ord_cells[-1L]
  w   <- rep(1, length(src))
  if (weight_mode == "pseudotime") {
    pts <- sort(pt_named, decreasing = FALSE)
    w   <- as.numeric(pts[-1L] - pts[-length(pts)])
  } else if (weight_mode == "pseudotime_scaled") {
    pts <- sort(pt_named, decreasing = FALSE)
    w   <- as.numeric((pts[-1L] - pts[-length(pts)]) * (length(pts) - 1L))
  }
  data.frame(source = src, target = tgt, weight = as.numeric(w),
             directed = as.integer(directed_flag), stringsAsFactors = FALSE)
}

# ── Build edges ────────────────────────────────────────────────────────────
edges_df <- NULL

if (edge_mode == "group") {
  if (!has_meta) {
    if (require_groups)
      stop("edge_mode=group but meta_csv/group_key not provided.")
    message("edge_mode=group: meta not available; falling back to waypoints.")
    edge_mode <- "waypoints"
  } else {
    edges_df <- build_group_edges(meta_csv, group_key, pt,
                                   weight_mode = weight_mode,
                                   directed_flag = directed_flag)
  }
}

if (is.null(edges_df) && edge_mode == "waypoints") {
  if (!("path" %in% names(traj))) {
    message("No traj$path; falling back to cell edges.")
    edges_df <- build_cell_edges(pt, weight_mode = weight_mode,
                                  directed_flag = directed_flag)
  } else {
    edges_df <- build_waypoint_edges(traj$path, pt,
                                      n_waypoints   = n_waypoints,
                                      weight_mode   = weight_mode,
                                      directed_flag = directed_flag)
  }
}

if (is.null(edges_df) && edge_mode == "cells")
  edges_df <- build_cell_edges(pt, weight_mode = weight_mode,
                                directed_flag = directed_flag)

if (is.null(edges_df))
  edges_df <- data.frame(source = character(0), target = character(0),
                          weight = numeric(0), directed = integer(0))

write.csv(edges_df, file = file.path(out_dir, "edges.csv"), row.names = FALSE)

message("SCORPIUS completed successfully.")
message(sprintf("Pseudotime orientation flipped (R layer): %s", pt_was_flipped))
message("Pseudotime: ", nrow(pt_df), " cells -> ",
        file.path(out_dir, "pseudotime.csv"))
message("Edges: ", nrow(edges_df), " rows (mode=", edge_mode,
        ", weight=", weight_mode, ", directed=", directed_flag, ") -> ",
        file.path(out_dir, "edges.csv"))