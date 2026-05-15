#!/usr/bin/env Rscript
# =============================================================================
# make_tier2_figures.R  — FINAL VERSION
# =============================================================================
# Outputs (all saved to OUTPUT_DIR):
#   tier2_bubble_plot.pdf            -- Luecken-style bubble/dot plot with
#                                       runtime column (colour + width only)
#   tier2_radar_overlaid.pdf         -- all methods on one radar
#   tier2_radar_small_multiples.pdf  -- 2×4 per-method radar charts
#   tier2_score_table.pdf            -- booktabs academic score table
#
# Inputs:
#   tier2_dotplot_data.csv           -- from make_tier2_overall_ranking.py
#   ti_methods_runtime_table.csv     -- per-task elapsed_seconds (methods × tasks)
#
# Runtime column encoding:
#   Rectangle COLOUR = mean runtime across tasks (log scale)
#                      soft pink (fast) → periwinkle blue (slow)
#                      Palette: celestial palette (Image 1)
#   Rectangle WIDTH  = std across tasks — wider = more variable across datasets
#   Rectangle HEIGHT = fixed for all methods
#   Numeric label to the right = exact mean in seconds
#
# Legend:
#   Three separate colour bars: metric rank | Friedman rank | runtime
#   Fully separated (BAR_SEP = 0.22 in) so labels never overlap
#   Two-paragraph caption explaining circle and rectangle encodings
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse); library(ggplot2); library(dplyr); library(tidyr)
  library(patchwork); library(cowplot); library(scales)
  library(ggtext); library(grid); library(Cairo)
})

has_fmsb <- requireNamespace("fmsb", quietly = TRUE)
if (!has_fmsb) stop("install.packages('fmsb')")

# =============================================================================
# Paths
# =============================================================================
SCRIPT_DIR <- tryCatch({
  args <- commandArgs(trailingOnly = FALSE)
  fa   <- grep("--file=", args, value = TRUE)
  if (length(fa) > 0) dirname(normalizePath(sub("--file=", "", fa)))
  else dirname(normalizePath(rstudioapi::getSourceEditorContext()$path))
}, error = function(e) getwd())

CSV_PATH    <- file.path(SCRIPT_DIR, "tier2_dotplot_data.csv")
RUNTIME_CSV <- "/data1/esraa/Thesis-Project/Results/Trajectory_Inference/Quantitive_Metrics_Plotting_scripts/ti_methods_runtime_table.csv"
OUTPUT_DIR  <- "/data1/esraa/Thesis-Project/Results/Trajectory_Inference/Tier2_Overall_Ranking"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

BASE_FONT <- "Helvetica"   # swap to "Liberation Sans" on Linux if needed
BASE_SIZE <- 9

METHOD_COLORS <- c(
  "CellRank"  = "#4E79A7", "ElPiGraph" = "#F28E2B", "Monocle3"  = "#59A14F",
  "PAGA + DPT"  = "#E15759", "SCORPIUS"  = "#B07AA1", "Slingshot" = "#9C755F",
  "TSCAN"     = "#BAB0AC", "VIA"       = "#EDC948"
)

# =============================================================================
# Load runtime data
# =============================================================================
message("\n-- Reading runtime: ", RUNTIME_CSV)

METHOD_NAME_MAP <- c(
  "cellrank"  = "CellRank",  "elpigraph" = "ElPiGraph",
  "monocle3"  = "Monocle3",  "paga"      = "PAGA + DPT",
  "scorpius"  = "SCORPIUS",  "slingshot" = "Slingshot",
  "tscan"     = "TSCAN",     "via"       = "VIA"
)

runtime_raw <- read_csv(RUNTIME_CSV, show_col_types = FALSE)
task_cols   <- colnames(runtime_raw)[-1]   # all columns except the first (method name)

runtime_stats <- runtime_raw %>%
  rename(method_raw = 1) %>%
  mutate(method = recode(tolower(method_raw), !!!METHOD_NAME_MAP)) %>%
  rowwise() %>%
  mutate(
    rt_mean = mean(c_across(all_of(task_cols)), na.rm = TRUE),
    rt_std  = sd(  c_across(all_of(task_cols)), na.rm = TRUE)
  ) %>%
  ungroup() %>%
  select(method, rt_mean, rt_std)

message("  Runtime summary (mean ± std, seconds):")
print(as.data.frame(runtime_stats))

# =============================================================================
# Load accuracy / ranking data
# =============================================================================
message("\n-- Reading: ", CSV_PATH)
if (!file.exists(CSV_PATH)) stop("CSV not found: ", CSV_PATH)

raw <- read_csv(CSV_PATH, show_col_types = FALSE)

method_order <- raw %>%
  filter(metric_key == "Composite rank") %>%
  arrange(friedman_rank) %>%
  pull(method)
message("  Method order (best→worst): ", paste(method_order, collapse = " > "))

raw <- raw %>% mutate(method = factor(method, levels = method_order))

MLEVELS <- c("Marker concordance", "Marker monotonicity", "kNN smoothness",
             "Root purity", "Topology consistency", "Composite rank")
MLMAP <- c(
  "Marker concordance"   = "Marker\nconcordance",
  "Marker monotonicity"  = "Marker\nmonotonicity",
  "kNN smoothness"       = "kNN\nsmoothness",
  "Root purity"          = "Root\npurity",
  "Topology consistency" = "Topology\nconsistency",
  "Composite rank"       = "Composite\nFriedman rank"
)

bubble_df <- raw %>%
  mutate(
    metric_key     = factor(metric_key, levels = MLEVELS),
    metric_display = factor(MLMAP[as.character(metric_key)], levels = unname(MLMAP)),
    size_norm      = replace_na(as.numeric(size_norm), 0),
    rank_label     = replace_na(as.character(rank_label), "N/A"),
    is_composite   = as.logical(is_composite)
  )

mean_wide <- raw %>%
  filter(!is_composite) %>%
  select(method, metric_key, mean_score) %>%
  pivot_wider(names_from = metric_key, values_from = mean_score) %>%
  column_to_rownames("method") %>%
  as.data.frame()
mean_wide  <- mean_wide[method_order, , drop = FALSE]

ranks_wide <- raw %>%
  select(method, metric_key, friedman_rank) %>%
  pivot_wider(names_from = metric_key, values_from = friedman_rank) %>%
  column_to_rownames("method") %>%
  as.data.frame()
ranks_wide <- ranks_wide[method_order, , drop = FALSE]
std_wide   <- mean_wide * NA

# =============================================================================
# Colour palettes
# =============================================================================

# Accuracy bubbles: peach → deep purple
MPAL <- colorRampPalette(c(
  "#F3D2CF", "#F2CCC6", "#EA99B0", "#EA91AD",
  "#D35195", "#A4237B", "#66106D", "#8A0F7A", "#5A0D6C"
))(100)

# Friedman composite rank squares: purple gradient
FPAL <- rev(colorRampPalette(c(
  "#2b1a3c", "#3b2c54", "#4b3d6c", "#6b5b95",
  "#836aa5", "#9a79b4", "#b798cd", "#d4b7e6"
))(100))

# Runtime rectangles: Celestial palette (Image 1)
# soft pink (fast / low runtime) → periwinkle blue (slow / high runtime)
RTPAL <- colorRampPalette(c(
  "#E8B4C0",  # soft pink       — fast
  "#C8DDD0",  # light mint
  "#8BBFAA",  # mint teal
  "#7BAAB8",  # steel teal
  "#6B8DC4"   # periwinkle blue — slow
))(100)

RT_HEADER_COLOR <- "#809bce"   # periwinkle blue (Image 2) for Runtime header band

# =============================================================================
# BUBBLE PLOT
# =============================================================================
make_bubble_plot <- function(df, runtime_stats, out) {

  # ── Method metadata ─────────────────────────────────────────────────────────
  LANG <- c(
    "CellRank"  = "Python", "ElPiGraph" = "Python", "Monocle3"  = "R",
    "PAGA + DPT"  = "Python", "SCORPIUS"  = "R",      "Slingshot" = "R",
    "TSCAN"     = "R",      "VIA"       = "Python"
  )
  CATG <- c(
    "CellRank"  = "Probabilistic/Markov model",
    "ElPiGraph" = "Principal graph",
    "Monocle3"  = "Principal graph",
    "PAGA + DPT"  = "Graph + diffusion pseudotime",
    "SCORPIUS"  = "MDS + principal curve",
    "Slingshot" = "Cluster-based MST + principal curves",
    "TSCAN"     = "Cluster-based MST",
    "VIA"       = "Graph + lazy random walk"
  )
  TOPO <- c(
    "CellRank"  = "Directed Graph", "ElPiGraph" = "Tree / Graph",
    "Monocle3"  = "Tree / Graph",          "PAGA + DPT"  = "Graph",
    "SCORPIUS"  = "Linear",         "Slingshot" = "Tree",
    "TSCAN"     = "Tree",           "VIA"       = "Graph"
  )

  methods  <- levels(df$method)
  n_m      <- length(methods)
  met_keys <- c("Marker concordance", "Marker monotonicity", "kNN smoothness",
                "Root purity", "Topology consistency", "Composite rank")
  n_bub    <- length(met_keys)
  n_acc    <- n_bub - 1L   # pure accuracy metrics (excludes composite)

  # Runtime lookup
  rt_lookup <- runtime_stats %>%
    filter(method %in% methods) %>%
    column_to_rownames("method")

  rt_means_v <- rt_lookup$rt_mean[is.finite(rt_lookup$rt_mean) & rt_lookup$rt_mean > 0]
  rt_log_min <- if (length(rt_means_v)) log10(min(rt_means_v)) else 0
  rt_log_max <- if (length(rt_means_v)) log10(max(rt_means_v)) else 1

  rt_stds_v  <- rt_lookup$rt_std[is.finite(rt_lookup$rt_std) & rt_lookup$rt_std > 0]
  rt_std_max <- if (length(rt_stds_v)) max(rt_stds_v) else 1

  # ── Column widths (inches) ──────────────────────────────────────────────────
  W_METHOD  <- 1.12
  W_LANG    <- 1.05
  W_CATG    <- 1.55
  W_TOPO    <- 1.65
  BUB_W     <- 0.25
  BUB_LAST  <- 0.45
  W_RUNTIME <- 0.80

  col_w <- c(W_METHOD, W_LANG, W_CATG, W_TOPO,
             rep(BUB_W, n_bub - 1L), BUB_LAST,
             W_RUNTIME)
  n_cols   <- length(col_w)
  cum_x    <- cumsum(c(0, col_w))
  TW       <- sum(col_w)
  col_cx   <- function(j) cum_x[j] + col_w[j] / 2

  n_txt    <- 4L
  bub_x0   <- cum_x[n_txt + 1]          # left edge of accuracy bubble columns
  rt_col_j <- n_txt + n_bub + 1L        # column index of runtime column

  # ── Heights ──────────────────────────────────────────────────────────────────
  ROW_H   <- 0.270
  HDR_H   <- 0.18
  LBL_H   <- 1.65   # space for rotated metric labels

  # ── Legend geometry ──────────────────────────────────────────────────────────
  BAR_W_L  <- 0.13   # colorbar width
  BAR_SEP  <- 0.17   # increased gap between bars (wider for runtime separation)
  LEG_GAP  <- 0.48   # increased clearance from table right edge to first bar

  # ── Figure dimensions ─────────────────────────────────────────────────────────
  L_MAR  <- 0.22
  R_MAR  <- LEG_GAP + 3 * (BAR_W_L + BAR_SEP) + 0.55
  T_MAR  <- 0.50
  B_MAR  <- LBL_H + 0.55   # extra space for two-paragraph caption

  FIG_W  <- L_MAR + TW + R_MAR
  FIG_H  <- T_MAR + HDR_H + n_m * ROW_H + B_MAR

  # ── Open PDF ──────────────────────────────────────────────────────────────────
  pdf(out, width = FIG_W, height = FIG_H, family = BASE_FONT)
  on.exit(dev.off(), add = TRUE)
  par(mar = c(0,0,0,0), bg = "white")
  plot(0, 0, type = "n",
       xlim = c(0, FIG_W), ylim = c(0, FIG_H),
       asp = NA, axes = FALSE, xlab = "", ylab = "", xaxs = "i", yaxs = "i")

  TX <- L_MAR

  # ── Y coordinates ─────────────────────────────────────────────────────────────
  hdr_bot  <- B_MAR + n_m * ROW_H
  hdr_top  <- hdr_bot + HDR_H
  tbl_bot2 <- B_MAR

  row_top2 <- function(i) hdr_bot - (i - 1) * ROW_H
  row_bot2 <- function(i) hdr_bot -  i      * ROW_H
  row_cy2  <- function(i) (row_top2(i) + row_bot2(i)) / 2

  MAX_R <- ROW_H * 0.38
  MIN_R <- ROW_H * 0.05

  # ── Title ─────────────────────────────────────────────────────────────────────
  text(TX + TW / 2, hdr_top + 0.30,
       "Overall method ranking  \u2013\u2013  Friedman rank-based composite",
       cex = 0.80, font = 1, col = "#111111", adj = c(0.5, 0.5))

  # ── Alternating row shading ───────────────────────────────────────────────────
  for (i in seq_len(n_m)) {
    shade <- if (i %% 2 == 1) "white" else "#E2E2E2"
    rect(TX, row_bot2(i), TX + TW, row_top2(i), col = shade, border = NA)
  }

  # ── Booktabs rules ────────────────────────────────────────────────────────────
  segments(TX, hdr_top,  TX + TW, hdr_top,  lwd = 1.6, lend = 1, col = "#111111")
  segments(TX, hdr_bot,  TX + TW, hdr_bot,  lwd = 0.9, lend = 1, col = "#111111")
  segments(TX, tbl_bot2, TX + TW, tbl_bot2, lwd = 1.6, lend = 1, col = "#111111")

  # Dotted separators: text block | accuracy | runtime
  sep_acc <- TX + bub_x0
  sep_rt  <- TX + cum_x[rt_col_j]
  for (sx in c(sep_acc, sep_rt))
    segments(sx, hdr_bot, sx, tbl_bot2, lwd = 0.4, lty = 3, col = "#BBBBBB")

  # ── Header bands ─────────────────────────────────────────────────────────────
  rect(TX,       hdr_bot, TX + bub_x0,         hdr_top, col = "#a3b7ca",        border = NA)
  rect(sep_acc,  hdr_bot, sep_rt,              hdr_top, col = "#9a79b4",        border = NA)
  rect(sep_rt,   hdr_bot, TX + TW,             hdr_top, col = RT_HEADER_COLOR,  border = NA)

  segments(TX, hdr_top, TX + TW, hdr_top, lwd = 1.6, lend = 1, col = "#111111")
  segments(TX, hdr_bot, TX + TW, hdr_bot, lwd = 0.9, lend = 1, col = "#111111")
  for (sx in c(sep_acc, sep_rt))
    segments(sx, hdr_bot, sx, hdr_top, lwd = 0.8, col = "#FFFFFF")

  hdr_cy <- (hdr_bot + hdr_top) / 2

  text(TX + cum_x[1] + 0.06, hdr_cy, "Method",
       cex = 0.68, font = 1, col = "white", adj = c(0, 0.5))
  text(TX + col_cx(2), hdr_cy, "Language",
       cex = 0.66, font = 1, col = "white", adj = c(0.5, 0.5))
  text(TX + col_cx(3), hdr_cy, "Category",
       cex = 0.66, font = 1, col = "white", adj = c(0.5, 0.5))
  text(TX + col_cx(4), hdr_cy, "Maximum produced Topology",
       cex = 0.66, font = 1, col = "white", adj = c(0.5, 0.5))

  acc_mid <- TX + bub_x0 + (cum_x[rt_col_j] - bub_x0) / 2
  text(acc_mid, hdr_cy, "Accuracy Metrics",
       cex = 0.68, font = 1, col = "white", adj = c(0.5, 0.5))

  text(TX + col_cx(rt_col_j), hdr_cy, "Runtime",
       cex = 0.68, font = 1, col = "white", adj = c(0.5, 0.5))

  # ── Data rows ─────────────────────────────────────────────────────────────────
  for (i in seq_along(methods)) {
    m  <- methods[i]
    cy <- row_cy2(i)

    text(TX + cum_x[1] + 0.06, cy, m,
         cex = 0.76, font = 1, col = "#111111", adj = c(0, 0.5))
    text(TX + col_cx(2), cy, LANG[[m]],
         cex = 0.68, font = 1, col = "#333333", adj = c(0.5, 0.5))
    text(TX + col_cx(3), cy, CATG[[m]],
         cex = 0.66, font = 1, col = "#333333", adj = c(0.5, 0.5))
    text(TX + col_cx(4), cy, TOPO[[m]],
         cex = 0.66, font = 1, col = "#333333", adj = c(0.5, 0.5))

    # ── Accuracy bubble columns ────────────────────────────────────────────────
    for (j in seq_along(met_keys)) {
      mk   <- met_keys[j]
      jj   <- n_txt + j
      bcx  <- TX + col_cx(jj)
      is_c <- (mk == "Composite rank")

      rd <- df %>%
        filter(as.character(method) == m,
               as.character(metric_key) == mk)

      if (nrow(rd) == 0 || is.na(rd$friedman_rank[1])) {
        if (is_c) {
          rw <- col_w[jj] * 0.78; rh <- ROW_H * 0.62
          rect(bcx - rw/2, cy - rh/2, bcx + rw/2, cy + rh/2,
               col = "#EEEEEE", border = "white", lwd = 0.4)
        } else {
          text(bcx, cy, "N/A", cex = 0.52, col = "#BBBBBB", adj = c(0.5, 0.5))
        }
        next
      }

      sz  <- max(rd$size_norm[1], 0)
      rv  <- rd$friedman_rank[1]
      r   <- MIN_R + (sz ^ 0.55) * (MAX_R - MIN_R)
      r   <- max(MIN_R, min(MAX_R, r))
      idx <- max(1L, min(100L, as.integer(ceiling((rv - 1) / max(n_m - 1, 1) * 99 + 1))))

      if (is_c) {
        sq <- min(col_w[jj], ROW_H) * 0.45
        rect(bcx - sq/2, cy - sq/2, bcx + sq/2, cy + sq/2,
             col = FPAL[idx], border = "black", lwd = 0.7)
      } else {
        th <- seq(0, 2*pi, length.out = 80)
        polygon(bcx + r * cos(th), cy + r * sin(th),
                col = MPAL[idx], border = "black", lwd = 0.7)
      }
    }

    # ── Runtime column: fixed-height rectangle, variable width only ───────────
    rt_left   <- TX + cum_x[rt_col_j]
    rt_cw     <- col_w[rt_col_j]
    rt_pad    <- rt_cw * 0.05
    rect_max_w <- rt_cw - 2 * rt_pad - 0.15
    rect_h    <- ROW_H * 0.38

    if (m %in% rownames(rt_lookup) &&
        is.finite(rt_lookup[m, "rt_mean"]) &&
        rt_lookup[m, "rt_mean"] > 0) {

      rt_m <- rt_lookup[m, "rt_mean"]
      rt_s <- rt_lookup[m, "rt_std"];  if (!is.finite(rt_s)) rt_s <- 0

      # Colour index: log-normalised mean runtime
      nc  <- max(0, min(1, (log10(rt_m) - rt_log_min) /
                             max(rt_log_max - rt_log_min, 1e-6)))
      idx_rt <- max(1L, min(100L, as.integer(ceiling(nc * 99 + 1))))

      # Rectangle width: std normalised
      ns     <- if (rt_std_max > 0) min(rt_s / rt_std_max, 1) else 0
      rect_w <- rt_pad + ns * (rect_max_w - rt_pad)

      rect(rt_left + rt_pad,
           cy - rect_h / 2,
           rt_left + rt_pad + rect_w,
           cy + rect_h / 2,
           col = RTPAL[idx_rt], border = "#555555", lwd = 0.5)

      lbl <- if (rt_m >= 100) sprintf("%.0fs", rt_m) else
             if (rt_m >= 10)  sprintf("%.1fs", rt_m) else
                              sprintf("%.2fs", rt_m)
      text(rt_left + rt_pad + rect_w + 0.03, cy, lbl,
           cex = 0.48, col = "#333333", adj = c(0, 0.5))

    } else {
      text(TX + col_cx(rt_col_j), cy, "N/A",
           cex = 0.52, col = "#BBBBBB", adj = c(0.5, 0.5))
    }
  }

  # ── Rotated metric labels below table ────────────────────────────────────────
  bub_hdrs <- c("Marker concordance", "Marker monotonicity", "kNN smoothness",
                "Root purity", "Topology consistency", "Friedman rank")
  for (j in seq_along(bub_hdrs)) {
    jj <- n_txt + j
    text(TX + col_cx(jj), tbl_bot2 - 0.05, bub_hdrs[j],
         cex = 0.66, font = 1, col = "#333333", adj = c(1, 0.5), srt = 90)
  }
  text(TX + col_cx(rt_col_j), tbl_bot2 - 0.05,
       "Runtime (mean \u00b1 variability)",
       cex = 0.66, font = 1, col = "#333333", adj = c(1, 0.5), srt = 90)

  # =============================================================================
  # RIGHT-SIDE LEGEND — three bars, fully separated
  # =============================================================================
  leg_x0  <- TX + TW + LEG_GAP
  bar_h_l <- n_m * ROW_H * 0.45
  bar_top <- hdr_bot - (n_m * ROW_H - bar_h_l) / 2
  bar_y0  <- bar_top - bar_h_l
  n_seg   <- 80

  draw_cbar <- function(x, y0, ytop, pal, reverse = FALSE) {
    bh <- (ytop - y0) / n_seg
    idx_seq <- if (reverse) rev(seq_len(n_seg)) else seq_len(n_seg)
    for (s in seq_len(n_seg)) {
      idx_c <- ceiling(idx_seq[s] / n_seg * 100)
      rect(x, y0 + (s - 1) * bh, x + BAR_W_L, y0 + s * bh,
           col = pal[max(1, min(100, idx_c))], border = NA)
    }
    rect(x, y0, x + BAR_W_L, ytop, col = NA, border = "#AAAAAA", lwd = 0.3)
  }

  # Bar 1: Metric rank (pink-purple)
  x1 <- leg_x0
  draw_cbar(x1, bar_y0, bar_top, MPAL, reverse = TRUE)
  text(x1 + BAR_W_L/2, bar_top + 0.06, "Metric\nrank",
       cex = 0.42, col = "#333333", adj = c(0.5, 0))
  text(x1 - 0.03, bar_top, "best",  cex = 0.36, col = "#666666", adj = c(1, 0.5))
  text(x1 - 0.03, bar_y0,  "worst", cex = 0.36, col = "#666666", adj = c(1, 0.5))

  # Bar 2: Friedman rank (purple)
  x2 <- x1 + BAR_W_L + BAR_SEP
  draw_cbar(x2, bar_y0, bar_top, FPAL, reverse = TRUE)
  text(x2 + BAR_W_L/2, bar_top + 0.06, "Friedman\nrank",
       cex = 0.42, col = "#333333", adj = c(0.5, 0))
  text(x2 + BAR_W_L + 0.03, bar_top, "best",  cex = 0.36, col = "#666666", adj = c(0, 0.5))
  text(x2 + BAR_W_L + 0.03, bar_y0,  "worst", cex = 0.36, col = "#666666", adj = c(0, 0.5))

  # Bar 3: Runtime colour (celestial palette)
  x3 <- x2 + BAR_W_L + BAR_SEP
  draw_cbar(x3, bar_y0, bar_top, RTPAL, reverse = TRUE)
  text(x3 + BAR_W_L/2, bar_top + 0.15, "Runtime",
       cex = 0.42, col = "#333333", adj = c(0.5, 0))
  text(x3 + BAR_W_L + 0.03, bar_top, "fast", cex = 0.36, col = "#666666", adj = c(0, 0.5))
  text(x3 + BAR_W_L + 0.03, bar_y0,  "slow", cex = 0.36, col = "#666666", adj = c(0, 0.5))

  # Runtime width mini-legend (below bar 3)
  vl_y      <- bar_y0 - 0.65
  h_vbar    <- 0.065
  TITLE_CX  <- x3 + BAR_W_L/2 - 0.2
  MAX_BW    <- 1.0 * BAR_W_L * 1.1
  BAR_START <- TITLE_CX - MAX_BW / 2
  LBL_X     <- TITLE_CX + MAX_BW / 2 + 0.05

  text(TITLE_CX, vl_y, "Runtime (width = std)",
       cex = 0.50, col = "#333333", adj = c(0.5, 1), font = 1)
  vl_y <- vl_y - 0.2

  for (pair in list(c(0.18, "low std"), c(1.0, "high std"))) {
    bw        <- as.numeric(pair[1]) * BAR_W_L * 1.1
    bar_left  <- TITLE_CX - bw / 2 - 0.2
    rect(bar_left, vl_y - h_vbar/2, bar_left + bw, vl_y + h_vbar/2,
         col = "#8BBFAA", border = "#555555", lwd = 0.4)
    text(LBL_X, vl_y, pair[2],
         cex = 0.44, col = "#555555", adj = c(0, 0.5))
    vl_y <- vl_y - (h_vbar + 0.09)
  }

  # Score circle size legend (centred below bars 1–2)
  sc_cx    <- (x1 + x2 + BAR_W_L) / 2 + 0.2
  pcts     <- c(0, 0.25, 0.50, 0.75, 1.00)
  sc_gap   <- MAX_R * 2 + 0.01
  total_sw <- (length(pcts) - 1) * sc_gap
  sc_x0    <- sc_cx - total_sw / 2
  sc_y     <- bar_y0 - MAX_R - 0.22

  text(sc_cx, bar_y0 - 0.1, "Score (circle size)",
       cex = 0.50, col = "#333333", adj = c(0.5, 1))
  for (k in seq_along(pcts)) {
    pct   <- pcts[k]
    r_leg <- MIN_R + (pct ^ 0.55) * (MAX_R - MIN_R)
    cx_k  <- sc_x0 + (k - 1) * sc_gap
    th    <- seq(0, 2*pi, length.out = 60)
    polygon(cx_k + r_leg * cos(th), sc_y + r_leg * sin(th),
            col = "white", border = "#555555", lwd = 0.5)
  }
  text(sc_x0,            sc_y - MAX_R - 0.04, "0%",   cex = 0.38, col = "#555555", adj = c(0.5, 1))
  text(sc_x0 + total_sw, sc_y - MAX_R - 0.04, "100%", cex = 0.38, col = "#555555", adj = c(0.5, 1))

  # =============================================================================
  # Two-paragraph caption
  # =============================================================================
  cap1 <- paste0(
    "Accuracy metrics: circle colour encodes Friedman rank (light pink = rank 1/best, ",
    "dark purple = rank 8/worst); circle size encodes normalised mean score across tasks ",
    "(larger = higher score). ", "Friedman rank column: filled square, purple scale."
  )
  cap2 <- paste0(
    "Runtime column: rectangle colour encodes mean runtime (log scale) using the celestial palette ",
    "\u2014 soft pink = fast, periwinkle blue = slow; rectangle width encodes the standard deviation ",
    "across the 6 biological tasks (wider rectangle = more variable runtime); ",
    "rectangle height is fixed across methods. ", "Numeric label = exact mean in seconds."
  )

  text(TX, tbl_bot2 - LBL_H + 0.08, cap1,
       cex = 0.56, font = 3, col = "#555555", adj = c(0, 1))
  text(TX, tbl_bot2 - LBL_H - 0.27, cap2,
       cex = 0.56, font = 3, col = "#555555", adj = c(0, 1))

  message("  [saved] ", basename(out))
}

# =============================================================================
# RADAR helpers  (unchanged)
# =============================================================================
draw_radar_labels <- function(n_vars, var_names, label_r = 1.32, cex = 0.72) {
  angles <- seq(90, 90 - 360, length.out = n_vars + 1)[-(n_vars + 1)] * pi / 180
  for (i in seq_len(n_vars)) {
    ang <- angles[i]
    lab <- switch(var_names[i],
      "Marker concordance"   = "Marker\nconcordance",
      "Marker monotonicity"  = "Marker\nmonotonicity",
      "kNN smoothness"       = "kNN\nsmoothness",
      "Root purity"          = "Root\npurity",
      "Topology consistency" = "Topology\nconsistency",
      var_names[i]
    )
    ha <- if (cos(ang) > 0.15) 0 else if (cos(ang) < -0.15) 1 else 0.5
    va <- if (sin(ang) > 0.15) 0 else if (sin(ang) < -0.15) 1 else 0.5
    text(label_r * cos(ang), label_r * sin(ang),
         labels = lab, cex = cex, col = "#333333", adj = c(ha, va), font = 1)
  }
}

draw_radar <- function(vals_df, clr, name, alpha = 0.20, lwd = 2.2,
                       calcex = 0.55, seg = 4) {
  nv   <- ncol(vals_df)
  rgb_ <- col2rgb(clr) / 255
  fc   <- rgb(rgb_[1], rgb_[2], rgb_[3], alpha = alpha)
  angs <- seq(90, 90 - 360, length.out = nv + 1)[-(nv + 1)] * pi / 180
  plot(0, 0, type = "n", xlim = c(-1.9,1.9), ylim = c(-1.9,1.9),
       asp = 1, axes = FALSE, xlab = "", ylab = "", xaxs = "i", yaxs = "i")
  ring_vals <- seq(1/seg, 1, by = 1/seg)
  for (rv in ring_vals) {
    px <- rv * cos(angs); py <- rv * sin(angs)
    for (k in seq_len(nv)) {
      k2 <- if (k == nv) 1L else k + 1L
      lines(c(px[k], px[k2]), c(py[k], py[k2]), col = "#CCCCCC", lty = 2, lwd = 1.0)
    }
  }
  for (ang in angs) lines(c(0, cos(ang)), c(0, sin(ang)), col = "#CCCCCC", lwd = 0.5)
  lab_ang <- angs[1] + 0.12
  for (rv in ring_vals)
    text(rv * cos(lab_ang) + 0.02, rv * sin(lab_ang),
         sprintf("%.0f (%%)", rv * 100), cex = calcex, col = "#AAAAAA", adj = c(0, 0.5))
  rv_vals <- as.numeric(vals_df)
  px <- rv_vals * cos(angs); py <- rv_vals * sin(angs)
  polygon(c(px, px[1]), c(py, py[1]), col = fc, border = NA)
  lines(c(px, px[1]), c(py, py[1]), col = clr, lwd = lwd, lend = "round", ljoin = "round")
  points(px, py, pch = 21, cex = 0.90, bg = "white", col = clr, lwd = 1.5)
  draw_radar_labels(nv, colnames(vals_df), label_r = 1.45, cex = 0.72)
}

make_radar_small_multiples <- function(df, out) {
  CairoPDF(out, width = 16, height = 9, family = BASE_FONT)
  par(mfrow = c(2, 4), mar = c(2.2, 2.2, 2.8, 2.2),
      oma = c(1.8, 0, 2.0, 0), bg = "white")
  for (m in rownames(df)) {
    clr <- METHOD_COLORS[[m]]; if (is.null(clr)) clr <- "#cbc7d8"
    draw_radar(df[m, , drop = FALSE], clr, m)
    title(main = m, col.main = clr, font.main = 2, cex.main = 1.08, line = 0.5)
  }
  mtext("Per-method performance profile  |  mean score across 6 tasks  |  higher = better",
        side = 3, outer = TRUE, cex = 0.92, font = 2, col = "#111111", line = 0.6)
  mtext("kNN smoothness = 1/(1+MAD)  |  All metrics: higher = better",
        side = 1, outer = TRUE, cex = 0.58, col = "#999999", font = 3, line = 0.3)
  dev.off()
  message("  [saved] ", basename(out))
}

make_radar_overlaid <- function(df, out) {
  methods   <- rownames(df)
  n_vars    <- ncol(df)
  var_names <- colnames(df)
  angles    <- seq(90, 90 - 360, length.out = n_vars + 1)[-(n_vars + 1)] * pi / 180
  pdf(out, width = 9, height = 10, family = BASE_FONT)
  on.exit(dev.off(), add = TRUE)
  par(mar = c(1, 1, 3.2, 1), bg = "white")
  plot(0, 0, type = "n", xlim = c(-1.75,1.75), ylim = c(-1.85,1.65),
       asp = 1, axes = FALSE, xlab = "", ylab = "", xaxs = "i", yaxs = "i")
  for (rv in seq(0.25, 1.00, by = 0.25)) {
    px <- rv * cos(angles); py <- rv * sin(angles)
    for (k in seq_len(n_vars)) {
      k2 <- if (k == n_vars) 1L else k + 1L
      lines(c(px[k], px[k2]), c(py[k], py[k2]), col = "#CCCCCC", lty = 2, lwd = 1.2)
    }
  }
  for (ang in angles) lines(c(0, cos(ang)), c(0, sin(ang)), col = "#CCCCCC", lwd = 0.6)
  lab_ang <- angles[1] + 0.14
  for (rv in c(0.25, 0.50, 0.75))
    text(rv * cos(lab_ang) + 0.02, rv * sin(lab_ang),
         sprintf("%.0f%%", rv * 100), cex = 0.52, col = "#AAAAAA", adj = c(0, 0.5))
  draw_radar_labels(n_vars, var_names, label_r = 1.38, cex = 0.82)
  for (m in methods) {
    clr  <- METHOD_COLORS[[m]]; if (is.null(clr)) clr <- "#555555"
    vals <- as.numeric(df[m, ])
    px   <- c(vals * cos(angles), vals[1] * cos(angles[1]))
    py   <- c(vals * sin(angles), vals[1] * sin(angles[1]))
    rgb_ <- col2rgb(clr) / 255
    polygon(px, py, col = rgb(rgb_[1], rgb_[2], rgb_[3], alpha = 0.13), border = NA)
  }
  for (m in methods) {
    clr  <- METHOD_COLORS[[m]]; if (is.null(clr)) clr <- "#555555"
    vals <- as.numeric(df[m, ])
    px   <- c(vals * cos(angles), vals[1] * cos(angles[1]))
    py   <- c(vals * sin(angles), vals[1] * sin(angles[1]))
    lines(px, py, col = clr, lwd = 2.2, lend = "round", ljoin = "round")
    points(px[-length(px)], py[-length(py)],
           pch = 21, cex = 0.95, bg = "white", col = clr, lwd = 1.5)
  }
  mtext("Overall method performance profile",
        side = 3, line = 1.8, cex = 1.10, font = 2, col = "#111111")
  mtext("Mean score across all 6 tasks  |  higher = better",
        side = 3, line = 0.50, cex = 0.75, font = 3, col = "#555555")
  legend(x = 0, y = -1.58, xjust = 0.5, yjust = 0.5, xpd = TRUE,
         legend = methods, col = unname(METHOD_COLORS[methods]),
         pt.bg = unname(METHOD_COLORS[methods]),
         pch = 21, pt.cex = 1.15, lwd = 2.0, lty = 1, ncol = 4,
         bty = "n", cex = 0.85, text.col = unname(METHOD_COLORS[methods]),
         x.intersp = 0.7, y.intersp = 1.0)
  message("  [saved] ", basename(out))
}

# =============================================================================
# SCORE TABLE  (unchanged)
# =============================================================================
make_score_table <- function(df_mean, df_std, ranks_df, out) {
  methods     <- rownames(df_mean)
  metric_cols <- colnames(df_mean)[colnames(df_mean) != "Composite rank"]
  n_methods   <- length(methods); n_metrics <- length(metric_cols)
  cell <- matrix("--", nrow = n_methods, ncol = n_metrics + 1,
                 dimnames = list(methods, c(metric_cols, "Composite rank")))
  for (m in methods) {
    for (met in metric_cols) {
      mn <- df_mean[m, met]
      cell[m, met] <- if (!is.na(mn)) sprintf("%.3f", round(mn, 3)) else "--"
    }
    comp <- ranks_df[m, "Composite rank"]
    cell[m, "Composite rank"] <- if (!is.na(comp)) sprintf("%.2f", comp) else "--"
  }
  best_per <- sapply(metric_cols, function(met) {
    v <- setNames(df_mean[, met], rownames(df_mean)); names(which.max(v))
  })
  W_RANK <- 0.28; W_METHOD <- 1.35; W_MET <- 1.15; W_COMP <- 0.90
  col_w  <- c(W_RANK, W_METHOD, rep(W_MET, n_metrics), W_COMP)
  n_cols <- length(col_w); cum_x <- cumsum(c(0, col_w)); TW <- sum(col_w)
  col_cx <- function(j) cum_x[j] + col_w[j] / 2
  ROW_H <- 0.330; HDR_H <- 0.420
  L_MAR <- 0.50; R_MAR <- 0.30; T_MAR <- 1.00; B_MAR <- 0.42
  FIG_W <- L_MAR + TW + R_MAR; FIG_H <- T_MAR + HDR_H + n_methods * ROW_H + B_MAR
  pdf(out, width = FIG_W, height = FIG_H, family = BASE_FONT)
  on.exit(dev.off(), add = TRUE)
  par(mar = c(0,0,0,0), bg = "white")
  plot(0, 0, type = "n", xlim = c(0,FIG_W), ylim = c(0,FIG_H),
       asp = NA, axes = FALSE, xlab = "", ylab = "", xaxs = "i", yaxs = "i")
  TX <- L_MAR; TY <- B_MAR + HDR_H + n_methods * ROW_H
  hdr_top <- TY; hdr_bot <- TY - HDR_H; data_top <- hdr_bot
  tbl_bot <- data_top - n_methods * ROW_H
  row_cy  <- function(i) data_top - (i - 0.5) * ROW_H
  hdr_cy  <- (hdr_top + hdr_bot) / 2
  text(FIG_W/2, TY + 0.66,
       "Table 1.  Performance summary of trajectory inference methods",
       cex = 1.00, font = 2, col = "#000000", adj = c(0.5, 0.5))
  text(FIG_W/2, TY + 0.36,
       paste0("Methods ranked by composite Friedman rank (rank\u00a01 = best).  ",
              "Mean score across 6 tasks.  All metrics: higher\u00a0=\u00a0better."),
       cex = 0.70, font = 3, col = "#333333", adj = c(0.5, 0.5))
  lend <- 1
  segments(TX, hdr_top, TX+TW, hdr_top, lwd=1.8, lend=lend, col="#000000")
  segments(TX, hdr_bot, TX+TW, hdr_bot, lwd=0.9, lend=lend, col="#000000")
  segments(TX, tbl_bot, TX+TW, tbl_bot, lwd=1.8, lend=lend, col="#000000")
  hdrs <- c("Rank","Method","Marker\nconcordance","Marker\nmonotonicity",
            "kNN\nsmoothness","Root\npurity","Topology\nconsistency","Composite\nrank")
  for (j in seq_len(n_cols)) {
    if (j == 2)
      text(TX + cum_x[2] + 0.06, hdr_cy, hdrs[j], cex=0.78, font=2, col="#000000", adj=c(0,0.5))
    else
      text(TX + cum_x[j] + col_w[j] - 0.06, hdr_cy, hdrs[j], cex=0.78, font=2, col="#000000", adj=c(1,0.5))
  }
  for (i in seq_along(methods)) {
    m <- methods[i]; cy <- row_cy(i)
    text(TX + cum_x[1] + col_w[1] - 0.04, cy, as.character(i),
         cex=0.74, font=1, col="#777777", adj=c(1,0.5))
    text(TX + cum_x[2] + 0.06, cy, m,
         cex=0.80, font=if(i==1)2 else 1, col="#000000", adj=c(0,0.5))
    for (j in seq_along(metric_cols)) {
      met <- metric_cols[j]; val <- cell[m, met]; is_best <- isTRUE(best_per[[met]] == m)
      rx <- TX + cum_x[j+2] + col_w[j+2] - 0.08
      text(rx, cy, val, cex=0.76, font=if(is_best)2 else 1, col="#000000", adj=c(1,0.5))
      if (is_best) {
        tw <- nchar(val) * 0.052
        segments(rx - tw*2, cy - ROW_H*0.35, rx, cy - ROW_H*0.35, lwd=0.75, col="#000000")
      }
    }
    rx_c <- TX + cum_x[n_cols] + col_w[n_cols] - 0.08
    text(rx_c, cy, cell[m, "Composite rank"],
         cex=0.76, font=if(i==1)2 else 1, col="#000000", adj=c(1,0.5))
  }
  text(TX, tbl_bot - 0.16,
       "Bold + underline = best value in column.  Composite = mean Friedman rank across 5 metrics.  kNN smoothness = 1/(1+MAD).",
       cex=0.62, font=3, col="#555555", adj=c(0,1))
  message("  [saved] ", basename(out))
}

# =============================================================================
# MAIN
# =============================================================================
radar_cols <- setdiff(colnames(mean_wide), "Composite rank")
radar_df   <- mean_wide[, radar_cols, drop = FALSE]
radar_df[is.na(radar_df)] <- 0

message("\n-- Bubble plot (with runtime column)...")
make_bubble_plot(
  bubble_df, runtime_stats,
  file.path(OUTPUT_DIR, "tier2_bubble_plot.pdf")
)

message("\n-- Overlaid radar...")
make_radar_overlaid(radar_df, file.path(OUTPUT_DIR, "tier2_radar_overlaid.pdf"))

message("\n-- Small multiples radar...")
make_radar_small_multiples(radar_df, file.path(OUTPUT_DIR, "tier2_radar_small_multiples.pdf"))

message("\n-- Score table...")
make_score_table(mean_wide, std_wide, ranks_wide,
                 file.path(OUTPUT_DIR, "tier2_score_table.pdf"))

message("\n-- Done. All files in: ", OUTPUT_DIR, "\n")