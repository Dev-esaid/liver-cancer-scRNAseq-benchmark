#!/usr/bin/env Rscript
# =============================================================================
# make_ti_figures.R
# Coupled benchmark: Trajectory Inference sensitivity & scalability
#
# FIXED VERSION
# -----------------------------------------------------------------------------
# Main fixes applied
# 1) Do NOT renormalize metrics already in [0,1]:
#      - kendalls_w
#      - topology_jaccard_mean
#      - root_consistency_total
#      - integration_sensitivity_score
#    These are now plotted directly.
#
# 2) Branch CV metrics are converted once into score-like quantities:
#      branch_score = 1 / (1 + cv)
#    so lower CV -> higher score, without relative rescaling across rows.
#
# 3) Bubble/rectangle colours now reflect the same plotted score as size/width,
#    rather than rank. This removes the previous inconsistency.
#
# 4) Runtime remains separately normalized (log-based, lower = better), because
#    runtime is not naturally bounded to [0,1].
#
# 5) Palettes now run LIGHT -> DARK by construction.
#    Higher/better scores map to lighter colours.
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(dplyr)
  library(tidyr)
  library(grid)
  library(Cairo)
})

METRICS_CSV <- "/data1/esraa/Thesis-Project/Results/coupled_benchmark/Quantitive_metrics/coupled_metrics_master.csv"
RUNTIME_CSV <- "/data1/esraa/Thesis-Project/Results/coupled_benchmark/Quantitive_metrics/coupled_ti_runtime.csv"
OUTPUT_DIR  <- "/data1/esraa/Thesis-Project/Results/coupled_benchmark/Quantitive_metrics/figures"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

BASE_FONT <- "Helvetica"
PNG_RES   <- 180L

# -- Names & colours ----------------------------------------------------------
METHOD_COLORS <- c(
  "cellrank"  = "#4E79A7",
  "monocle3"  = "#F28E2B",
  "slingshot" = "#59A14F",
  "tscan"     = "#E15759"
)

TASK_LABELS <- c(
  "task_1" = "Monocyte / Macrophage TAM",
  "task_2" = "CD8 T-cell differentiation"
)

# -----------------------------------------------------------------------------
# RAW metric columns from CSV
# -----------------------------------------------------------------------------
RAW_METRIC_COLS <- c(
  "kendalls_w",
  "topology_jaccard_mean",
  "branch_leaves_cv",
  "branch_points_cv",
  "root_consistency_total",
  "integration_sensitivity_score"
)

# -----------------------------------------------------------------------------
# Plotting score columns
# These are the quantities actually visualized.
# -----------------------------------------------------------------------------
CIRC_METRICS <- c(
  "kendalls_w_score",
  "topology_jaccard_score",
  "branch_leaves_score",
  "branch_points_score",
  "root_consistency_score"
)

RECT_METRIC <- "integration_sensitivity_score_score"
ALL_PLOT_METRICS <- c(CIRC_METRICS, RECT_METRIC)

METRIC_LABELS_FLAT <- c(
  "kendalls_w_score"                    = "Kendall's W",
  "topology_jaccard_score"             = "Topology Jaccard",
  "branch_leaves_score"                = "Branch leaves score",
  "branch_points_score"                = "Branch points score",
  "root_consistency_score"             = "Root consistency",
  "integration_sensitivity_score_score" = "Integration sensitivity"
)

# -----------------------------------------------------------------------------
# Palettes: LIGHT -> DARK
# Higher score / better performance should look lighter.
# -----------------------------------------------------------------------------
TRAJ_PAL <- colorRampPalette(c(
  "#3d010e", "#79021c", "#8f0422","#9b0324","#b6042a", "#bc072e","#bf1b3e","#b80229"
))(100)

SENS_PAL <- colorRampPalette(c(
  "#050511","#201f5f","#203394","#4666ce","#6286e9"

))(100)

RTPAL <- colorRampPalette(c(
   "#3c4d30", "#587641","#66874c","#86bb5e"

))(100)

META_HEADER_COL  <- "#a3b7ca"
TRAJ_HEADER_COL  <- "#a32c2c"
SCALE_HEADER_COL <- "#526A40"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
clamp01 <- function(x) {
  x <- as.numeric(x)
  x[!is.finite(x)] <- NA_real_
  pmin(pmax(x, 0), 1)
}

cv_to_score <- function(cv) {
  out <- 1 / (1 + as.numeric(cv))
  out[!is.finite(out)] <- NA_real_
  clamp01(out)
}

log_norm_low_better <- function(x) {
  x <- as.numeric(x)
  out <- rep(NA_real_, length(x))
  ok <- is.finite(x) & x >= 0
  if (!any(ok)) return(out)

  lx <- log1p(x[ok])
  mn <- min(lx, na.rm = TRUE)
  mx <- max(lx, na.rm = TRUE)

  if (!is.finite(mn) || !is.finite(mx) || isTRUE(all.equal(mx, mn))) {
    out[ok] <- 0.5
    return(out)
  }

  out[ok] <- 1 - (lx - mn) / (mx - mn)
  clamp01(out)
}

score_to_palette_index <- function(score, n = 100L) {
  s <- clamp01(score)
  if (is.na(s)) return(NA_integer_)
  max(1L, min(n, as.integer(floor(s * (n - 1L)) + 1L)))
}

score_to_radius <- function(score, min_r, max_r) {
  s <- clamp01(score)
  if (is.na(s)) return(NA_real_)
  min_r + (s ^ 0.55) * (max_r - min_r)
}

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
message("-- Loading metrics: ", METRICS_CSV)
raw <- read_csv(METRICS_CSV, show_col_types = FALSE) %>%
  mutate(
    method = trimws(tolower(ti_method)),
    task   = trimws(task)
  ) %>%
  filter(method %in% names(METHOD_COLORS))

message("-- Loading runtime: ", RUNTIME_CSV)
rt_raw <- read_csv(RUNTIME_CSV, show_col_types = FALSE) %>%
  mutate(method = trimws(tolower(ti_method)))

# Build one row per method × task using raw metrics
rows_df <- raw %>%
  select(method, task, all_of(RAW_METRIC_COLS)) %>%
  arrange(method, task)

row_ids <- paste(rows_df$method, rows_df$task, sep = "__")
n_rows  <- nrow(rows_df)
methods_all <- rows_df$method
tasks_all   <- rows_df$task

message("Rows (", n_rows, "): ", paste(row_ids, collapse = " | "))

# -----------------------------------------------------------------------------
# Build plotted score table
# -----------------------------------------------------------------------------
plot_rows <- rows_df %>%
  mutate(
    kendalls_w_score                     = clamp01(kendalls_w),
    topology_jaccard_score              = clamp01(topology_jaccard_mean),
    branch_leaves_score                 = cv_to_score(branch_leaves_cv),
    branch_points_score                 = cv_to_score(branch_points_cv),
    root_consistency_score              = clamp01(root_consistency_total),
    integration_sensitivity_score_score = clamp01(integration_sensitivity_score)
  )

# -----------------------------------------------------------------------------
# Runtime lookup
# -----------------------------------------------------------------------------
get_rt <- function(method, task, col_suffix) {
  task_num <- sub("task_", "", task)
  col <- paste0("task_", task_num, "_", col_suffix)
  if (!col %in% colnames(rt_raw)) return(NA_real_)
  val <- rt_raw[[col]][rt_raw$method == method]
  if (length(val) == 0) NA_real_ else suppressWarnings(as.numeric(val[1]))
}

rt_total <- mapply(get_rt, methods_all, tasks_all, "total")
rt_mean  <- mapply(get_rt, methods_all, tasks_all, "mean")

rt_total_score <- log_norm_low_better(rt_total)
rt_mean_score  <- log_norm_low_better(rt_mean)

# =============================================================================
# BUBBLE PLOT
# =============================================================================
make_ti_bubble_plot <- function(out) {

  n_circ <- length(CIRC_METRICS)

  # Column widths
  W_METHOD <- 0.90
  W_TASK   <- 2.00
  BUB_W    <- 0.28
  BUB_LAST <- 0.40
  W_SENS   <- 0.52
  W_RT_DOT <- 0.32

  col_w <- c(
    W_METHOD, W_TASK,
    rep(BUB_W, n_circ - 1L), BUB_LAST,
    W_SENS,
    W_RT_DOT, W_RT_DOT
  )
  cum_x  <- cumsum(c(0, col_w))
  TW     <- sum(col_w)
  col_cx <- function(j) cum_x[j] + col_w[j] / 2

  n_meta       <- 2L
  j_circ_start <- n_meta + 1L
  j_circ_end   <- j_circ_start + n_circ - 1L
  j_sens       <- j_circ_end + 1L
  j_rt_total   <- j_sens + 1L
  j_rt_mean    <- j_rt_total + 1L

  x_traj_sep  <- cum_x[j_circ_start]
  x_scale_sep <- cum_x[j_rt_total]
  RECT_PAD    <- 0.05

  ROW_H   <- 0.285
  HDR_H   <- 0.20
  LBL_H   <- 1.85
  BAR_W_L <- 0.13
  BAR_SEP <- 0.17
  LEG_GAP <- 0.52
  L_MAR   <- 0.22
  T_MAR   <- 0.50
  B_MAR   <- LBL_H + 0.60
  R_MAR   <- LEG_GAP + 2 * (BAR_W_L + BAR_SEP) + 0.60
  FIG_W   <- L_MAR + TW + R_MAR
  FIG_H   <- T_MAR + HDR_H + n_rows * ROW_H + B_MAR

  pdf(out, width = FIG_W, height = FIG_H, family = BASE_FONT)
  on.exit(dev.off(), add = TRUE)

  par(mar = c(0, 0, 0, 0), bg = "white")
  plot(
    0, 0, type = "n",
    xlim = c(0, FIG_W), ylim = c(0, FIG_H),
    asp = NA, axes = FALSE, xlab = "", ylab = "",
    xaxs = "i", yaxs = "i"
  )

  TX      <- L_MAR
  hdr_bot <- B_MAR + n_rows * ROW_H
  hdr_top <- hdr_bot + HDR_H
  tbl_bot <- B_MAR

  row_top <- function(i) hdr_bot - (i - 1L) * ROW_H
  row_bot <- function(i) hdr_bot - i * ROW_H
  row_cy  <- function(i) (row_top(i) + row_bot(i)) / 2

  MAX_R <- ROW_H * 0.38
  MIN_R <- ROW_H * 0.05

  # Title
  text(
    TX + TW / 2, hdr_top + 0.30,
    "Coupled TI benchmark -- Trajectory sensitivity & Scalability",
    cex = 0.80, font = 1, col = "#111111", adj = c(0.5, 0.5)
  )

  # Alternating row shading
  for (i in seq_len(n_rows)) {
    shade <- if (i %% 2 == 1) "white" else "#E2E2E2"
    rect(TX, row_bot(i), TX + TW, row_top(i), col = shade, border = NA)
  }

  # Borders
  for (y in c(hdr_top, tbl_bot))
    segments(TX, y, TX + TW, y, lwd = 1.6, lend = 1, col = "#111111")
  segments(TX, hdr_bot, TX + TW, hdr_bot, lwd = 0.9, lend = 1, col = "#111111")

  # Vertical dotted separators in body
  for (sx in c(TX + x_traj_sep, TX + x_scale_sep))
    segments(sx, hdr_bot, sx, tbl_bot, lwd = 0.4, lty = 3, col = "#BBBBBB")

  # Header blocks
  rect(TX, hdr_bot, TX + x_traj_sep, hdr_top, col = META_HEADER_COL, border = NA)
  rect(TX + x_traj_sep, hdr_bot, TX + x_scale_sep, hdr_top, col = TRAJ_HEADER_COL, border = NA)
  rect(TX + x_scale_sep, hdr_bot, TX + TW, hdr_top, col = SCALE_HEADER_COL, border = NA)

  for (y in c(hdr_top, hdr_bot))
    segments(TX, y, TX + TW, y, lwd = 1.6, lend = 1, col = "#111111")
  for (sx in c(TX + x_traj_sep, TX + x_scale_sep))
    segments(sx, hdr_bot, sx, hdr_top, lwd = 0.8, col = "#FFFFFF")

  hdr_cy <- (hdr_bot + hdr_top) / 2
  text(TX + cum_x[1] + 0.06, hdr_cy, "Method", cex = 0.68, font = 1, col = "white", adj = c(0, 0.5))
  text(TX + col_cx(2), hdr_cy, "Task", cex = 0.64, font = 1, col = "white", adj = c(0.5, 0.5))

  traj_mid  <- TX + x_traj_sep + (x_scale_sep - x_traj_sep) / 2
  scale_mid <- TX + x_scale_sep + (TW - x_scale_sep) / 2
  text(traj_mid, hdr_cy, "Trajectory Sensitivity", cex = 0.66, font = 1, col = "white", adj = c(0.5, 0.5))
  text(scale_mid, hdr_cy, "Scalability", cex = 0.64, font = 1, col = "white", adj = c(0.5, 0.5))

  # ---- Data rows ----
  for (i in seq_len(n_rows)) {
    m   <- methods_all[i]
    tsk <- tasks_all[i]
    cy  <- row_cy(i)

    text(TX + cum_x[1] + 0.06, cy, m,
         cex = 0.74, font = 1, col = "black", adj = c(0, 0.5))

    task_lbl <- TASK_LABELS[[tsk]]
    if (is.null(task_lbl)) task_lbl <- tsk
    text(TX + col_cx(2), cy, task_lbl,
         cex = 0.74, font = 1, col = "#333333", adj = c(0.5, 0.5))

    # ---- Circle metric dots ----
    for (j in seq_along(CIRC_METRICS)) {
      met <- CIRC_METRICS[j]
      jj  <- j_circ_start + j - 1L
      bcx <- TX + col_cx(jj)

      sv <- plot_rows[[met]][i]
      if (is.na(sv)) {
        text(bcx, cy, "N/A", cex = 0.50, col = "#BBBBBB", adj = c(0.5, 0.5))
        next
      }

      r   <- score_to_radius(sv, MIN_R, MAX_R)
      idx <- score_to_palette_index(sv, length(TRAJ_PAL))
      th  <- seq(0, 2 * pi, length.out = 80)
      polygon(bcx + r * cos(th), cy + r * sin(th), col = TRAJ_PAL[idx], border = "black", lwd = 0.7)
    }

    # ---- Sensitivity rectangle ----
    {
      sv <- plot_rows[[RECT_METRIC]][i]
      col_left <- TX + cum_x[j_sens]
      cw_s <- col_w[j_sens]
      rect_h <- ROW_H * 0.62
      max_w <- cw_s - 2 * RECT_PAD

      if (!is.na(sv)) {
        idx_s  <- score_to_palette_index(sv, length(SENS_PAL))
        rect_w <- RECT_PAD + sv * (max_w - RECT_PAD)
        rect(
          col_left + RECT_PAD, cy - rect_h / 2,
          col_left + RECT_PAD + rect_w, cy + rect_h / 2,
          col = SENS_PAL[idx_s], border = "#555555", lwd = 0.6
        )
      } else {
        text(TX + col_cx(j_sens), cy, "N/A", cex = 0.50, col = "#BBBBBB", adj = c(0.5, 0.5))
      }
    }

    # ---- Runtime dots ----
    draw_rt_dot <- function(j_col, score_val) {
      bcx <- TX + col_cx(j_col)
      if (is.na(score_val)) {
        text(bcx, cy, "N/A", cex = 0.50, col = "#BBBBBB", adj = c(0.5, 0.5))
        return()
      }
      r   <- score_to_radius(score_val, MIN_R, MAX_R)
      idx <- score_to_palette_index(score_val, length(RTPAL))
      th  <- seq(0, 2 * pi, length.out = 80)
      polygon(bcx + r * cos(th), cy + r * sin(th), col = RTPAL[idx], border = "black", lwd = 0.7)
    }
    draw_rt_dot(j_rt_total, rt_total_score[i])
    draw_rt_dot(j_rt_mean,  rt_mean_score[i])
  }

  # ---- Rotated column labels ----
  for (j in seq_along(CIRC_METRICS)) {
    jj  <- j_circ_start + j - 1L
    lbl <- METRIC_LABELS_FLAT[[CIRC_METRICS[j]]]
    text(TX + col_cx(jj), tbl_bot - 0.05, lbl,
         cex = 0.56, col = "#333333", adj = c(1, 0.5), srt = 90)
  }
  text(TX + col_cx(j_sens), tbl_bot - 0.05, "Integration sensitivity",
       cex = 0.56, col = "#333333", adj = c(1, 0.5), srt = 90)
  text(TX + col_cx(j_rt_total), tbl_bot - 0.05, "Total runtime (s)",
       cex = 0.56, col = "#333333", adj = c(1, 0.5), srt = 90)
  text(TX + col_cx(j_rt_mean), tbl_bot - 0.05, "Mean runtime (s)",
       cex = 0.56, col = "#333333", adj = c(1, 0.5), srt = 90)

  # ---- Legends ----
  leg_x0  <- TX + TW + LEG_GAP
  bar_hl  <- n_rows * ROW_H * 0.50
  bar_top <- hdr_bot - (n_rows * ROW_H - bar_hl) / 2
  bar_y0  <- bar_top - bar_hl
  n_seg   <- 80

  draw_cbar <- function(x, y0, ytop, pal) {
    bh <- (ytop - y0) / n_seg
    for (s in seq_len(n_seg)) {
      ic <- ceiling(s / n_seg * length(pal))
      rect(x, y0 + (s - 1) * bh, x + BAR_W_L, y0 + s * bh,
           col = pal[max(1, min(length(pal), ic))], border = NA)
    }
    rect(x, y0, x + BAR_W_L, ytop, col = NA, border = "#AAAAAA", lwd = 0.3)
  }

  x1 <- leg_x0
  draw_cbar(x1, bar_y0, bar_top, TRAJ_PAL)
  text(x1 + BAR_W_L/2, bar_top + 0.06, "Trajectory\nscore",
       cex = 0.42, col = "#333333", adj = c(0.5, 0))
  text(x1 - 0.03, bar_top, "high", cex = 0.36, col = "#666666", adj = c(1, 0.5))
  text(x1 - 0.03, bar_y0, "low",  cex = 0.36, col = "#666666", adj = c(1, 0.5))

  x2 <- x1 + BAR_W_L + BAR_SEP
  draw_cbar(x2, bar_y0, bar_top, RTPAL)
  text(x2 + BAR_W_L/2, bar_top + 0.06, "Runtime\nscore",
       cex = 0.42, col = "#333333", adj = c(0.5, 0))
  text(x2 + BAR_W_L + 0.03, bar_top, "fast", cex = 0.36, col = "#666666", adj = c(0, 0.5))
  text(x2 + BAR_W_L + 0.03, bar_y0, "slow", cex = 0.36, col = "#666666", adj = c(0, 0.5))

  # Circle size legend
  sc_cx <- (x1 + x2 + BAR_W_L) / 2
  pcts  <- c(0, 0.25, 0.50, 0.75, 1.00)
  sc_gap <- MAX_R * 2 + 0.01
  sc_x0  <- sc_cx - (length(pcts) - 1) * sc_gap / 2
  sc_y   <- bar_y0 - MAX_R - 0.22

  text(sc_cx, bar_y0 - 0.10, "Score (circle size)",
       cex = 0.50, col = "#333333", adj = c(0.5, 1))

  for (k in seq_along(pcts)) {
    r_leg <- score_to_radius(pcts[k], MIN_R, MAX_R)
    cx_k  <- sc_x0 + (k - 1L) * sc_gap
    th    <- seq(0, 2 * pi, length.out = 60)
    polygon(cx_k + r_leg * cos(th), sc_y + r_leg * sin(th),
            col = "white", border = "#555555", lwd = 0.5)
  }
  text(sc_x0, sc_y - MAX_R - 0.04, "0%", cex = 0.38, col = "#555555", adj = c(0.5, 1))
  text(sc_x0 + (length(pcts) - 1) * sc_gap, sc_y - MAX_R - 0.04, "100%",
       cex = 0.38, col = "#555555", adj = c(0.5, 1))

  # Captions
  cap1 <- "Trajectory circles: colour and size both reflect the plotted score. Raw [0,1] metrics are used directly; branch CVs are transformed as 1/(1+CV)."
  cap2 <- "Integration sensitivity rectangle: width and colour reflect the raw ISS score. Runtime circles use log-normalised inverse scores, where lighter/larger indicates faster execution."
  text(TX, tbl_bot - LBL_H + 0.08, cap1, cex = 0.54, font = 3, col = "#555555", adj = c(0, 1))
  text(TX, tbl_bot - LBL_H - 0.25, cap2, cex = 0.54, font = 3, col = "#555555", adj = c(0, 1))

  message("  [saved] ", basename(out))
}

# =============================================================================
# RAW SCORE TABLE
# =============================================================================
make_ti_table <- function(out, title_line1, title_line2) {

  col_order <- c(
    "kendalls_w",
    "topology_jaccard_mean",
    "branch_leaves_cv",
    "branch_points_cv",
    "root_consistency_total",
    "integration_sensitivity_score"
  )

  col_display <- c(
    "Kendall's W",
    "Topology Jaccard",
    "Branch leaves CV",
    "Branch points CV",
    "Root consistency",
    "Integration sensitivity"
  )

  n_data <- length(col_order)
  n_m    <- n_rows

  cell <- matrix("--", nrow = n_m, ncol = n_data,
                 dimnames = list(row_ids, col_order))
  for (i in seq_len(n_m)) {
    for (met in col_order) {
      v <- rows_df[[met]][i]
      if (!is.na(v)) cell[row_ids[i], met] <- sprintf("%.4f", round(v, 4))
    }
  }

  best_per <- list()
  for (met in col_order) {
    vals <- rows_df[[met]]
    if (met %in% c("branch_leaves_cv", "branch_points_cv")) {
      best_per[[met]] <- row_ids[which.min(vals)]
    } else {
      best_per[[met]] <- row_ids[which.max(vals)]
    }
  }

  W_RANK   <- 0.28
  W_METHOD <- 0.90
  W_TASK   <- 2.00
  W_DATA   <- 1.00
  col_w    <- c(W_RANK, W_METHOD, W_TASK, rep(W_DATA, n_data))
  cum_x    <- cumsum(c(0, col_w))
  TW       <- sum(col_w)

  ROW_H <- 0.330
  HDR_H <- 0.380
  L_MAR <- 0.50
  R_MAR <- 0.40
  T_MAR <- 1.10
  B_MAR <- 0.55
  FIG_W <- L_MAR + TW + R_MAR
  FIG_H <- T_MAR + HDR_H + n_m * ROW_H + B_MAR

  png(out,
      width = round(FIG_W * PNG_RES),
      height = round(FIG_H * PNG_RES),
      res = PNG_RES,
      bg = "white",
      family = BASE_FONT)
  on.exit(dev.off(), add = TRUE)

  par(mar = c(0, 0, 0, 0), bg = "white")
  plot(0, 0, type = "n",
       xlim = c(0, FIG_W), ylim = c(0, FIG_H),
       asp = NA, axes = FALSE, xlab = "", ylab = "",
       xaxs = "i", yaxs = "i")

  TX <- L_MAR
  TY <- B_MAR + HDR_H + n_m * ROW_H
  hdr_top  <- TY
  hdr_bot  <- TY - HDR_H
  data_top <- hdr_bot
  tbl_bot  <- data_top - n_m * ROW_H
  row_cy   <- function(i) data_top - (i - 0.5) * ROW_H
  hdr_cy   <- (hdr_top + hdr_bot) / 2

  text(FIG_W / 2, TY + 0.72, title_line1,
       cex = 1.00, font = 2, col = "#000000", adj = c(0.5, 0.5))
  text(FIG_W / 2, TY + 0.38, title_line2,
       cex = 0.68, font = 3, col = "#333333", adj = c(0.5, 0.5))

  for (y in c(hdr_top, hdr_bot, tbl_bot))
    segments(TX, y, TX + TW, y, lwd = 1.8, lend = 1, col = "#FFFFFF")

  for (i in seq_len(n_m)) {
    shade <- if (i %% 2 == 0) "#F0F0F0" else "white"
    rect(TX, data_top - i * ROW_H, TX + TW, data_top - (i - 1) * ROW_H, col = shade, border = NA)
  }

  rect(TX, hdr_bot, TX + TW, hdr_top, col = TRAJ_HEADER_COL, border = NA)
  for (y in c(hdr_top, hdr_bot))
    segments(TX, y, TX + TW, y, lwd = 1.8, lend = 1, col = "#FFFFFF")

  text(TX + cum_x[1] + W_RANK - 0.04, hdr_cy, "#", cex = 0.76, font = 1, col = "white", adj = c(1, 0.5))
  text(TX + cum_x[2] + 0.06, hdr_cy, "Method", cex = 0.76, font = 1, col = "white", adj = c(0, 0.5))
  text(TX + cum_x[3] + 0.06, hdr_cy, "Task", cex = 0.70, font = 1, col = "white", adj = c(0, 0.5))
  for (j in seq_len(n_data)) {
    text(TX + cum_x[j + 3] + W_DATA - 0.06, hdr_cy, col_display[j],
         cex = 0.60, font = 1, col = "white", adj = c(1, 0.5))
  }

  for (i in seq_len(n_m)) {
    rid <- row_ids[i]
    cy  <- row_cy(i)
    m   <- methods_all[i]
    tsk <- tasks_all[i]

    text(TX + cum_x[1] + W_RANK - 0.04, cy, as.character(i),
         cex = 0.72, col = "#777777", adj = c(1, 0.5))
    text(TX + cum_x[2] + 0.06, cy, m,
         cex = 0.78, font = 1, col = "black", adj = c(0, 0.5))
    lbl <- TASK_LABELS[[tsk]]
    if (is.null(lbl)) lbl <- tsk
    text(TX + cum_x[3] + 0.06, cy, lbl,
         cex = 0.78, col = "#444444", adj = c(0, 0.5))

    for (j in seq_len(n_data)) {
      cn <- col_order[j]
      val <- cell[rid, cn]
      rx <- TX + cum_x[j + 3] + W_DATA - 0.08
      is_best <- isTRUE(best_per[[cn]] == rid)
      fv <- if (is_best) 2 else 1

      text(rx, cy, val, cex = 0.72, font = fv, col = "#111111", adj = c(1, 0.5))
      if (is_best && val != "--") {
        tw <- nchar(val) * 0.056
        segments(rx - tw * 2, cy - ROW_H * 0.36, rx, cy - ROW_H * 0.36, lwd = 0.8, col = "#000000")
      }
    }
  }

  notes <- "Bold + underline = best value per column. For branch_leaves_cv and branch_points_cv, lower values are better."
  text(TX, tbl_bot - 0.18, notes,
       cex = 0.56, font = 3, col = "#555555", adj = c(0, 1))

  message("  [saved] ", basename(out))
}

# =============================================================================
# RADAR HELPERS
# =============================================================================
draw_ti_radar_labels <- function(n_vars, var_names, label_r = 1.42, cex = 0.72) {
  angles <- seq(90, 90 - 360, length.out = n_vars + 1)[-(n_vars + 1)] * pi / 180
  for (i in seq_len(n_vars)) {
    ang <- angles[i]
    ha  <- if (cos(ang) > 0.15) 0 else if (cos(ang) < -0.15) 1 else 0.5
    va  <- if (sin(ang) > 0.15) 0 else if (sin(ang) < -0.15) 1 else 0.5
    text(label_r * cos(ang), label_r * sin(ang),
         labels = var_names[i], cex = cex, col = "#333333", adj = c(ha, va), font = 1)
  }
}

draw_ti_radar_grid <- function(n_vars, seg = 4) {
  angs <- seq(90, 90 - 360, length.out = n_vars + 1)[-(n_vars + 1)] * pi / 180
  ring_vals <- seq(1 / seg, 1, by = 1 / seg)

  for (rv in ring_vals) {
    px <- rv * cos(angs)
    py <- rv * sin(angs)
    for (k in seq_len(n_vars)) {
      k2 <- if (k == n_vars) 1L else k + 1L
      lines(c(px[k], px[k2]), c(py[k], py[k2]), col = "#CCCCCC", lty = 2, lwd = 1.0)
    }
  }
  for (ang in angs) {
    lines(c(0, cos(ang)), c(0, sin(ang)), col = "#CCCCCC", lwd = 0.5)
  }

  lab_ang <- angs[1] + 0.14
  for (rv in ring_vals) {
    text(rv * cos(lab_ang) + 0.02, rv * sin(lab_ang),
         sprintf("%.0f%%", rv * 100), cex = 0.46, col = "#AAAAAA", adj = c(0, 0.5))
  }
}

make_ti_radar_overlaid <- function(data_wide, metric_cols, out,
                                   title1, title2,
                                   width = 9, height = 10) {

  methods <- intersect(names(METHOD_COLORS), rownames(data_wide))
  nv      <- length(metric_cols)
  var_lbl <- unname(sapply(metric_cols, function(x) {
    lbl <- METRIC_LABELS_FLAT[[x]]
    if (is.null(lbl)) x else lbl
  }))
  angs <- seq(90, 90 - 360, length.out = nv + 1)[-(nv + 1)] * pi / 180

  pdf(out, width = width, height = height, family = BASE_FONT)
  on.exit(dev.off(), add = TRUE)

  par(mar = c(1, 1, 3.4, 1), bg = "white")
  plot(0, 0, type = "n",
       xlim = c(-1.80, 1.80), ylim = c(-1.95, 1.70),
       asp = 1, axes = FALSE, xlab = "", ylab = "",
       xaxs = "i", yaxs = "i")

  draw_ti_radar_grid(nv, seg = 4)
  draw_ti_radar_labels(nv, var_lbl, label_r = 1.44, cex = 0.82)

  # Filled polygons
  for (m in methods) {
    clr  <- METHOD_COLORS[[m]]
    vals <- as.numeric(data_wide[m, metric_cols])
    vals[!is.finite(vals)] <- 0
    vals <- clamp01(vals)

    px <- c(vals * cos(angs), vals[1] * cos(angs[1]))
    py <- c(vals * sin(angs), vals[1] * sin(angs[1]))
    rgb_ <- col2rgb(clr) / 255
    polygon(px, py, col = rgb(rgb_[1], rgb_[2], rgb_[3], alpha = 0.12), border = NA)
  }

  # Lines + points
  for (m in methods) {
    clr  <- METHOD_COLORS[[m]]
    vals <- as.numeric(data_wide[m, metric_cols])
    vals[!is.finite(vals)] <- 0
    vals <- clamp01(vals)

    px <- c(vals * cos(angs), vals[1] * cos(angs[1]))
    py <- c(vals * sin(angs), vals[1] * sin(angs[1]))
    lines(px, py, col = clr, lwd = 2.4, lend = "round", ljoin = "round")
    points(px[-length(px)], py[-length(py)],
           pch = 21, cex = 1.00, bg = "white", col = clr, lwd = 1.6)
  }

  mtext(title1, side = 3, line = 2.0, cex = 1.05, font = 2, col = "#111111")
  mtext(title2, side = 3, line = 0.6, cex = 0.70, font = 3, col = "#555555")

  n_leg_col <- min(4L, ceiling(length(methods) / 2))
  legend(x = 0, y = -1.65, xjust = 0.5, yjust = 0.5, xpd = TRUE,
         legend = methods,
         col = unname(METHOD_COLORS[methods]),
         pt.bg = unname(METHOD_COLORS[methods]),
         pch = 21, pt.cex = 1.15, lwd = 2.2, lty = 1,
         ncol = n_leg_col, bty = "n", cex = 0.80,
         text.col = unname(METHOD_COLORS[methods]),
         x.intersp = 0.75, y.intersp = 1.05)

  message("  [saved] ", basename(out))
}

# -----------------------------------------------------------------------------
# Build radar matrices from TRUE plotted scores, not global min-max normalization
# -----------------------------------------------------------------------------
RADAR_METRICS <- ALL_PLOT_METRICS

build_score_wide <- function(task_filter = NULL) {
  if (is.null(task_filter)) {
    df <- plot_rows %>%
      group_by(method) %>%
      summarise(across(all_of(RADAR_METRICS), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
      column_to_rownames("method")
  } else {
    df <- plot_rows %>%
      filter(task == task_filter) %>%
      select(method, all_of(RADAR_METRICS)) %>%
      column_to_rownames("method")
  }
  df[!is.finite(as.matrix(df))] <- 0
  as.data.frame(df)
}

score_wide_task1 <- build_score_wide("task_1")
score_wide_task2 <- build_score_wide("task_2")
score_wide_mean  <- build_score_wide(NULL)

# =============================================================================
# MAIN
# =============================================================================
message("\n-- TI bubble plot ...")
make_ti_bubble_plot(file.path(OUTPUT_DIR, "ti_bubble_plot.pdf"))

message("\n-- TI score table ...")
make_ti_table(
  out = file.path(OUTPUT_DIR, "ti_score_table.png"),
  title_line1 = "Coupled TI Benchmark -- Trajectory Sensitivity Scores",
  title_line2 = paste0(
    "Raw scores for ", length(unique(methods_all)),
    " TI methods across ", length(unique(tasks_all)),
    " tasks  |  branch CV columns reported in raw form"
  )
)

message("\n-- Radar: Task 1 (Monocyte / Macrophage TAM) ...")
make_ti_radar_overlaid(
  data_wide = score_wide_task1,
  metric_cols = RADAR_METRICS,
  out = file.path(OUTPUT_DIR, "ti_radar_task1.pdf"),
  title1 = "TI benchmark -- Task 1: Monocyte / Macrophage TAM",
  title2 = "Direct scores for bounded metrics; branch CVs mapped as 1/(1+CV)",
  width = 9, height = 10
)

message("\n-- Radar: Task 2 (CD8 T-cell differentiation) ...")
make_ti_radar_overlaid(
  data_wide = score_wide_task2,
  metric_cols = RADAR_METRICS,
  out = file.path(OUTPUT_DIR, "ti_radar_task2.pdf"),
  title1 = "TI benchmark -- Task 2: CD8 T-cell differentiation",
  title2 = "Direct scores for bounded metrics; branch CVs mapped as 1/(1+CV)",
  width = 9, height = 10
)

message("\n-- Radar: Mean across both tasks ...")
make_ti_radar_overlaid(
  data_wide = score_wide_mean,
  metric_cols = RADAR_METRICS,
  out = file.path(OUTPUT_DIR, "ti_radar_mean.pdf"),
  title1 = "TI benchmark -- Mean across both tasks",
  title2 = "Mean plotted scores across tasks",
  width = 9, height = 10
)

message("\n-- Done. All outputs in: ", OUTPUT_DIR, "\n")