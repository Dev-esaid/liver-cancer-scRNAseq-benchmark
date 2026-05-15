#!/usr/bin/env Rscript
# =============================================================================
# make_bio_vs_batch_plot.R
# =============================================================================
# Scatter plot:
#   x    = mean batch correction score (normalised, 0-1)
#   y    = mean bio conservation score (normalised, 0-1)
#   error bars = 0.5 * SD of individual normalised metric scores within each category
#   colour + shape = method
#
# Top-right corner = best (high bio AND high batch correction)
# 14 metrics | 11 methods
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(ggrepel)
})

# =============================================================================
# Paths
# =============================================================================
METRICS_CSV <- "/data1/esraa/Thesis-Project/Results/Integration/Quantitive_metrics_scripts_and_results/Final_results_tables/metrics/methods_x_metrics.csv"
OUTPUT_DIR   <- "/data1/esraa/Thesis-Project/Results/Integration/Quantitive_metrics_scripts_and_results/individual_plots"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# =============================================================================
# Method metadata — all 11 methods
# =============================================================================
METHOD_NAME_MAP <- c(
  "bbknn"    = "BBKNN",
  "scvi"     = "scVI",
  "scanorama"= "Scanorama",
  "liger"    = "LIGER",
  "scanvi"   = "scANVI",
  "combat"   = "ComBat",
  "scgen"    = "scGen",
  "harmony"  = "Harmony",
  "seurat"   = "Seurat v5",
  "fastmnn"  = "fastMNN",
  "mnn"      = "MNN"
)

METHOD_COLORS <- c(
  "BBKNN"     = "#D62828",
  "scVI"      = "#023E8A",
  "Scanorama" = "#007200",
  "LIGER"     = "#E85D04",
  "scANVI"    = "#6A0572",
  "ComBat"    = "#0077B6",
  "scGen"     = "#BB3E03",
  "Harmony"   = "#386641",
  "Seurat v5" = "#7B2D8B",
  "fastMNN"   = "#C77DFF",
  "MNN"       = "#00B4D8"
)

METHOD_SHAPES <- setNames(
  rep(21, 11),
  c("BBKNN","scVI","Scanorama","LIGER","scANVI",
    "ComBat","scGen","Harmony","Seurat v5","fastMNN","MNN")
)


# Metric definitions — all 13 metrics

# =========================
# BIOLOGICAL METRICS
# =========================
BIO_METRICS <- c(
  "NMI",
  "ARI",
  "cLISI",
  "cell_type_ASW",
  "cell_cycle_conservation",
  "isolated_label_ASW",
  "isolated_label_F1",
  "hvg_conservation"
)

# =========================
# BATCH METRICS
# =========================
BATCH_METRICS <- c( 
  "kBET",
  "iLISI",
  "batch_ASW",
  "pcr_comparison",
  "graph_connectivity"
)

# =========================
# METRIC DIRECTION
# =========================
# TRUE  = higher is better
# FALSE = lower is better (will be inverted)
METRIC_DIRECTION <- c(
  "NMI"                      = TRUE,
  "ARI"                      = TRUE,
  "cLISI"                    = TRUE,
  "cell_type_ASW"            = TRUE,
  "cell_cycle_conservation"  = TRUE,
  "isolated_label_ASW"       = TRUE,
  "isolated_label_F1"        = TRUE,
  "hvg_conservation"         = TRUE,
  "kBET"                     = TRUE,
  "iLISI"                    = TRUE,
  "batch_ASW"                = FALSE,  # lower = better
  "pcr_comparison"           = FALSE,  # lower = better
  "graph_connectivity"       = TRUE
)


# Helpers


# Strips all known suffix patterns and leading/trailing whitespace,
# then maps to display name. Returns NA for unrecognised methods.
clean_raw_method <- function(x) {
  x <- trimws(x)                              # catches MNN leading whitespace
  x <- sub("_parallel\\d+_hvg_seed\\d+$", "", x)
  x <- sub("_full_hvg_seed\\d+$",         "", x)
  x <- sub("_hvg_seed\\d+$",              "", x)
  x <- sub("_seed\\d+$",                  "", x)
  x <- sub("_full$",                      "", x)
  x <- trimws(tolower(x))
  mapped <- METHOD_NAME_MAP[x]
  ifelse(is.na(mapped), NA_character_, mapped)
}

# Min-max normalisation; direction-aware
normalize_metric <- function(vals, higher_is_better = TRUE) {
  if (higher_is_better) {
    mn <- min(vals, na.rm = TRUE)
    mx <- max(vals, na.rm = TRUE)
    if (!is.finite(mn) || !is.finite(mx) || mx == mn)
      return(rep(0.5, length(vals)))
    (vals - mn) / (mx - mn)
  } else {
    av <- abs(vals)
    mn <- min(av, na.rm = TRUE)
    mx <- max(av, na.rm = TRUE)
    if (!is.finite(mn) || !is.finite(mx) || mx == mn)
      return(rep(0.5, length(vals)))
    1 - (av - mn) / (mx - mn)
  }
}

# Rank normalisation for unbounded metrics (Calinski-Harabasz)
normalize_rank <- function(vals) {
  r <- rank(vals, ties.method = "average", na.last = "keep")
  r / sum(!is.na(vals))
}


# Load data

message("-- Loading metrics: ", METRICS_CSV)

metrics_raw <- read_csv(METRICS_CSV, show_col_types = FALSE) %>%
  mutate(
    method       = trimws(method),           # strip whitespace before cleaning
    method_clean = sapply(method, clean_raw_method)
  ) %>%
  filter(!is.na(method_clean))

# Report which methods were loaded
message("Methods loaded: ",
        paste(sort(unique(metrics_raw$method_clean)), collapse = ", "))
message("N methods: ", n_distinct(metrics_raw$method_clean))

# Verify all 11 expected methods are present
expected_methods <- names(METHOD_COLORS)
missing_methods  <- setdiff(expected_methods, metrics_raw$method_clean)
if (length(missing_methods) > 0) {
  warning("Missing methods: ", paste(missing_methods, collapse = ", "))
}


# Select and verify all 14 metrics

all_metrics  <- c(BIO_METRICS, BATCH_METRICS)
all_avail    <- intersect(all_metrics, colnames(metrics_raw))
bio_avail    <- intersect(BIO_METRICS,   all_avail)
batch_avail  <- intersect(BATCH_METRICS, all_avail)

message("Bio metrics available (", length(bio_avail), "/9): ",
        paste(bio_avail, collapse = ", "))
message("Batch metrics available (", length(batch_avail), "/5): ",
        paste(batch_avail, collapse = ", "))

missing_metrics <- setdiff(all_metrics, all_avail)
if (length(missing_metrics) > 0) {
  warning("Missing metrics: ", paste(missing_metrics, collapse = ", "))
}


# Filter and normalise

met_df <- metrics_raw %>%
  filter(method_clean %in% expected_methods) %>%
  select(method = method_clean, all_of(all_avail))

norm_df <- met_df
for (met in all_avail) {
  dir <- METRIC_DIRECTION[[met]]
  if (is.null(dir)) dir <- TRUE
  norm_df[[met]] <- if (met == "calinski_harabasz") {
    normalize_rank(met_df[[met]])
  } else {
    normalize_metric(met_df[[met]], higher_is_better = dir)
  }
}


# Compute per-method bio/batch mean +/- SD

plot_df <- norm_df %>%
  rowwise() %>%
  mutate(
    bio_mean   = mean(c_across(all_of(bio_avail)),   na.rm = TRUE),
    bio_sd     = sd(  c_across(all_of(bio_avail)),   na.rm = TRUE),
    batch_mean = mean(c_across(all_of(batch_avail)), na.rm = TRUE),
    batch_sd   = sd(  c_across(all_of(batch_avail)), na.rm = TRUE)
  ) %>%
  ungroup() %>%
  select(method, bio_mean, bio_sd, batch_mean, batch_sd) %>%
  mutate(method = factor(method, levels = names(METHOD_COLORS)))

message("\n-- Per-method summary (sorted by bio_mean):")
print(plot_df %>% arrange(desc(bio_mean)), n = Inf)


# Theme
# =============================================================================
theme_bio_batch <- function() {
  theme_classic(base_size = 13, base_family = "Helvetica") +
  theme(
    plot.title       = element_text(face = "bold", size = 15, hjust = 0.5,
                                    margin = margin(b = 6)),
    plot.subtitle    = element_text(size = 9.5, hjust = 0.5, colour = "#555555",
                                    margin = margin(b = 14)),
    axis.title       = element_text(size = 12),
    axis.text        = element_text(size = 10, colour = "#333333"),
    axis.line        = element_line(colour = "#444444", linewidth = 0.5),
    axis.ticks       = element_line(colour = "#888888"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position  = "right",
    legend.title     = element_text(face = "bold", size = 10),
    legend.text      = element_text(size = 9),
    legend.key.size  = unit(1.0, "lines"),
    plot.background  = element_rect(fill = "white", colour = NA),
    panel.background = element_rect(fill = "white", colour = NA),
    plot.margin      = margin(16, 16, 16, 16)
  )
}

# =============================================================================
# Build plot
# =============================================================================
message("\n-- Building bio vs batch scatter plot ...")

p <- ggplot(plot_df,
            aes(x     = batch_mean,
                y     = bio_mean,
                color = method,
                fill  = method,
                shape = method)) +

  # Error bars (behind dots); SD halved for visual clarity
  geom_errorbar(
    aes(ymin = pmax(bio_mean   - bio_sd   * 0.5, 0),
        ymax = pmin(bio_mean   + bio_sd   * 0.5, 1)),
    width     = 0,
    linewidth = 0.55,
    alpha     = 0.70
  ) +
  geom_errorbarh(
    aes(xmin = pmax(batch_mean - batch_sd * 0.5, 0),
        xmax = pmin(batch_mean + batch_sd * 0.5, 1)),
    height    = 0,
    linewidth = 0.55,
    alpha     = 0.70
  ) +

  # Points
  geom_point(size = 4.0, stroke = 1.0, alpha = 0.95) +

  # Labels
  geom_text_repel(
    aes(label = method),
    size          = 3.1,
    fontface      = "bold",
    box.padding   = 0.4,
    point.padding = 0.3,
    force         = 2,
    segment.color = "#CCCCCC",
    segment.size  = 0.3,
    max.overlaps  = Inf,
    show.legend   = FALSE
  ) +

  scale_color_manual(name = "Method", values = METHOD_COLORS,
                     breaks = names(METHOD_COLORS), drop = FALSE) +
  scale_fill_manual( name = "Method", values = METHOD_COLORS,
                     breaks = names(METHOD_COLORS), drop = FALSE) +
  scale_shape_manual(name = "Method", values = METHOD_SHAPES,
                     breaks = names(METHOD_SHAPES), drop = FALSE) +

  scale_x_continuous(
    name   = "Batch correction score",
    limits = c(0, 1),
    breaks = seq(0, 1, 0.25),
    expand = expansion(mult = c(0.02, 0.02))
  ) +
  scale_y_continuous(
    name   = "Bio-conservation score",
    limits = c(0, 1),
    breaks = seq(0, 1, 0.25),
    expand = expansion(mult = c(0.02, 0.02))
  ) +

  annotate("text", x = 0.97, y = 0.97, label = "best",
           colour = "#AAAAAA", size = 3.2, fontface = "italic",
           hjust = 1, vjust = 1) +
  annotate("text", x = 0.03, y = 0.03, label = "worst",
           colour = "#AAAAAA", size = 3.2, fontface = "italic",
           hjust = 0, vjust = 0) +

  labs(
    title    = "Bio-conservation vs. Batch correction",
    subtitle = paste0(
      "Each point = one method  |  11 methods  |  14 metrics ",
      "(9 bio + 5 batch)\n",
      "Error bars = \u00b10.5 SD across normalised metric scores  |  ",
      "Top-right = best trade-off"
    )
  ) +

  guides(
    color = guide_legend(override.aes = list(size = 4.5), order = 1),
    fill  = guide_legend(override.aes = list(size = 4.5), order = 1),
    shape = guide_legend(override.aes = list(size = 4.5), order = 1)
  ) +

  theme_bio_batch()

# =============================================================================
# Save
# =============================================================================
out_stem <- file.path(OUTPUT_DIR, "bio_vs_batch")

ggsave(paste0(out_stem, ".pdf"),
       plot = p, width = 9, height = 7.5, device = cairo_pdf)
message("  [saved] bio_vs_batch.pdf")

ggsave(paste0(out_stem, ".png"),
       plot = p, width = 9, height = 7.5, dpi = 200, bg = "white")
message("  [saved] bio_vs_batch.png")

message("\n-- Done. Output in: ", OUTPUT_DIR, "\n")