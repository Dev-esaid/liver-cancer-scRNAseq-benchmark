#!/usr/bin/env Rscript
# =============================================================================
# make_scalability_plots.R
# =============================================================================
# Single scatter plot:
#   x    = wall-clock runtime (minutes)
#   y    = peak RAM (GB)
#   size = overall performance score (0-1, larger = better)
#   colour + shape = method
#
# A method in the bottom-left with a large symbol is ideal:
#   fast, memory-efficient, AND high-performing.
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
METRICS_CSV  <- "/data1/esraa/Thesis-Project/Results/Integration/Quantitive_metrics_scripts_and_results/Final_results_tables/metrics/methods_x_metrics.csv"
PERF_CSV     <- "/data1/esraa/Thesis-Project/Results/Integration/Quantitive_metrics_scripts_and_results/Final_results_tables/performance/methods_x_performance.csv"
OUTPUT_DIR   <- "/data1/esraa/Thesis-Project/Results/Integration/Quantitive_metrics_scripts_and_results/individual_plots"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# =============================================================================
# Method metadata
# =============================================================================
METHOD_NAME_MAP <- c(
  "bbknn"     = "BBKNN",
  "scvi"      = "scVI",
  "scanorama" = "Scanorama",
  "liger"     = "LIGER",
  "scanvi"    = "scANVI",
  "combat"    = "ComBat",
  "scgen"     = "scGen",
  "harmony"   = "Harmony",
  "seurat"    = "Seurat v5",
  "fastmnn"   = "fastMNN",
  "mnn"       = "MNN"
)

METHOD_COLORS <- c(
  "BBKNN"     = "#E63946",
  "scVI"      = "#FF9F1C",
  "Scanorama" = "#2DC653",
  "LIGER"     = "#0096C7",
  "scANVI"    = "#9B2FE0",
  "ComBat"    = "#F72585",
  "scGen"     = "#3A86FF",
  "Harmony"   = "#FB5607",
  "Seurat v5" = "#06D6A0",
  "fastMNN"   = "#FFBE0B",
  "MNN"       = "#8338EC"
)

METHOD_SHAPES <- c(
  "BBKNN"     = 21,
  "scVI"      = 22,
  "Scanorama" = 24,
  "LIGER"     = 23,
  "scANVI"    = 25,
  "ComBat"    = 21,
  "scGen"     = 22,
  "Harmony"   = 24,
  "Seurat v5" = 23,
  "fastMNN"   = 25,
  "MNN"       = 21
)

# =============================================================================
# Metric definitions
# =============================================================================
BIO_METRICS <- c(
"NMI","ARI","cell_type_ASW","isolated_label_F1","isolated_label_ASW","cell_cycle_conservation",
"hvg_conservation","cLISI"
)

BATCH_METRICS <- c(
  "kBET", "iLISI", "batch_ASW",
  "pc_regression_r2_mean", "pcr_comparison"
)

ALL_METRICS <- c(BIO_METRICS, BATCH_METRICS)

METRIC_DIRECTION <- c(
  "NMI"            = TRUE,
  "ARI"            = TRUE,
  "cLISI"           = TRUE,
  "cell_type_ASW"         = TRUE,
  "cell_cycle_conservation"            = TRUE,
  "isolated_label_ASW"    = TRUE,
  "isolated_label_F1"     = TRUE,
  "hvg_conservation"  = TRUE,
  "kBET"                  = TRUE,
  "iLISI"           = TRUE,
  "batch_ASW"             = FALSE,  # inverted
  "pc_regression_r2_mean" = FALSE,  # inverted
  "pcr_comparison"    = TRUE
)

# =============================================================================
# Helpers
# =============================================================================
clean_raw_method <- function(x) {
  x <- trimws(x)                                    # fix MNN leading whitespace
  x <- sub("_parallel\\d+_hvg_seed\\d+$", "", x)   # parallel naming variant
  x <- sub("_full_hvg_seed\\d+$",         "", x)
  x <- sub("_hvg_seed\\d+$",              "", x)
  x <- sub("_seed\\d+$",                  "", x)
  x <- sub("_full$",                      "", x)
  x <- trimws(tolower(x))
  mapped <- METHOD_NAME_MAP[x]
  ifelse(is.na(mapped), NA_character_, mapped)
}

normalize_metric <- function(vals, higher_is_better = TRUE) {
  if (higher_is_better) {
    mn <- min(vals, na.rm = TRUE); mx <- max(vals, na.rm = TRUE)
    if (!is.finite(mn) || !is.finite(mx) || mx == mn)
      return(rep(0.5, length(vals)))
    (vals - mn) / (mx - mn)
  } else {
    av <- abs(vals)
    mn <- min(av, na.rm = TRUE); mx <- max(av, na.rm = TRUE)
    if (!is.finite(mn) || !is.finite(mx) || mx == mn)
      return(rep(0.5, length(vals)))
    1 - (av - mn) / (mx - mn)
  }
}

normalize_rank <- function(vals) {
  r <- rank(vals, ties.method = "average", na.last = "keep")
  r / sum(!is.na(vals))
}

# =============================================================================
# Load & process
# =============================================================================
message("-- Loading metrics: ", METRICS_CSV)
metrics_raw <- read_csv(METRICS_CSV, show_col_types = FALSE) %>%
  mutate(
    method       = trimws(method),
    method_clean = sapply(method, clean_raw_method)
  ) %>%
  filter(!is.na(method_clean))

message("-- Loading performance: ", PERF_CSV)
perf_raw <- read_csv(PERF_CSV, show_col_types = FALSE) %>%
  mutate(
    method       = trimws(method),
    method_clean = sapply(method, clean_raw_method)
  ) %>%
  filter(!is.na(method_clean))

methods_present <- intersect(metrics_raw$method_clean, names(METHOD_COLORS))
all_avail       <- intersect(ALL_METRICS, colnames(metrics_raw))

message("Methods loaded (", length(methods_present), "/11): ",
        paste(sort(methods_present), collapse = ", "))
message("Metrics available (", length(all_avail), "/14): ",
        paste(all_avail, collapse = ", "))

missing_methods <- setdiff(names(METHOD_COLORS), methods_present)
if (length(missing_methods) > 0)
  warning("Missing methods: ", paste(missing_methods, collapse = ", "))

missing_metrics <- setdiff(ALL_METRICS, all_avail)
if (length(missing_metrics) > 0)
  warning("Missing metrics: ", paste(missing_metrics, collapse = ", "))

# Normalise metrics
met_df <- metrics_raw %>%
  filter(method_clean %in% methods_present) %>%
  select(method = method_clean, all_of(all_avail))

norm_df <- met_df
for (met in all_avail) {
  dir <- METRIC_DIRECTION[[met]]; if (is.null(dir)) dir <- TRUE
  norm_df[[met]] <- if (met == "calinski_harabasz") normalize_rank(met_df[[met]])
                    else normalize_metric(met_df[[met]], higher_is_better = dir)
}

# Overall score = mean normalised score across all 14 metrics
score_df <- norm_df %>%
  rowwise() %>%
  mutate(overall = mean(c_across(all_of(all_avail)), na.rm = TRUE)) %>%
  ungroup() %>%
  select(method, overall)

# Merge with performance data
perf_df <- perf_raw %>%
  filter(method_clean %in% methods_present) %>%
  select(method = method_clean, perf_total_time_s, perf_peak_rss_gb)

plot_df <- score_df %>%
  left_join(perf_df, by = "method") %>%
  filter(!is.na(perf_total_time_s), !is.na(perf_peak_rss_gb)) %>%
  mutate(
    runtime_min = perf_total_time_s / 60,
    method      = factor(method, levels = names(METHOD_COLORS))
  )

message("Methods in plot (", nrow(plot_df), "): ",
        paste(sort(as.character(plot_df$method)), collapse = ", "))
message("Score range: [",
        round(min(plot_df$overall, na.rm = TRUE), 3), ", ",
        round(max(plot_df$overall, na.rm = TRUE), 3), "]")

# =============================================================================
# Theme
# =============================================================================
theme_benchmark <- function() {
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
    panel.grid.major = element_line(colour = "#EEEEEE", linewidth = 0.4),
    panel.grid.minor = element_blank(),
    legend.position  = "right",
    legend.title     = element_text(face = "bold", size = 10),
    legend.text      = element_text(size = 9),
    legend.key.size  = unit(1.0, "lines"),
    plot.background  = element_rect(fill = "white", colour = NA),
    panel.background = element_rect(fill = "white", colour = NA),
    plot.margin      = margin(16, 16, 16, 16),
    plot.caption     = element_text(size = 8, colour = "#777777",
                                    hjust = 0, margin = margin(t = 8))
  )
}

# =============================================================================
# Build plot
# =============================================================================
message("\n-- Building scalability scatter plot ...")

p <- ggplot(plot_df,
            aes(x     = runtime_min,
                y     = perf_peak_rss_gb,
                color = method,
                fill  = method,
                shape = method,
                label = method)) +

  geom_point(size = 2.8, stroke = 0.9, alpha = 0.95) +

  geom_text_repel(
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
                     breaks = names(METHOD_COLORS), drop = FALSE,
                     guide = "none") +
  scale_shape_manual(name = "Method", values = METHOD_SHAPES,
                     breaks = names(METHOD_SHAPES), drop = FALSE,
                     guide = "none") +
  scale_x_continuous(
    name   = "Wall-clock runtime (minutes)",
    expand = expansion(mult = c(0.05, 0.10))
  ) +
  scale_y_continuous(
    name   = "Peak memory usage (GB)",
    expand = expansion(mult = c(0.05, 0.10))
  ) +

  labs(
    title    = "Scalability: Runtime vs. Memory",
    subtitle = paste0(
      "Symbol colour & shape = method  |  11 methods  |  ",
      "Bottom-left = best trade-off (fast & low memory)"
    )
  ) +

  # Single guide call — ggplot2 merges color/fill/shape sharing the same name
guides(
    color = guide_legend(
      override.aes = list(
        size  = 4.5,
        shape = unname(METHOD_SHAPES[names(METHOD_COLORS)]),
        fill  = unname(METHOD_COLORS[names(METHOD_COLORS)])
      ),
      order = 1
    )
  ) +
  theme_benchmark()

# =============================================================================
# Save
# =============================================================================
out_stem <- file.path(OUTPUT_DIR, "scalability_runtime_vs_memory")

ggsave(paste0(out_stem, ".pdf"),
       plot = p, width = 9, height = 6.5, device = cairo_pdf)
message("  [saved] scalability_runtime_vs_memory.pdf")

ggsave(paste0(out_stem, ".png"),
       plot = p, width = 9, height = 6.5, dpi = 200, bg = "white")
message("  [saved] scalability_runtime_vs_memory.png")

message("\n-- Done. Output in: ", OUTPUT_DIR, "\n")