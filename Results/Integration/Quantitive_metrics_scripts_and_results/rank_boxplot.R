#!/usr/bin/env Rscript
# =============================================================================
# make_metric_rank_boxplot.R
# =============================================================================
# Box plot of per-metric ranks across all 14 metrics for each method.
# Each box = one method; y-axis = rank (1 = best) across all metrics.
# Methods sorted by median rank (best = lowest median on left).
# Each method box uses the sharp palette from the other plots.
# Individual metric points shown as jittered dots inside boxes.
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
})

# =============================================================================
# Paths
# =============================================================================
METRICS_CSV <- "/data1/esraa/Thesis-Project/Results/Integration/Quantitive_metrics_scripts_and_results/Final_results_tables/metrics/methods_x_metrics.csv"
OUTPUT_DIR  <- "/data1/esraa/Thesis-Project/Results/Integration/Quantitive_metrics_scripts_and_results/individual_plots"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# =============================================================================
# Method metadata
# =============================================================================
METHOD_NAME_MAP <- c(
  "bbknn"="BBKNN","scvi"="scVI","scanorama"="Scanorama","liger"="LIGER",
  "scanvi"="scANVI","combat"="ComBat","scgen"="scGen","harmony"="Harmony",
  "seurat"="Seurat v5","fastmnn"="fastMNN","mnn"="MNN"
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
  "MNN"       = "#8338EC"
)

# =============================================================================
# Metric definitions
# =============================================================================
BIO_METRICS <- c("NMI_leiden","ARI_leiden","cLISI_label","cell_type_ASW",
                 "silhouette","isolated_label_ASW","isolated_label_F1",
                 "rare_knn_purity_mean","calinski_harabasz")
BATCH_METRICS <- c("kBET","iLISI_batch","batch_ASW",
                   "pc_regression_r2_mean","graph_connectivity")
ALL_METRICS <- c(BIO_METRICS, BATCH_METRICS)

# TRUE = higher raw value is better; FALSE = lower absolute value is better
METRIC_DIRECTION <- c(
  "NMI_leiden"=TRUE,"ARI_leiden"=TRUE,"cLISI_label"=TRUE,
  "cell_type_ASW"=TRUE,"silhouette"=TRUE,"isolated_label_ASW"=TRUE,
  "isolated_label_F1"=TRUE,"rare_knn_purity_mean"=TRUE,"calinski_harabasz"=TRUE,
  "kBET"=TRUE,"iLISI_batch"=TRUE,
  "batch_ASW"=FALSE,"pc_regression_r2_mean"=FALSE,"graph_connectivity"=TRUE
)

# Clean display labels for each metric
METRIC_LABELS <- c(
  "NMI_leiden"="NMI (leiden)","ARI_leiden"="ARI (leiden)",
  "cLISI_label"="cLISI (label)","cell_type_ASW"="Cell type ASW",
  "silhouette"="Silhouette ASW","isolated_label_ASW"="Isol. label ASW",
  "isolated_label_F1"="Isol. label F1","rare_knn_purity_mean"="Rare kNN purity",
  "calinski_harabasz"="Calinski-Harabasz",
  "kBET"="kBET","iLISI_batch"="iLISI (batch)",
  "batch_ASW"="Batch ASW","pc_regression_r2_mean"="PC regression R2",
  "graph_connectivity"="Graph connect."
)

# =============================================================================
# Helpers
# =============================================================================
clean_raw_method <- function(x) {
  x <- sub("_full_hvg_seed\\d+$","",x); x <- sub("_hvg_seed\\d+$","",x)
  x <- sub("_seed\\d+$","",x);          x <- sub("_full$","",x)
  x <- trimws(tolower(x))
  mapped <- METHOD_NAME_MAP[x]
  ifelse(is.na(mapped), NA_character_, mapped)
}

# =============================================================================
# Load data
# =============================================================================
message("-- Loading metrics: ", METRICS_CSV)
metrics_raw <- read_csv(METRICS_CSV, show_col_types=FALSE) %>%
  mutate(method_clean = sapply(method, clean_raw_method)) %>%
  filter(!is.na(method_clean))

methods_present <- intersect(metrics_raw$method_clean, names(METHOD_COLORS))
all_avail       <- intersect(ALL_METRICS, colnames(metrics_raw))

met_df <- metrics_raw %>%
  filter(method_clean %in% methods_present) %>%
  select(method = method_clean, all_of(all_avail))

# =============================================================================
# Compute per-metric ranks (rank 1 = best for each metric)
# For "lower is better" metrics, rank in ascending order of abs(value)
# =============================================================================
rank_df <- met_df
for (met in all_avail) {
  dir <- METRIC_DIRECTION[[met]]; if (is.null(dir)) dir <- TRUE
  vals <- met_df[[met]]
  if (dir) {
    # higher raw = better → rank descending (highest gets rank 1)
    rank_df[[met]] <- rank(-vals, ties.method = "average", na.last = "keep")
  } else {
    # lower absolute = better → rank ascending of abs(val)
    rank_df[[met]] <- rank(abs(vals), ties.method = "average", na.last = "keep")
  }
}

# Pivot to long format: one row per (method × metric)
rank_long <- rank_df %>%
  pivot_longer(cols = all_of(all_avail),
               names_to  = "metric",
               values_to = "rank_val") %>%
  filter(!is.na(rank_val)) %>%
  mutate(
    metric_label = METRIC_LABELS[metric],
    # Bio vs batch category for colouring the strip / x-axis later
    category = ifelse(metric %in% BIO_METRICS, "Bio conservation", "Batch correction")
  )

# Sort methods by median rank (lowest = best = leftmost)
method_order <- rank_long %>%
  group_by(method) %>%
  summarise(med = median(rank_val, na.rm = TRUE), .groups = "drop") %>%
  arrange(med) %>%
  pull(method)

rank_long <- rank_long %>%
  mutate(method = factor(method, levels = method_order))

n_methods <- length(method_order)
message("Method order (best → worst): ", paste(method_order, collapse = " > "))

# =============================================================================
# Theme
# =============================================================================
theme_rankbox <- function() {
  theme_classic(base_size = 13, base_family = "Helvetica") +
  theme(
    plot.title       = element_text(face = "bold", size = 15, hjust = 0.5,
                                    margin = margin(b = 6)),
    plot.subtitle    = element_text(size = 9.5, hjust = 0.5, colour = "#555555",
                                    margin = margin(b = 14)),
    axis.title.x     = element_text(size = 12, margin = margin(t = 10)),
    axis.title.y     = element_text(size = 12, margin = margin(r = 8)),
    axis.text.x      = element_text(size = 10, colour = "#333333",
                                    angle = 35, hjust = 1, vjust = 1),
    axis.text.y      = element_text(size = 10, colour = "#333333"),
    axis.line        = element_line(colour = "#444444", linewidth = 0.5),
    axis.ticks       = element_line(colour = "#888888"),
    panel.grid.major.y = element_line(colour = "#F0F0F0", linewidth = 0.4),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    legend.position  = "none",
    plot.background  = element_rect(fill = "white", colour = NA),
    panel.background = element_rect(fill = "white", colour = NA),
    plot.margin      = margin(16, 16, 24, 16)
  )
}

# =============================================================================
# Build plot
# =============================================================================
message("\n-- Building metric rank box plot ...")

# Fill colours: match METHOD_COLORS, lighter alpha for box fill
fill_colors <- setNames(
  paste0(METHOD_COLORS, "55"),   # add 33% alpha hex
  names(METHOD_COLORS)
)

p <- ggplot(rank_long,
            aes(x    = method,
                y    = rank_val,
                fill = method,
                colour = method)) +

  # Box
  geom_boxplot(
    outlier.shape  = NA,     # hide outliers — shown via jitter below
    width          = 0.65,
    linewidth      = 0.6,
    alpha          = 0.45
  ) +

  # Individual metric points (jittered)
  geom_jitter(
    width  = 0.18,
    height = 0,
    size   = 1.8,
    alpha  = 0.80,
    shape  = 16
  ) +

  scale_fill_manual(  values = METHOD_COLORS, breaks = names(METHOD_COLORS)) +
  scale_colour_manual(values = METHOD_COLORS, breaks = names(METHOD_COLORS)) +

  scale_y_continuous(
    name   = "Metric rank  (1 = best)",
    breaks = seq(1, n_methods, by = 1),
    limits = c(0.5, n_methods + 0.5),
    expand = expansion(mult = c(0.01, 0.03))
  ) +

  scale_x_discrete(name = "Method") +

  labs(
    title    = "Per-metric rank distribution across integration methods",
    subtitle = paste0(
      "Each point = one metric  |  Box = IQR + median  |  ",
      "Rank 1 = best  |  Methods sorted by median rank (best left)"
    )
  ) +

  theme_rankbox()

# =============================================================================
# Save
# =============================================================================
out_stem <- file.path(OUTPUT_DIR, "metric_rank_boxplot")

ggsave(paste0(out_stem, ".pdf"),
       plot = p, width = 10, height = 6.5, device = cairo_pdf)
message("  [saved] metric_rank_boxplot.pdf")

ggsave(paste0(out_stem, ".png"),
       plot = p, width = 10, height = 6.5, dpi = 200, bg = "white")
message("  [saved] metric_rank_boxplot.png")

message("\n-- Done. Output in: ", OUTPUT_DIR, "\n")