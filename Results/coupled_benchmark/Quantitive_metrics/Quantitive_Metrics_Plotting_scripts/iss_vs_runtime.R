#!/usr/bin/env Rscript
# =============================================================================
# make_iss_vs_runtime_plot.R
# =============================================================================
# Scatter plot for coupled benchmark:
#   x    = total runtime across both tasks (seconds)
#   y    = mean integration sensitivity score across both tasks
#   y-error bars = SD of integration sensitivity score across task_1 and task_2
#   colour + shape = TI method
#
# Top-left  = best trade-off (high ISS, low runtime)
# Top-right = high ISS but expensive
# Bottom-left = fast but weak
# Bottom-right = slow and weak
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(ggrepel)
})

# =============================================================================
# Paths
# =============================================================================
METRICS_CSV <- "/data1/esraa/Thesis-Project/Results/coupled_benchmark/Quantitive_metrics/coupled_metrics_master.csv"
RUNTIME_CSV <- "/data1/esraa/Thesis-Project/Results/coupled_benchmark/Quantitive_metrics/coupled_ti_runtime.csv"
OUTPUT_DIR  <- "/data1/esraa/Thesis-Project/Results/coupled_benchmark/Quantitive_metrics/figures"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# =============================================================================
# Method metadata
# =============================================================================
METHOD_NAME_MAP <- c(
  "cellrank"  = "CellRank",
  "monocle3"  = "Monocle3",
  "slingshot" = "Slingshot",
  "tscan"     = "TSCAN"
)

METHOD_COLORS <- c(
  "CellRank"  = "#4E79A7",
  "Monocle3"  = "#F28E2B",
  "Slingshot" = "#59A14F",
  "TSCAN"     = "#E15759"
)

METHOD_SHAPES <- c(
  "CellRank"  = 21,
  "Monocle3"  = 22,
  "Slingshot" = 23,
  "TSCAN"     = 24
)

# =============================================================================
# Helpers
# =============================================================================
clean_method <- function(x) {
  x <- trimws(tolower(as.character(x)))
  mapped <- METHOD_NAME_MAP[x]
  ifelse(is.na(mapped), x, mapped)
}

theme_iss_runtime <- function() {
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
# Load coupled metrics
# =============================================================================
message("-- Loading metrics: ", METRICS_CSV)

metrics_df <- read_csv(METRICS_CSV, show_col_types = FALSE) %>%
  mutate(
    ti_method = clean_method(ti_method),
    task      = trimws(task)
  ) %>%
  filter(ti_method %in% names(METHOD_COLORS))

message("Methods in metrics: ", paste(sort(unique(metrics_df$ti_method)), collapse = ", "))

# Expect one row per method x task
iss_summary <- metrics_df %>%
  group_by(ti_method) %>%
  summarise(
    iss_mean = mean(integration_sensitivity_score, na.rm = TRUE),
    iss_sd   = sd(integration_sensitivity_score, na.rm = TRUE),
    n_tasks  = sum(!is.na(integration_sensitivity_score)),
    .groups  = "drop"
  )

# =============================================================================
# Load runtime table
# =============================================================================
message("-- Loading runtime: ", RUNTIME_CSV)

runtime_df <- read_csv(RUNTIME_CSV, show_col_types = FALSE) %>%
  mutate(
    ti_method = clean_method(ti_method)
  ) %>%
  filter(ti_method %in% names(METHOD_COLORS))

message("Methods in runtime: ", paste(sort(unique(runtime_df$ti_method)), collapse = ", "))

# Expected columns from your runtime aggregation script:
#   ti_method, task_1_total, task_1_mean, task_2_total, task_2_mean,
#   overall_total, overall_mean
required_cols <- c("ti_method", "task_1_total", "task_2_total")
missing_cols <- setdiff(required_cols, colnames(runtime_df))
if (length(missing_cols) > 0) {
  stop("Runtime CSV is missing required column(s): ", paste(missing_cols, collapse = ", "))
}

runtime_summary <- runtime_df %>%
  mutate(
    total_runtime_both_tasks = task_1_total + task_2_total
  ) %>%
  select(ti_method, total_runtime_both_tasks)

# =============================================================================
# Merge
# =============================================================================
plot_df <- iss_summary %>%
  inner_join(runtime_summary, by = "ti_method") %>%
  mutate(
    ti_method = factor(ti_method, levels = names(METHOD_COLORS))
  ) %>%
  arrange(desc(iss_mean))

message("\n-- Plot table:")
print(plot_df, n = Inf)

# =============================================================================
# Build plot
# =============================================================================
message("\n-- Building ISS vs runtime scatter plot ...")

x_max <- max(plot_df$total_runtime_both_tasks, na.rm = TRUE)
y_max <- max(plot_df$iss_mean + ifelse(is.na(plot_df$iss_sd), 0, plot_df$iss_sd), na.rm = TRUE)

p <- ggplot(
  plot_df,
  aes(
    x     = total_runtime_both_tasks,
    y     = iss_mean,
    color = ti_method,
    fill  = ti_method,
    shape = ti_method
  )
) +
  geom_point(size = 4.2, stroke = 1.0, alpha = 0.95) +
  geom_text_repel(
    aes(label = ti_method),
    size          = 3.4,
    fontface      = "bold",
    box.padding   = 0.4,
    point.padding = 0.3,
    force         = 2,
    segment.color = "#CCCCCC",
    segment.size  = 0.3,
    max.overlaps  = Inf,
    show.legend   = FALSE
  ) +
  scale_color_manual(
    name   = "TI method",
    values = METHOD_COLORS,
    breaks = names(METHOD_COLORS),
    drop   = FALSE
  ) +
  scale_fill_manual(
    name   = "TI method",
    values = METHOD_COLORS,
    breaks = names(METHOD_COLORS),
    drop   = FALSE
  ) +
  scale_shape_manual(
    name   = "TI method",
    values = METHOD_SHAPES,
    breaks = names(METHOD_SHAPES),
    drop   = FALSE
  ) +
  scale_x_continuous(
    name   = "Total runtime across both tasks (seconds)",
    expand = expansion(mult = c(0.03, 0.08))
  ) +
  scale_y_continuous(
    name   = "Mean integration sensitivity score",
    limits = c(0, 1),
    breaks = seq(0, 1, 0.25),
    expand = expansion(mult = c(0.02, 0.04))
  ) +
  annotate(
    "text",
    x = min(plot_df$total_runtime_both_tasks, na.rm = TRUE),
    y = 0.98,
    label = "best trade-off",
    colour = "#AAAAAA",
    size = 3.2,
    fontface = "italic",
    hjust = 0,
    vjust = 1
  ) +
  annotate(
    "text",
    x = x_max,
    y = 0.05,
    label = "slow + weak",
    colour = "#AAAAAA",
    size = 3.2,
    fontface = "italic",
    hjust = 1,
    vjust = 0
  ) +
  labs(
    title = "Integration sensitivity score vs. total runtime",
    subtitle = paste0(
      "Each point = one TI method  |  x = summed runtime across task_1 and task_2  |  ",
      "y = mean ISS across both tasks  |  Upper-left = best trade-off"
    )
  ) +
  guides(
    color = guide_legend(override.aes = list(size = 4.5), order = 1),
    fill  = guide_legend(override.aes = list(size = 4.5), order = 1),
    shape = guide_legend(override.aes = list(size = 4.5), order = 1)
  ) +
  theme_iss_runtime()

# =============================================================================
# Save
# =============================================================================
out_stem <- file.path(OUTPUT_DIR, "iss_vs_total_runtime")

ggsave(
  paste0(out_stem, ".pdf"),
  plot   = p,
  width  = 9,
  height = 7.2,
  device = cairo_pdf
)
message("  [saved] iss_vs_total_runtime.pdf")

ggsave(
  paste0(out_stem, ".png"),
  plot   = p,
  width  = 9,
  height = 7.2,
  dpi    = 200,
  bg     = "white"
)
message("  [saved] iss_vs_total_runtime.png")

message("\n-- Done. Output in: ", OUTPUT_DIR, "\n")