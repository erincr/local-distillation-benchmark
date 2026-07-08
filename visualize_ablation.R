# Ablation figure: are BOTH components of local distillation necessary?

library(jsonlite)
library(data.table)
library(ggplot2)
library(scales)

# --- config (repo-relative) --------------------------------------------------
ROOT   <- "."
JSONL  <- file.path(ROOT, "results", "local_explanation_benchmark", "results.jsonl")
OUTDIR <- file.path(ROOT, "figures")
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)
DROP_DATASETS <- c("forest_fires", "energy_efficiency")


# --- load (tolerant of malformed lines) --------------------------------------
read_jsonl <- function(path) {
  raw <- readLines(path)
  raw <- gsub("\\bNaN\\b", "null", raw)
  raw <- gsub("-?Infinity", "null", raw)
  parsed <- lapply(raw, function(l)
    tryCatch(fromJSON(l, simplifyVector = TRUE), error = function(e) NULL))
  parsed <- parsed[!sapply(parsed, is.null)]
  parsed <- lapply(parsed, function(x) lapply(x, function(v) if (length(v) == 0) NA else v))
  rbindlist(parsed, fill = TRUE)
}

df <- read_jsonl(JSONL)
df <- df[status == "ok" & !(dataset %in% DROP_DATASETS)]

# --- keep the four 2x2 corners, label them -----------------------------------
cell_map <- c(global_lasso = "Neither",
              ld_weights   = "Weights only",
              ld_anchor    = "Prior only",
              ld           = "Both (full)")
cell_levels <- c("Neither", "Weights only", "Prior only", "Both (full)")
cell_colors <- c("Neither"      = "#9aa0a6",   
                 "Weights only" = "#7d9bd4",   
                 "Prior only"   = "#2d5aa8",   
                 "Both (full)"  = "#d1a06a" )   

abl <- df[method %in% names(cell_map)]
abl[, cell := factor(cell_map[method], levels = cell_levels)]
abl[, dataset := tolower(gsub("_", " ", dataset))]
abl[dataset == "student performance por", dataset := "student performance"]

# =============================================================================
# Figure 1: per-dataset dot plot (free x-scale so each dataset keeps its scale)
# =============================================================================
forplot <- abl[, .(m = median(test_r2), se = mad(test_r2) / sqrt(.N)),
               by = .(cell, dataset)]

# order facets by the full method's median R^2 (best first)
ord <- forplot[cell == "Both (full)"][order(-m), dataset]
forplot[, dataset := factor(dataset, levels = ord)]
# nicer facet labels
forplot[dataset == "infrared thermography temperature", dataset := "infrared therm. temp."]
forplot[dataset == "concrete compressive strength",     dataset := "concrete comp. strength"]
# plot bottom->top: Neither, Weights, Prior, Both
forplot[, cell := factor(cell, levels = rev(cell_levels))]

p_by_dataset <- ggplot(forplot, aes(x = m, y = cell, color = cell)) +
  geom_point(size = 1.6) +
  geom_errorbar(aes(xmin = m - se, xmax = m + se), width = 0) +
  facet_wrap(~dataset, scales = "free_x") +
  scale_color_manual(values = cell_colors, guide = "none") +
  scale_x_continuous(labels = number_format(accuracy = 0.01),
                     breaks = pretty_breaks(n = 3)) +
  theme_minimal(base_size = 12) +
  theme(panel.spacing.x  = unit(1.0, "lines"),
        plot.background  = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA)) +
  labs(y = NULL, x = expression(paste("Test ", R^2)),
       title = "Ablation: both components are useful for prediction",
       subtitle = paste0("Similarity weights and the teacher prior, on/off. ",
                         "Points are medians; error bars 1 SE across 20 splits."))
ggsave(file.path(OUTDIR, "ablation_by_dataset.pdf"), p_by_dataset,
       width = 10, height = 5, bg = "white")

