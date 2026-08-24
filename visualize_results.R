library(jsonlite)
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)
library(data.table)
library(latex2exp)


# --- config ------------------------------------------------------------------
# Paths are relative to the repo root (run this script from there, or set ROOT).
ROOT   <- ""
JSONL  <- file.path(ROOT, "results", "local_explanation_benchmark", "results.jsonl")
OUTDIR <- file.path(ROOT, "figures")
PAPER  <- OUTDIR   # where plots are written
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)

# Dataset ordering (n_train ascending — matches the experiment script order)
dataset_order <- c(
  "Automobile","Servo","Liver_Disorders","Auto_MPG","Real_Estate_Valuation",
  "forest_fires","student_performance_por","energy_efficiency","cars",
  "QSAR_fish_toxicity","concrete_compressive_strength",
  "Infrared_Thermography_Temperature","socmob","red_wine","airfoil_self_noise",
  "auction_verification","space_ga","white_wine","abalone"
)

# --- colors ------------------------------------------------------------------
# Teacher families: TabPFN = blues, XGBoost = wine-rose, TabFM = purples.
# Each family has a teacher, a lasso student "(L)", and a ridge student "(R)".
# The ridge students only appear once the Great Lakes rerun has produced them.
method_colors <- c(
  "Lasso"              = "#2d2e6f",  # deep ultramarine — global lasso baseline
  "Ridge"              = "#77778a",  # blue-gray — global ridge baseline
  # TabPFN family (blues)
  "TabPFN"             = "#2d5aa8",  # rich blue — teacher
  "TabPFN, dist (L)"   = "#7d9bd4",  # lighter blue — lasso student
  "TabPFN, dist (R)"   = "#4a78b8",  # mid blue — ridge student
  # XGBoost family (warm wine-rose)
  "XGBoost"            = "#8a2148",  # deep wine-rose — teacher
  "XGB, dist (L)"      = "#d97aa3",  # lighter rose — lasso student
  "XGB, dist (R)"      = "#b85478",  # mid rose — ridge student
  # TabFM family (purples/violets)
  "TabFM"              = "#5b2d8a",  # deep violet — teacher
  "TabFM, dist (L)"    = "#9b7dd4",  # lighter violet — lasso student
  "TabFM, dist (R)"    = "#7a5aae",  # mid violet — ridge student
  # Adaptive teacher-selection method (its own family)
  "Dist (best teacher)" = "#d1a000", # gold — per-sim argmax mu_hat teacher
  # Other local methods
  "LLF"                = "#2a9d8f",  # teal — local linear forest baseline
  "LOESS"              = "#3d5a2e",  # sage-teal
  "LIME"               = "#8ec4a0",  # soft mint
  "MAPLE"              = "#5ea87e"   # deep olive
)

# Bottom-to-top plotting / factor order, grouped by family.
# Requested "other" tail order: Lasso, LLF, LOESS, LIME, MAPLE.
method_levels <- c("TabPFN","TabPFN, dist (L)","TabPFN, dist (R)",
                   "XGBoost","XGB, dist (L)","XGB, dist (R)",
                   "TabFM","TabFM, dist (L)","TabFM, dist (R)",
                   "Dist (best teacher)",
                   "Lasso","Ridge","LLF","LOESS","LIME","MAPLE")

# --- load --------------------------------------------------------------------
read_jsonl <- function(path) {
  if (!file.exists(path)) return(NULL)
  cat("Reading", path, "...\n")
  raw <- readLines(path)
  raw <- gsub("\\bNaN\\b", "null", raw)
  raw <- gsub("-?Infinity", "null", raw)
  parsed <- lapply(raw, function(l)
    tryCatch(jsonlite::fromJSON(l, simplifyVector = TRUE), error = function(e) NULL))
  n_bad <- sum(sapply(parsed, is.null))
  if (n_bad > 0) cat("  skipped", n_bad, "malformed lines\n")
  parsed <- parsed[!sapply(parsed, is.null)]
  parsed_clean <- lapply(parsed, function(x)
    lapply(x, function(v) if (length(v) == 0) NA else v))
  rbindlist(parsed_clean, fill = TRUE)
}

df <- read_jsonl(JSONL)

# Drop ablation-only variants (kept out of the main comparison plots), plus
# LIME and MAPLE: these are post-hoc local *explainers* of a fixed black box,
# not predictive models, so test R² is off-purpose for them.  They are scoped
# out of the paper's empirical comparison and discussed in related work instead.
df <- df[!(method %in% c("ld_full", "ld_anchor", "ld_weights",
                         "lime", "maple")), ]
df <- df[status == "ok"]

# Datasets dropped from all analyses:
#   forest_fires:      famously hard regression; all methods give R^2 ≈ 0
#   energy_efficiency: loader behavior changed between runs
DROP_DATASETS <- c("forest_fires", "energy_efficiency")
df <- df[!(dataset %in% DROP_DATASETS)]

# --- NEW METHOD: per-simulation teacher selection by largest mu_hat ----------
# For each (dataset, seed), among the three lasso distillations (one per
# teacher) pick the row whose distillation put the most weight on its teacher
# (largest mu_hat on the OOF training data).  This is an oracle-FREE selector:
# it never looks at test performance, only the CV-estimated mu_hat that the
# experiment already logged per distillation row.  Computed on the raw method
# names, before renaming.  (For a ridge analog, swap the method set below to
# c("ld_ridge","ld_xgb_ridge","ld_tabfm_ridge") once those results exist.)
tsel_pool <- df[method %in% c("ld", "ld_xgb", "ld_tabfm") & !is.na(mu_hat)]
if (nrow(tsel_pool) > 0) {
  tsel <- tsel_pool[tsel_pool[, .I[which.max(mu_hat)], by = .(dataset, seed)]$V1]
  tsel[, method := "ld_muselect"]
  df <- rbindlist(list(df, tsel), fill = TRUE)
  cat("Added ld_muselect for", nrow(tsel), "(dataset, seed) simulations\n")
} else {
  cat("WARNING: no mu_hat rows found; skipping ld_muselect\n")
}

df <- data.table(df)
df$dataset <- gsub("_", " ", df$dataset)
df[dataset == "student performance por", dataset := "student performance"]
df[, dataset := tolower(dataset)]

# --- method display names ----------------------------------------------------
df[method == "global_lasso",     method := "Lasso"]
df[method == "global_ridge",     method := "Ridge"]
df[method == "teacher",          method := "TabPFN"]
df[method == "ld",               method := "TabPFN, dist (L)"]
df[method == "ld_ridge",         method := "TabPFN, dist (R)"]
df[method == "xgboost",          method := "XGBoost"]
df[method == "ld_xgb",           method := "XGB, dist (L)"]
df[method == "ld_xgb_ridge",     method := "XGB, dist (R)"]
df[method == "tabfm",            method := "TabFM"]
df[method == "ld_tabfm",         method := "TabFM, dist (L)"]
df[method == "ld_tabfm_ridge",   method := "TabFM, dist (R)"]
df[method == "ld_muselect",      method := "Dist (best teacher)"]
df[method == "lime",             method := "LIME"]
df[method == "loess",            method := "LOESS"]
df[method == "maple",            method := "MAPLE"]
df[method == "llf",              method := "LLF"]

df[, method := factor(method, levels = method_levels)]

# Datasets ordered by the teacher's R² gain over the lasso (used by some plots).
# Uses MEDIAN R² per method: mean is fragile because a few catastrophic seeds
# (e.g. Infrared Thermography's lasso) tank the mean and inflate the gap.
gap_order <- df[method %in% c("TabPFN", "Lasso"),
                .(r2 = median(test_r2)), by = .(dataset, method)] |>
  dcast(dataset ~ method, value.var = "r2")
gap_order[, gap := TabPFN - Lasso]
ordered_datasets <- gap_order[order(-gap), dataset]   # descending median gap

#########################
# Dot plot
#########################
make_dot_plot <- function(df,
                          metric_col,
                          exclude = character(0),
                          datasets = unique(df$dataset),
                          ncol = 1,
                          title = "",
                          xlab = "",
                          direction = "max",
                          method_order = c("TabPFN","TabPFN, dist (L)","TabPFN, dist (R)",
                                           "XGBoost","XGB, dist (L)","XGB, dist (R)",
                                           "TabFM","TabFM, dist (L)","TabFM, dist (R)",
                                           "Dist (best teacher)",
                                           "Lasso","Ridge","LLF","LOESS","LIME","MAPLE"),
                          dataset_facet_order = NULL,
                          caption, label, out_path) {
  # Median center + robust SE (MAD/sqrt(n)).  Median resists outlier seeds so
  # a single catastrophic split (e.g. LIME's ~-10 R^2 on space_ga) no longer
  # drags the point off-scale; MAD keeps the error bar from re-stretching the
  # free x-axis the way sd would.
  forplot <- df[
    !(method %in% exclude) & dataset %in% datasets,
    .(m = median(get(metric_col)), se.1 = mad(get(metric_col))/sqrt(.N)),
    by=.(method, dataset)]

  forplot <- forplot[!is.na(method)]

  forplot[dataset == "infrared thermography temperature",
          dataset := "infrared therm. temp."]
  forplot[dataset == "concrete compressive strength",
          dataset := "concrete comp. strength"]

  if (!is.null(dataset_facet_order)) {
    # Apply the same abbreviations to the ordering vector so it still matches
    # forplot$dataset after the renames above (needed for the full appendix,
    # which includes infrared/concrete).
    dfo <- as.character(dataset_facet_order)
    dfo[dfo == "infrared thermography temperature"] <- "infrared therm. temp."
    dfo[dfo == "concrete compressive strength"]     <- "concrete comp. strength"
    facet_order <- intersect(dfo, unique(as.character(forplot$dataset)))
    forplot[, dataset := factor(dataset, levels = facet_order)]
  }

  # Restrict factor levels to methods actually present, in the requested
  # bottom-to-top y-axis order.
  present_in_order <- intersect(rev(method_order), unique(as.character(forplot$method)))
  forplot[, method := factor(method, levels = present_in_order)]

  # Family dividers: light gray hlines between method groups
  # (TabPFN | XGB | TabFM | selected | other locals).
  family_of <- function(m) {
    if (m %in% c("TabPFN","TabPFN, dist (L)","TabPFN, dist (R)")) "tabpfn"
    else if (m %in% c("XGBoost","XGB, dist (L)","XGB, dist (R)")) "xgb"
    else if (m %in% c("TabFM","TabFM, dist (L)","TabFM, dist (R)")) "tabfm"
    else if (m == "Dist (best teacher)") "select"
    else if (m %in% c("Lasso","Ridge")) "global_linear"
    else "local"
  }
  fam_seq <- sapply(present_in_order, family_of)
  divider_ys <- which(diff(as.integer(factor(fam_seq, levels = unique(fam_seq)))) != 0) + 0.5

  return(
    ggplot(forplot) +
      geom_hline(yintercept = divider_ys, color = "#aaaaaa", linewidth = 0.4) +
      geom_point(aes(x=m, y=method, color=method)) +
      geom_errorbar(aes(xmin=m-se.1, xmax=m+se.1, y=method, color=method)) +
      facet_wrap(~dataset, ncol=ncol) +#, scales = "free_x") +
      scale_color_manual(values = method_colors, guide = "none") +
      scale_x_continuous(
        labels = scales::number_format(accuracy = 0.01),
        breaks = scales::pretty_breaks(n = 3)
      ) +
      # theme_minimal FIRST (it is a *complete* theme and would otherwise wipe
      # any theme() tweaks added before it), then override.  theme_minimal sets
      # plot/panel backgrounds to element_blank() (transparent), which renders
      # as gray in many viewers/PNGs — so we force solid white here.
      theme_minimal(base_size = 12) +
      theme(
        panel.spacing.x  = unit(1.2, "lines"),
        plot.background  = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA),
        # Grey label bars above each facet, and a box around each panel.
        strip.background = element_rect(fill = "grey90", color = NA),
        strip.text       = element_text(margin = margin(4, 0, 4, 0)),
        panel.border     = element_rect(color = "grey70", fill = NA, linewidth = 0.4)
      ) +
      labs(y = NULL, x = xlab,
           title = title,
           subtitle = "Points are medians; error bars are 1 SE across 20 train/test splits.")
  )
}

# --- appendix: all datasets, split across two portrait full pages ------------
# 13 method rows per facet (once ridge lands) across 17 datasets won't fit one
# page legibly, so we split the datasets into two ncol=3 figures ordered by the
# teacher-over-lasso gap (page 1 = highest-gap half, page 2 = the rest).  Caption
# them as one logical figure: "Figure AX" and "Figure AX (continued)".
render_appendix <- function(ds_subset, out_path) {
  p <- make_dot_plot(df, "test_r2",
                     datasets = ds_subset,
                     ncol = 3,
                     dataset_facet_order = ordered_datasets,
                     xlab = expression(paste("Test ", R^2)),
                     title = expression(paste(R^2, " comparison across methods")))
  ggsave(out_path, plot = p, width = 10, height = 9, bg = "white")
}
appendix_ds <- as.character(ordered_datasets)
half <- ceiling(length(appendix_ds) / 2)
render_appendix(appendix_ds[seq_len(half)],
                file.path(PAPER, "appendix_r2_p1.pdf"))
render_appendix(appendix_ds[(half + 1):length(appendix_ds)],
                file.path(PAPER, "appendix_r2_p2.pdf"))

# --- main figure: two teachers (TabPFN + XGBoost), each with lasso + ridge
# students, plus Lasso / Ridge / LLF / LOESS.  Shows generality across teachers and the
# lasso-vs-ridge student comparison.  TabFM and Dist(best teacher) are dropped
# here and live in the full appendix figure.  (LIME/MAPLE are already removed
# globally at load time.)
MAIN_EXCLUDE <- c("TabFM", "TabFM, dist (L)", "TabFM, dist (R)",
                  "Dist (best teacher)")
short.plot <- make_dot_plot(df, "test_r2",
                            ncol = 3,
                            exclude = MAIN_EXCLUDE,
                            xlab = expression(paste("Test ", R^2)),
                            title = expression(paste(R^2, " comparison across methods")),
                            datasets = c("airfoil self noise", "automobile", "cars", "servo", "socmob", "student performance"),
                            dataset_facet_order = ordered_datasets) +
       labs(
         subtitle = "Datasets ordered by TabPFN's median R² improvement over the lasso.\nError bars represent 1 SE across 20 train/test splits."
         )

ggsave(file.path(PAPER, "r2_first.pdf"), plot = short.plot, width = 10, height = 5.1, bg = "white")
