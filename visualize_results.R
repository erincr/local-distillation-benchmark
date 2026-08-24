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

# Dataset ordering (n_train ascending)
dataset_order <- c(
  "Automobile","Servo","Liver_Disorders","Auto_MPG","Real_Estate_Valuation",
  "forest_fires","student_performance_por","energy_efficiency","cars",
  "QSAR_fish_toxicity","concrete_compressive_strength",
  "Infrared_Thermography_Temperature","socmob","red_wine","airfoil_self_noise",
  "auction_verification","space_ga","white_wine","abalone"
)

# --- colors ------------------------------------------------------------------
# Teacher families: TabPFN = blues, XGBoost = rose, TabFM = purples.
method_colors <- c(
  "Lasso"              = "#2d2e6f",  
  "Ridge"              = "#77778a",  
  # TabPFN family (blues)
  "TabPFN"             = "#2d5aa8",  
  "LD (TabPFN, L)"   = "#7d9bd4", 
  "LD (TabPFN, R)"   = "#4a78b8",  
  # XGBoost family (warm rose)
  "XGBoost"            = "#8a2148",  
  "LD (XGB, L)"      = "#d97aa3",  
  "LD (XGB, R)"      = "#b85478",  
  # TabFM family (purples/violets)
  "TabFM"              = "#5b2d8a",  
  "LD (TabFM, L)"    = "#9b7dd4",  
  "LD (TabFM, R)"    = "#7a5aae",  
  # Adaptive teacher-selection method (its own family)
  "Dist (best teacher)" = "#d1a000",
  # Other local methods
  "LLF"                = "#2a9d8f",  
  "LOESS"              = "#3d5a2e",  
  "LIME"               = "#8ec4a0", 
  "MAPLE"              = "#5ea87e" 
)

method_levels <- c("TabPFN","LD (TabPFN, L)","LD (TabPFN, R)",
                   "XGBoost","LD (XGB, L)","LD (XGB, R)",
                   "TabFM","LD (TabFM, L)","LD (TabFM, R)",
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

# Ablation not in main plots 
# MAPLE not in main plots
df <- df[!(method %in% c("ld_full", "ld_anchor", "ld_weights",
                         "lime",
                         "maple")), ]
df <- df[status == "ok"]

# Datasets dropped from all analyses:
#   forest_fires:      famously hard regression; all methods give R^2 = 0 skews y range
#   energy_efficiency: loader behavior changed between runs
DROP_DATASETS <- c("forest_fires", "energy_efficiency")
df <- df[!(dataset %in% DROP_DATASETS)]

# --- Best teacher ----------
# For each (dataset, seed), among the three lasso distillations, 
# pick the row whose distillation put the most weight on its teacher
# (largest mu_hat on the OOF training data).  
# Equivalent to: lowest CV error among the teacher methods
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
df[method == "ld",               method := "LD (TabPFN, L)"]
df[method == "ld_ridge",         method := "LD (TabPFN, R)"]
df[method == "xgboost",          method := "XGBoost"]
df[method == "ld_xgb",           method := "LD (XGB, L)"]
df[method == "ld_xgb_ridge",     method := "LD (XGB, R)"]
df[method == "tabfm",            method := "TabFM"]
df[method == "ld_tabfm",         method := "LD (TabFM, L)"]
df[method == "ld_tabfm_ridge",   method := "LD (TabFM, R)"]
df[method == "ld_muselect",      method := "Dist (best teacher)"]
df[method == "lime",             method := "LIME"]
df[method == "loess",            method := "LOESS"]
df[method == "maple",            method := "MAPLE"]
df[method == "llf",              method := "LLF"]

df[, method := factor(method, levels = method_levels)]

# Datasets ordered by the teacher's improvement over the lasso 
# (used by some plots).
gap_order <- df[method %in% c("TabPFN", "Lasso"),
                .(r2 = median(test_r2)), by = .(dataset, method)] |>
  dcast(dataset ~ method, value.var = "r2")
gap_order[, gap := TabPFN - Lasso]
ordered_datasets <- gap_order[order(-gap), dataset] 

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
                          method_order = c("TabPFN","LD (TabPFN, L)","LD (TabPFN, R)",
                                           "XGBoost","LD (XGB, L)","LD (XGB, R)",
                                           "TabFM","LD (TabFM, L)","LD (TabFM, R)",
                                           "Dist (best teacher)",
                                           "Lasso","Ridge","LLF","LOESS","LIME","MAPLE"),
                          dataset_facet_order = NULL,
                          caption, label, out_path) {
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
    dfo <- as.character(dataset_facet_order)
    dfo[dfo == "infrared thermography temperature"] <- "infrared therm. temp."
    dfo[dfo == "concrete compressive strength"]     <- "concrete comp. strength"
    facet_order <- intersect(dfo, unique(as.character(forplot$dataset)))
    forplot[, dataset := factor(dataset, levels = facet_order)]
  }

  present_in_order <- intersect(rev(method_order), unique(as.character(forplot$method)))
  forplot[, method := factor(method, levels = present_in_order)]

  # Family dividers: hlines between method groups
  # (TabPFN | XGB | TabFM | selected | other locals).
  family_of <- function(m) {
    if (m %in% c("TabPFN","LD (TabPFN, L)","LD (TabPFN, R)")) "tabpfn"
    else if (m %in% c("XGBoost","LD (XGB, L)","LD (XGB, R)")) "xgb"
    else if (m %in% c("TabFM","LD (TabFM, L)","LD (TabFM, R)")) "tabfm"
    else if (m == "Dist (best teacher)") "select"
    else if (m %in% c("Lasso","Ridge")) "global_linear"
    else "local"
  }
  fam_seq <- sapply(present_in_order, family_of)
  divider_ys <- which(diff(as.integer(factor(fam_seq, levels = unique(fam_seq)))) != 0) + 0.5

  return(
    ggplot(forplot) +
      geom_hline(yintercept = divider_ys, color = "#aaaaaa", linewidth = 0.4) +
      geom_errorbar(aes(xmin=m-se.1, xmax=m+se.1, y=method, color=method)) +
      geom_point(aes(x=m, y=method, color=method)) +
      facet_wrap(~dataset, ncol=ncol) +#, scales = "free_x") +
      scale_color_manual(values = method_colors, guide = "none") +
      # scale_x_continuous(
      #   labels = scales::number_format(accuracy = 0.01),
      #   breaks = scales::pretty_breaks(n = 3)
      # ) +
      scale_x_continuous(
        labels = scales::number_format(accuracy = 0.01),
        breaks = scales::pretty_breaks(n = 3),
        limits = c(-0.25, 1), oob = scales::squish
      ) +
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
MAIN_EXCLUDE <- c("TabFM", "LD (TabFM, L)", "LD (TabFM, R)",
                  "Dist (best teacher)",
                  "LIME")
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

ggsave(file.path(PAPER, "r2_first.pdf"), plot = short.plot, width = 10, height = 5.5, bg = "white")


