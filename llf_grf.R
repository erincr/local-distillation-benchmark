#!/usr/bin/env Rscript
# Local Linear Forests (grf::ll_regression_forest) as a teacher-free comparator,
# parallel to LOESS.  Reads the EXACT standardized splits written by
# export_splits.py (results/llf/splits/), so LLF runs on byte-identical
# train/test splits as every Python method -- R never reseeds.
#
# One dataset per SLURM_ARRAY_TASK_ID (0-16); seeds 0-19 inside one task.
# Writes one JSON line per (dataset, seed) to results/llf/llf_results.jsonl,
# which merge_llf.py folds into the main results.jsonl as method "llf".
#
# NOTE (confirm on the installed grf): predict() on an ll_regression_forest
# selects the ridge penalty ll.lambda by CV when ll.lambda is not supplied, and
# uses all covariates as linear-correction variables by default -- matching the
# LOESS comparator (dense local linear fit, no feature selection).

suppressMessages(library(grf))

root    <- getwd()
outdir  <- file.path(root, "results", "llf")
splits  <- file.path(outdir, "splits")
preddir <- file.path(outdir, "preds")
dir.create(preddir, recursive = TRUE, showWarnings = FALSE)
resfile <- file.path(outdir, "llf_results.jsonl")

datasets <- readLines(file.path(outdir, "datasets.txt"))
datasets <- datasets[nzchar(datasets)]

task_id <- as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID", "-1"))
if (task_id >= 0) {
  if (task_id >= length(datasets)) { cat("task_id out of range\n"); quit(status = 0) }
  names_to_run <- datasets[task_id + 1]     # R is 1-indexed; the array is 0-indexed
} else {
  names_to_run <- datasets
}

parse_seeds <- function(spec) {
  out <- integer(0)
  for (chunk in strsplit(spec, ",")[[1]]) {
    chunk <- trimws(chunk)
    if (grepl("-", chunk)) {
      ab <- as.integer(strsplit(chunk, "-")[[1]]); out <- c(out, ab[1]:ab[2])
    } else if (nchar(chunk)) out <- c(out, as.integer(chunk))
  }
  out
}
seeds <- parse_seeds(Sys.getenv("LLF_SEEDS", "0-19"))

jnum <- function(x) if (is.na(x)) "null" else format(x, digits = 10, scientific = FALSE)

fit_one <- function(name, seed) {
  tr_csv   <- file.path(splits,  sprintf("%s__%d__train.csv", name, seed))
  te_csv   <- file.path(splits,  sprintf("%s__%d__test.csv",  name, seed))
  pred_csv <- file.path(preddir, sprintf("%s__%d.csv",        name, seed))
  if (!file.exists(tr_csv) || !file.exists(te_csv)) {
    cat(sprintf("  MISSING split for %s seed=%d -- run export_splits.py first\n", name, seed))
    return(invisible(NULL))
  }
  if (file.exists(pred_csv)) {
    cat(sprintf("  skip %s seed=%d: exists\n", name, seed)); return(invisible(NULL))
  }
  tr <- read.csv(tr_csv); te <- read.csv(te_csv)
  ycol <- ncol(tr)
  X   <- as.matrix(tr[, -ycol, drop = FALSE]); Y   <- tr[[ycol]]
  Xte <- as.matrix(te[, -ycol, drop = FALSE]); Yte <- te[[ycol]]

  forest <- ll_regression_forest(X, Y, seed = seed)
  pred   <- predict(forest, Xte)$predictions           # ll.lambda tuned by CV

  mse <- mean((Yte - pred)^2)
  r2  <- 1 - sum((Yte - pred)^2) / sum((Yte - mean(Yte))^2)

  write.csv(data.frame(yhat = pred, y = Yte), pred_csv, row.names = FALSE)
  line <- sprintf(
    paste0('{"dataset": "%s", "method": "llf", "seed": %d, "status": "ok", ',
           '"n_train": %d, "n_test": %d, "p": %d, "test_mse": %s, "test_r2": %s}'),
    name, seed, nrow(X), nrow(Xte), ncol(X), jnum(mse), jnum(r2))
  cat(line, "\n", file = resfile, sep = "", append = TRUE)
  cat(sprintf("  OK %s seed=%d  MSE=%.4f R2=%.4f\n", name, seed, mse, r2))
}

for (name in names_to_run) {
  for (seed in seeds) {
    tryCatch(fit_one(name, seed),
             error = function(e)
               cat(sprintf("  ERR %s seed=%d: %s\n", name, seed, conditionMessage(e))))
  }
}
