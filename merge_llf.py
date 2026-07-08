"""Fold LLF (R) results into the main benchmark results.jsonl as method "llf".

Reads results/llf/llf_results.jsonl (written by llf_grf.R) and appends any
(dataset, "llf", seed) rows not already present to
results/local_explanation_benchmark/results.jsonl, then recomputes the
aggregate tables so "llf" appears alongside LOESS in the teacher-unaware column.

LLF is teacher-unaware (no phi, no per-point betas), so it logs only the
predictive metrics (test_mse, test_r2); interpretability columns are absent and
aggregate as NaN.  Run after the R step:  `python merge_llf.py`
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LLF = ROOT / "results" / "llf" / "llf_results.jsonl"
MAIN = ROOT / "results" / "local_explanation_benchmark" / "results.jsonl"


def existing_keys():
    keys = set()
    if MAIN.exists():
        with open(MAIN) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("status") == "ok":
                        keys.add((r["dataset"], r["method"], int(r["seed"])))
                except Exception:
                    pass
    return keys


def main():
    if not LLF.exists():
        print(f"no LLF results at {LLF}"); return
    keys = existing_keys()
    MAIN.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    with open(MAIN, "a") as out, open(LLF) as src:
        for line in src:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") != "ok":
                continue
            k = (r["dataset"], "llf", int(r["seed"]))
            if k in keys:
                continue
            keys.add(k)
            out.write(json.dumps(r) + "\n"); added += 1
    print(f"merged {added} LLF rows into {MAIN}")

    # Recompute aggregates so llf shows up in the tables.
    sys.path.insert(0, str(ROOT))
    try:
        import experiment_explanation_benchmark as B
        B.write_aggregates()
        print("aggregates rewritten")
    except Exception as e:
        print(f"[warn] could not rewrite aggregates ({e!s:.100}); "
              f"re-run the benchmark once to refresh them")


if __name__ == "__main__":
    main()
