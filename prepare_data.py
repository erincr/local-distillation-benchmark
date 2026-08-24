"""Fetch the 17 benchmark datasets BY NAME and cache them as CSVs.

Writes CSVs that ``benchmark_methods.common.load_dataset`` reads:

    openml_data/<name>.csv   (OpenML-CTR23 datasets)
    uci_data/<name>.csv      (UCI datasets)

each with a ``target`` column and every other column a feature.

    pip install openml ucimlrepo pandas
    python prepare_data.py
    python prepare_data.py --only red_wine abalone

If a dataset does not match by name, the script prints the available names so
you can adjust the alias below (change a *string*, not an id).  Dataset versions
on OpenML/UCI drift over time; verify against the paper's cached data if you
need bit-exact reproduction (see DATA_SOURCES.md).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OPENML_DIR = ROOT / "openml_data"
UCI_DIR = ROOT / "uci_data"

# internal name -> dataset name on OpenML (within the OpenML-CTR23 suite).
# Matching is case/punctuation-insensitive, so these only need to be close.
OPENML_NAMES = {
    "cars":                          "cars",
    "QSAR_fish_toxicity":            "QSAR_fish_toxicity",
    "concrete_compressive_strength": "concrete_compressive_strength",
    "socmob":                        "socmob",
    "red_wine":                      "red_wine",
    "airfoil_self_noise":            "airfoil_self_noise",
    "auction_verification":          "auction_verification",
    "space_ga":                      "space_ga",
    "abalone":                       "abalone",
    "white_wine":                    "white_wine",
}
# internal name -> dataset name on the UCI ML Repository.
UCI_NAMES = {
    "Automobile":                        "Automobile",
    "Servo":                             "Servo",
    "Liver_Disorders":                   "Liver Disorders",
    "Auto_MPG":                          "Auto MPG",
    "Real_Estate_Valuation":             "Real Estate Valuation",
    "Infrared_Thermography_Temperature": "Infrared Thermography Temperature",
    "student_performance_por":           "Student Performance",
}


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _write_csv(dest_dir, name, X, y):
    dest_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X).reset_index(drop=True)
    df["target"] = pd.Series(list(y)).reset_index(drop=True)
    out = dest_dir / f"{name}.csv"
    df.to_csv(out, index=False)
    print(f"  wrote {out.relative_to(ROOT)}  shape={df.shape}")


def fetch_openml(names):
    import openml
    # The canonical regression suite.  If the alias ever fails, pass the numeric
    # suite id instead: openml.study.get_suite(<id>)  (see www.openml.org).
    #suite = openml.study.get_suite("OpenML-CTR23")
    suite = openml.study.get_suite(353)
    avail = {}  # normalized dataset name -> data id
    for did in suite.data:
        try:
            avail[_norm(openml.datasets.get_dataset(did, download_data=False).name)] = did
        except Exception:
            pass
    for name in names:
        oname = OPENML_NAMES[name]
        did = avail.get(_norm(oname))
        if did is None:
            print(f"  [SKIP] {name}: '{oname}' not in OpenML-CTR23. "
                  f"Available: {sorted(avail)}")
            continue
        ds = openml.datasets.get_dataset(did)
        X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
        _write_csv(OPENML_DIR, name, X, y)


def fetch_uci(names):
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        print("  [SKIP all UCI] pip install ucimlrepo"); return
    for name in names:
        uname = UCI_NAMES[name]
        try:
            repo = fetch_ucirepo(name=uname)
        except Exception as e:
            print(f"  [SKIP] {name}: UCI name '{uname}' not found ({e!s:.80})")
            continue
        X = repo.data.features
        y = repo.data.targets.iloc[:, 0]   # single-target regression
        _write_csv(UCI_DIR, name, X, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these names")
    args = ap.parse_args()

    want = set(args.only) if args.only else None
    ompl = [n for n in OPENML_NAMES if want is None or n in want]
    uci = [n for n in UCI_NAMES if want is None or n in want]

    if ompl:
        print(f"Fetching {len(ompl)} OpenML-CTR23 datasets by name ...")
        fetch_openml(ompl)
    if uci:
        print(f"Fetching {len(uci)} UCI datasets by name ...")
        fetch_uci(uci)
    print("\nDone. Any [SKIP] lines: adjust the string in OPENML_NAMES/UCI_NAMES "
          "using the 'Available' list printed above.")


if __name__ == "__main__":
    main()
