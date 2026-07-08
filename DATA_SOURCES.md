# Data sources

The 17 regression datasets are **not** redistributed with this repository; they
are fetched from their original public sources by `prepare_data.py`. They come
from two collections (7 UCI + 10 OpenML-CTR23):

- **UCI Machine Learning Repository** — <https://archive.ics.uci.edu/>
- **OpenML-CTR23**, a curated tabular regression benchmark suite on OpenML —
  <https://www.openml.org/> (Fischer et al., *OpenML-CTR23 – A curated tabular
  regression benchmarking suite*, 2023).

`benchmark_methods/common.load_dataset(name)` resolves each dataset by looking,
in order, for `<name>.npz`, then `openml_data/<name>.csv`, then
`uci_data/<name>.csv` (case-insensitive; CSVs need a `target` column). It then
drops NaN rows, one-hot encodes categoricals (`drop_first=True`), and removes
exact-duplicate rows — so the modelled `n`/`p`can differ from the raw dataset sizes.

| Internal name | Dataset | Source |
|---|---|---|
| `Automobile` | Automobile | UCI |
| `Servo` | Servo | UCI |
| `Liver_Disorders` | Liver Disorders (BUPA) | UCI |
| `Auto_MPG` | Auto MPG | UCI |
| `Real_Estate_Valuation` | Real Estate Valuation | UCI |
| `Infrared_Thermography_Temperature` | Infrared Thermography Temperature | UCI |
| `student_performance_por` | Student Performance (Portuguese) | UCI |
| `cars` | cars | OpenML-CTR23 |
| `QSAR_fish_toxicity` | QSAR fish toxicity | OpenML-CTR23 |
| `concrete_compressive_strength` | Concrete Compressive Strength | OpenML-CTR23 |
| `socmob` | socmob | OpenML-CTR23 |
| `red_wine` | Wine Quality (red) | OpenML-CTR23 |
| `airfoil_self_noise` | Airfoil Self-Noise | OpenML-CTR23 |
| `auction_verification` | Auction Verification | OpenML-CTR23 |
| `space_ga` | space_ga | OpenML-CTR23 |
| `abalone` | Abalone | OpenML-CTR23 |
| `white_wine` | Wine Quality (white) | OpenML-CTR23 |
