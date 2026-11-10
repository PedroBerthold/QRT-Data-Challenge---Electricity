# QRT Data Challenge: explaining daily electricity price variations (FR / DE)

Metric: Spearman correlation pooled over FR and DE rows. Final model `K01_ensemble_main`: **0.60** out-of-fold
(90% interval on a test-sized sample 0.55 to 0.65), against 0.23 for a per-country OLS on same-day levels and
15.9% public score for the organisers' benchmark.

The result rests on one structural fact: `DAY_ID` is shuffled but `ID` is chronological, so the days can be
put back in time order (`src/data.py`, evidence in `outputs/diagnostics/audit.md`). The targets are explained
by how fundamentals change through time, not by their level on the day.

## Setup

```bash
pip install -r requirements.txt      # Python 3.10+
```

## How to run

| Step | Command | Output |
|---|---|---|
| Data audit | `python -m src.audit` | `outputs/diagnostics/audit.md` + 3 figures |
| List experiments | `python main.py list` | |
| One experiment, or a block by prefix | `python main.py run B01_d1` / `python main.py run H` | see below |
| Reproduce every result | `python main.py run --all` (about 6 min) | `experiments/experiment_results.csv` |
| Robustness checks | `python -m src.robustness` | `outputs/diagnostics/robustness.txt` |
| Final model + submission | `python main.py run H05 H03_FR J03 J11 J09 K01 && python main.py submit K01_ensemble_main` | `outputs/submission_K01_ensemble_main.csv` |

Each experiment stores its config (`experiments/configurations/`), fold scores for both validation schemes
(`experiments/fold_scores/`), out-of-fold and test predictions (`outputs/predictions/`) and one row in
`experiments/experiment_results.csv`. Ensembles read the stored predictions of their members.

## Layout

- `data/` raw files, never modified
- `src/data.py` loading and time axis; `src/features.py` feature blocks (levels, changes, leads, lags, trends)
- `src/models.py` models and target transforms; `src/validation.py` the two validation schemes
- `src/experiment.py` runner, including the neighbour-residual second stage; `src/registry.py` every experiment
- `src/audit.py`, `src/robustness.py` diagnostics
- `experiments/RESEARCH_LOG.md` hypotheses, results, conclusions; `PROGRESS.md` status

## Method in short

- One model per country, fitted on within-country ranks of the target.
- DE: ridge / SVR on commodity returns and the change of every fundamental since the previous day, then a
  second stage using the residuals of the adjacent training days.
- FR: ridge on commodity returns of the day and the day before, and the 3-day trend of fundamentals
  (mean of the 3 following days minus mean of the 3 previous days).
- Final prediction: average of 5 such runs.

Submissions in `outputs/`: `K01` (main), `K02` (no neighbour-target stage), `K03` (two best single runs).
