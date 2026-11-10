"""Robustness checks for the final choice. Run: python -m src.robustness

1. Neighbour-stage realism: in 5-fold CV 80% of a day's neighbours have a known
   target; in the real test only 70% do (365 of 1,216 days are test days). We
   re-score the key configuration with 3 and 2 folds (67% and 50% known).
2. Bootstrap over days of the out-of-fold Spearman: interval for a train-sized
   sample and for a 365-day sample, and paired differences between ensembles.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.experiment import data, _country_matrix, _fit_predict, PRED, sp
from src.registry import BY_ID
from src.validation import random_day_folds
from src.data import ROOT


def cv_score(cfg, n_splits, n_repeats=3):
    tr, te, days = data()
    y, c, t = tr.TARGET.values, tr.COUNTRY.values, tr.t.values
    Xs = {k: _country_matrix(cfg[k], days, t) for k in ("FR", "DE")}
    out = []
    for r in range(n_repeats):
        oof = np.zeros(len(y))
        for rr, f, itr, iva in random_day_folds(t, n_splits, n_repeats):
            if rr == r:
                oof[iva] = _fit_predict(cfg, Xs, y, c, itr, iva, 100 * r + f, t)
        out.append([sp(y, oof), sp(y[c == "FR"], oof[c == "FR"]), sp(y[c == "DE"], oof[c == "DE"])])
    return np.mean(out, axis=0)


def main():
    L = []
    P = lambda s: (print(s, flush=True), L.append(s))
    P("== 1. Neighbour stage vs share of neighbours with a known target")
    for eid in ("H01_noNbr_ref", "H05_FR_tr3_nolevel"):
        for k in (5, 3, 2):
            a = cv_score(BY_ID[eid], k)
            P(f"{eid:22s} {k}-fold ({100 * (1 - 1 / k):.0f}% of days known): pooled {a[0]:.4f}  FR {a[1]:.4f}  DE {a[2]:.4f}")
    P("\n== 2. Bootstrap over days (1000 resamples)")
    tr = data()[0]
    y, t = tr.TARGET.values, tr.t.values
    uniq = np.unique(t)
    rows_of = {d: np.where(t == d)[0] for d in uniq}
    ids = ["K01_ensemble_main", "K02_ensemble_no_neighbour", "K03_ensemble_top2", "A03_ridge_level_rank", "A01_ols_level_raw"]
    oofs = {e: pd.read_csv(PRED / f"oof_{e}.csv").set_index("ID").loc[tr.ID].values for e in ids}
    rng = np.random.RandomState(0)
    res = {}
    for label, nd in (("train", len(uniq)), ("365d", 365)):
        st = {e: [] for e in ids}
        for b in range(1000):
            idx = np.concatenate([rows_of[d] for d in rng.choice(uniq, nd, replace=True)])
            for e in ids:
                st[e].append(spearmanr(y[idx], oofs[e][idx, b % oofs[e].shape[1]]).correlation)
        res[label] = pd.DataFrame(st)
    rep = pd.DataFrame({"mean": res["train"].mean(), "p05": res["train"].quantile(.05), "p95": res["train"].quantile(.95),
                        "p05_365d": res["365d"].quantile(.05), "p95_365d": res["365d"].quantile(.95),
                        "diff_vs_K01": (res["train"].sub(res["train"][ids[0]], axis=0)).mean(),
                        "P(better than K01)": (res["train"].gt(res["train"][ids[0]], axis=0)).mean()}).round(4)
    P(rep.to_string())
    (ROOT / "outputs" / "diagnostics" / "robustness.txt").write_text("\n".join(L))


if __name__ == "__main__":
    main()
