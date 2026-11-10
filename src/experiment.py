"""Experiment runner. Each experiment is cross-validated with both schemes,
refitted on all training rows, and every artefact is stored (never discarded)."""
import json
import time
import datetime as dt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.data import load, ROOT
from src.features import build
from src.models import make_model, transform_target
from src.validation import random_day_folds, time_block_folds, N_SPLITS, N_REPEATS

EXP = ROOT / "experiments"
PRED = ROOT / "outputs" / "predictions"
RESULTS = EXP / "experiment_results.csv"
_CACHE = {}


def data():
    if "d" not in _CACHE:
        _CACHE["d"] = load()
    return _CACHE["d"]


def sp(a, b):
    return float(spearmanr(a, b).correlation) if len(a) > 2 and np.std(b) > 0 else np.nan


def _country_matrix(spec, days, t):
    F = build(spec["blocks"], days)
    if spec.get("cols"):
        F = F[spec["cols"]]
    return F.loc[t].values


def _fit_predict(cfg, Xs, y, c, itr, iva, seed, tt=None):
    """Per-country fit. Xs: dict country -> matrix for all rows passed.
    Optional spec["split_t"]: separate models before/after that time (regimes)."""
    pred = np.zeros(len(iva))
    for k in ("FR", "DE"):
        spec = cfg[k]
        regimes = [np.ones(len(c), bool)]
        if spec.get("split_t") is not None:
            regimes = [tt < spec["split_t"], tt >= spec["split_t"]]
        for reg in regimes:
            a = itr[(c[itr] == k) & reg[itr]]
            b = (c[iva] == k) & reg[iva]
            if b.sum() == 0 or len(a) == 0:  # e.g. early time blocks contain no DE row
                continue
            yt = transform_target(y[a], spec.get("target", "rank"))
            m = make_model(spec["model"], seed)
            m.fit(Xs[k][a], yt)
            p = np.ravel(m.predict(Xs[k][iva[b]]))
            if spec.get("nbr"):
                p = p + _neighbour_correction(spec, Xs[k], yt, tt, a, iva[b], seed)
            pred[b] = p * spec.get("scale", 1.0)
    return pred


def _neighbour_correction(spec, X, yt, tt, a, b, seed):
    """Second stage using the known targets of neighbouring days (same country).

    If the price level has a transient component, the residuals of consecutive
    days are negatively correlated, so the residuals of the training days just
    before/after a day carry information about it. Only targets of training rows
    of the current fold are used; at test time these are the real train targets.
    Residuals of training rows come from an inner K-fold, so they are out-of-sample.
    """
    nb = spec["nbr"]
    L, inner = nb.get("lags", 1), nb.get("inner", 5)
    p_tr = np.zeros(len(a))
    days = np.unique(tt[a])
    perm = np.random.RandomState(seed + 7).permutation(days)
    for chunk in np.array_split(perm, inner):
        va = np.isin(tt[a], chunk)
        mm = make_model(spec["model"], seed).fit(X[a[~va]], yt[~va])
        p_tr[va] = np.ravel(mm.predict(X[a[va]]))
    r_tr = yt - p_tr
    res = dict(zip(tt[a], r_tr))

    def feats(ts):
        Z = np.zeros((len(ts), 4 * L))
        for i, t0 in enumerate(ts):
            for side, sgn in ((0, -1), (1, 1)):
                acc = 0.0
                for k in range(1, L + 1):   # contiguous run of known neighbours on this side
                    v = res.get(t0 + sgn * k)
                    if v is None:
                        break
                    acc += v
                    Z[i, 4 * (k - 1) + side] = v          # residual of the k-th neighbour
                    Z[i, 4 * (k - 1) + 2 + side] = 1.0    # availability flag
        return Z

    from sklearn.linear_model import Ridge
    st2 = Ridge(alpha=nb.get("alpha", 10.0)).fit(feats(tt[a]), r_tr)
    return st2.predict(feats(tt[b]))


def _z(a):
    return (a - a.mean(axis=0)) / a.std(axis=0)


def run(cfg, verbose=True):
    t0 = time.time()
    tr, te, days = data()
    y, c, t = tr.TARGET.values, tr.COUNTRY.values, tr.t.values
    n = len(tr)
    if "blend" in cfg:
        w = np.array(cfg.get("weights", [1.0] * len(cfg["blend"])), float)
        ld = lambda kind, e, idx: pd.read_csv(PRED / f"{kind}_{e}.csv").set_index("ID").loc[idx].values
        mode = cfg.get("blend_mode", "z")  # z: pooled z-score; the FR/DE relative scale of each member is kept
        oof = sum(wi * _z(ld("oof", e, tr.ID)) for wi, e in zip(w, cfg["blend"]))
        blk = sum(wi * _z(ld("oofblock", e, tr.ID)) for wi, e in zip(w, cfg["blend"]))[:, 0]
        test = sum(wi * _z(ld("test", e, te.ID)) for wi, e in zip(w, cfg["blend"]))[:, 0]
    else:
        Xs = {k: _country_matrix(cfg[k], days, tr.t.values) for k in ("FR", "DE")}
        Xt = {k: _country_matrix(cfg[k], days, te.t.values) for k in ("FR", "DE")}
        oof = np.zeros((n, N_REPEATS))
        for r, f, itr, iva in random_day_folds(t):
            oof[iva, r] = _fit_predict(cfg, Xs, y, c, itr, iva, 100 * r + f, t)
        blk = np.zeros(n)
        for r, f, itr, iva in time_block_folds(t):
            blk[iva] = _fit_predict(cfg, Xs, y, c, itr, iva, f, t)
        # final fit on all training rows -> test
        allX = {k: np.vstack([Xs[k], Xt[k]]) for k in ("FR", "DE")}
        call = np.concatenate([c, te.COUNTRY.values])
        test = _fit_predict(cfg, allX, np.concatenate([y, np.zeros(len(te))]), call, np.arange(n), np.arange(n, n + len(te)), 0, np.concatenate([t, te.t.values]))

    folds = pd.DataFrame([{"scheme": "random", "repeat": r, "fold": f, "n": len(iva), "spearman": sp(y[iva], oof[iva, r])}
                          for r, f, itr, iva in random_day_folds(t)] +
                         [{"scheme": "time_block", "repeat": 0, "fold": f, "n": len(iva), "spearman": sp(y[iva], blk[iva])}
                          for r, f, itr, iva in time_block_folds(t)])
    rnd = folds[folds.scheme == "random"].spearman
    tb = folds[folds.scheme == "time_block"].spearman
    fr, de = c == "FR", c == "DE"
    rep = lambda m: float(np.mean([sp(y[m], oof[m, r]) for r in range(N_REPEATS)]))
    eid = cfg["id"]
    for d in (EXP / "configurations", EXP / "fold_scores", PRED):
        d.mkdir(parents=True, exist_ok=True)
    (EXP / "configurations" / f"{eid}.json").write_text(json.dumps(cfg, indent=1))
    folds.to_csv(EXP / "fold_scores" / f"{eid}.csv", index=False)
    pd.DataFrame(oof, columns=[f"rep{r}" for r in range(N_REPEATS)], index=pd.Index(tr.ID, name="ID")).round(6).to_csv(PRED / f"oof_{eid}.csv")
    pd.DataFrame({"ID": tr.ID, "pred": np.round(blk, 6)}).to_csv(PRED / f"oofblock_{eid}.csv", index=False)
    pd.DataFrame({"ID": te.ID, "TARGET": test}).to_csv(PRED / f"test_{eid}.csv", index=False)
    desc = lambda k: "+".join(cfg[k]["blocks"]) + " | " + json.dumps(cfg[k]["model"]) + " | target=" + cfg[k].get("target", "rank")
    row = {"id": eid, "date": dt.date.today().isoformat(), "hypothesis": cfg.get("hypothesis", ""),
           "FR": desc("FR") if "FR" in cfg else "blend", "DE": desc("DE") if "DE" in cfg else "+".join(cfg.get("blend", [])),
           "validation": f"{N_REPEATS}x{N_SPLITS} random day folds (seeds 0-{N_REPEATS - 1}) + 5 time blocks",
           "spearman": round(rep(np.ones(n, bool)), 4), "FR_spearman": round(rep(fr), 4), "DE_spearman": round(rep(de), 4),
           "fold_mean": round(rnd.mean(), 4), "fold_std": round(rnd.std(), 4), "fold_min": round(rnd.min(), 4),
           "time_block_pooled": round(sp(y, blk), 4), "time_block_min": round(tb.min(), 4),
           "seconds": round(time.time() - t0, 1)}
    res = pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame()
    if len(res):
        res = res[res.id != eid]
    pd.concat([res, pd.DataFrame([row])], ignore_index=True).to_csv(RESULTS, index=False)
    if verbose:
        print(f"{eid:26s} all={row['spearman']:.4f} FR={row['FR_spearman']:.4f} DE={row['DE_spearman']:.4f} "
              f"fold {row['fold_mean']:.3f}±{row['fold_std']:.3f} min {row['fold_min']:.3f} | blocks {row['time_block_pooled']:.4f} min {row['time_block_min']:.3f} ({row['seconds']}s)", flush=True)
    return row
