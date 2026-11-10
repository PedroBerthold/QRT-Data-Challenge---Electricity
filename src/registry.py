"""All experiments in research order. IDs are permanent: never reuse or delete."""
R = lambda a: {"type": "ridge", "params": {"alpha": a}}


def E(id, hyp, FR=None, DE=None, **kw):
    d = dict(id=id, hypothesis=hyp, **kw)
    if FR:
        d["FR"], d["DE"] = FR, DE or FR
    return d


def S(blocks, model, target="rank", **kw):
    return dict(blocks=blocks, model=model, target=target, **kw)


EXPERIMENTS = [
    # ---- A. What a static model can do (no time information) ---------------
    E("A01_ols_level_raw", "Benchmark-like: per-country OLS on same-day levels, raw target", S(["level"], {"type": "ols"}, "none")),
    E("A02_ridge_level_raw", "Shrinkage", S(["level"], R(100), "none")),
    E("A03_ridge_level_rank", "Rank metric + fat tails: fit ranks of the target", S(["level"], R(100))),
    # ---- B. Time information: day-over-day changes -------------------------
    E("B01_d1", "Target is a price variation: explain it by the change of fundamentals since yesterday", S(["ret", "d1"], R(30))),
    E("B02_level_d1", "Levels + changes", S(["level", "d1"], R(30))),
    E("B03_level_d1_retlag1", "Add yesterday's commodity returns (FR target reacts to them)", S(["level", "d1", "retlag1"], R(30))),
    E("B04_level_d1_d2_retlag", "Two-day changes and two lags of returns", S(["level", "d1", "d2", "retlag1", "retlag2"], R(30))),
    E("B05_level_d1_lead1", "Day-ahead contract: tomorrow's fundamentals may matter", S(["level", "d1", "lead1", "retlag1"], R(30))),
    E("B06_level_d1_lag1d", "Yesterday's change (delayed adjustment / reversal)", S(["level", "d1", "lag1d", "retlag1"], R(30))),
    E("B07_level_d1_dev7", "Deviation from the previous week", S(["level", "d1", "dev7", "retlag1"], R(30))),
]
BY_ID = {e["id"]: e for e in EXPERIMENTS}
assert len(BY_ID) == len(EXPERIMENTS)
