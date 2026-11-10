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
    # ---- C. France: a forward-looking financial return? (DE fixed to ret+d1 ridge) ----
    E("C01_FR_level_retlag1", "FR reacts to today's and yesterday's commodity returns", S(["level", "retlag1"], R(300)), S(["ret", "d1"], R(30))),
    E("C02_FR_retlags12", "Two lags of returns", S(["level", "retlag1", "retlag2"], R(300)), S(["ret", "d1"], R(30))),
    E("C03_FR_leads", "Futures price in tomorrow's fundamentals: add levels of t+1 and t+2", S(["level", "retlag1", "lvllead1", "lvllead2"], R(300)), S(["ret", "d1"], R(30))),
    E("C04_FR_leads_a1000", "Same, more shrinkage", S(["level", "retlag1", "lvllead1", "lvllead2"], R(1000)), S(["ret", "d1"], R(30))),
    E("C05_FR_leads_lags", "Leads and lags of levels", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllag1"], R(1000)), S(["ret", "d1"], R(30))),
    E("C06_FR_leads_d1", "Leads + changes", S(["level", "retlag1", "lvllead1", "lvllead2", "d1"], R(1000)), S(["ret", "d1"], R(30))),
    E("C07_FR_leads3", "Three days ahead", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllead3"], R(1000)), S(["ret", "d1"], R(30))),
    E("C08_FR_leads_a3000", "Even more shrinkage", S(["level", "retlag1", "lvllead1", "lvllead2"], R(3000)), S(["ret", "d1"], R(30))),
    E("C09_FR_leads_retlead", "Next-day commodity returns as a placebo/lead check", S(["level", "retlag1", "retlead1", "lvllead1", "lvllead2"], R(1000)), S(["ret", "d1"], R(30))),
]
BY_ID = {e["id"]: e for e in EXPERIMENTS}
assert len(BY_ID) == len(EXPERIMENTS)
