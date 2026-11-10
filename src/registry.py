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
    # ---- D. Germany: refine the change-based model (FR fixed to C05 spec) ----
    E("D01_DE_d1_a3", "DE signal is strong: less shrinkage", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllag1"], R(1000)), S(["ret", "d1"], R(3))),
    E("D02_DE_d1_a100", "More shrinkage", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllag1"], R(1000)), S(["ret", "d1"], R(100))),
    E("D03_DE_d1_gauss", "Normal-score target instead of uniform ranks", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllag1"], R(1000)), S(["ret", "d1"], R(30), "gauss")),
    E("D04_DE_d1_raw", "Raw target (fat tails) as a check", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllag1"], R(1000)), S(["ret", "d1"], R(30), "none")),
    E("D05_DE_d1_signlog", "sign*log1p target", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllag1"], R(1000)), S(["ret", "d1"], R(30), "signlog")),
    E("D06_DE_d1_lag1d", "Yesterday's change too (delayed adjustment)", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllag1"], R(1000)), S(["ret", "d1", "lag1d"], R(30))),
    E("D07_DE_d1_lead1", "Tomorrow's change", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllag1"], R(1000)), S(["ret", "d1", "lead1"], R(30))),
    E("D08_DE_d1_d2", "Two-day change", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllag1"], R(1000)), S(["ret", "d1", "d2"], R(30))),
    E("D09_DE_d1_level_lgbm", "Non-linear price curve: boosting on changes + levels", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllag1"], R(1000)), S(["level", "d1"], {"type": "lgbm", "params": {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 8, "min_child_samples": 15, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.6, "reg_lambda": 5}})),
    E("D10_DE_d1_lgbm", "Boosting on changes only", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllag1"], R(1000)), S(["ret", "d1"], {"type": "lgbm", "params": {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 8, "min_child_samples": 15, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.6, "reg_lambda": 5}})),
    E("D11_DE_d1_level_lgbm_gauss", "Boosting, normal-score target", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllag1"], R(1000)), S(["level", "d1"], {"type": "lgbm", "params": {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 8, "min_child_samples": 15, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.6, "reg_lambda": 5}}, "gauss")),
    E("D12_DE_d1_svr", "RBF SVR on changes", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllag1"], R(1000)), S(["ret", "d1"], {"type": "svr", "params": {"C": 1.0, "epsilon": 0.05, "gamma": 0.005}})),
    E("D13_DE_d1_level_svr", "RBF SVR on changes + levels", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllag1"], R(1000)), S(["level", "d1"], {"type": "svr", "params": {"C": 1.0, "epsilon": 0.05, "gamma": 0.003}})),
    E("D14_DE_retlag", "Commodity lags for DE", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllag1"], R(1000)), S(["ret", "retlag1", "d1"], R(30))),
    # ---- E. France: model class, regimes, relative scale (DE fixed to ret+d1 ridge) ----
    E("E01_FR_lgbm", "FR non-linear: shallow boosting on the same blocks", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllag1"], {"type": "lgbm", "params": {"n_estimators": 300, "learning_rate": 0.02, "num_leaves": 5, "min_child_samples": 30, "subsample": 0.7, "subsample_freq": 1, "colsample_bytree": 0.4, "reg_lambda": 10}}), S(["ret", "d1"], R(30))),
    E("E02_FR_gauss", "Normal-score target for FR", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllag1"], R(1000), "gauss"), S(["ret", "d1"], R(30))),
    E("E03_FR_deltas", "FR in changes: yesterday->today, today->tomorrow, tomorrow->day after", S(["ret", "retlag1", "d1", "lead1", "lead2"], R(1000)), S(["ret", "d1"], R(30))),
    E("E04_FR_levels_and_deltas", "Levels (t-1..t+2) and all changes", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllag1"] + ["d1", "lead1", "lead2"], R(1000)), S(["ret", "d1"], R(30))),
    E("E05_FR_svr", "RBF SVR for FR", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllag1"], {"type": "svr", "params": {"C": 0.3, "epsilon": 0.1, "gamma": 0.002}}), S(["ret", "d1"], R(30))),
    E("E06_FR_regimes", "FR relationships differ before/after t=284 (start of DE rows): separate models", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllag1"], R(1000), split_t=284), S(["ret", "d1"], R(30))),
    E("E07_FR_regimes_small", "Same with a small feature set", S(["level", "retlag1"], R(300), split_t=284), S(["ret", "d1"], R(30))),
    E("E08_FR_a300", "Less shrinkage", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllag1"], R(300)), S(["ret", "d1"], R(30))),
    E("E09_FR_rf", "Random forest for FR", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllag1"], {"type": "rf", "params": {"n_estimators": 300, "min_samples_leaf": 10, "max_features": 0.3}}), S(["ret", "d1"], R(30))),
    E("E10_FR_scale05", "Pooled ranking: shrink FR predictions vs DE", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllag1"], R(1000), scale=0.5), S(["ret", "d1"], R(30))),
    E("E11_FR_scale2", "Pooled ranking: amplify FR predictions vs DE", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllag1"], R(1000), scale=2.0), S(["ret", "d1"], R(30))),
    E("E12_FR_scale4", "Amplify more", S(["level", "retlag1", "lvllead1", "lvllead2", "lvllag1"], R(1000), scale=4.0), S(["ret", "d1"], R(30))),
]
BY_ID = {e["id"]: e for e in EXPERIMENTS}
assert len(BY_ID) == len(EXPERIMENTS)
