"""Figures of the README. Run from the repo root: python -m docs.make_figures
Needs the stored predictions (python main.py run --all) and writes docs/figures/*.png."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.stats import spearmanr, rankdata, norm
from src.data import load, load_raw, ROOT
from src.features import build
from src.models import make_model
from src.validation import random_day_folds

OUT = ROOT / "docs" / "figures"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#b9b8b2"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10})


def style(ax, title=None, xl=None, yl=None, grid="y"):
    if title:
        ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=10, fontweight="bold")
    ax.set_xlabel(xl or "")
    ax.set_ylabel(yl or "")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    if grid:
        ax.grid(True, axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=170)
    plt.close(fig)
    print("written", name)


def fig_progress():
    steps = [("Organisers' benchmark (public score)", 0.1586, NEUTRAL),
             ("Same-day levels, one model per country, rank target", 0.277, NEUTRAL),
             ("+ change of fundamentals since the previous day", 0.494, BLUE),
             ("+ France: commodity returns of t-1 and 3-day trend", 0.582, BLUE),
             ("+ Germany: residuals of neighbouring days, ensemble", 0.602, BLUE),
             ("Public leaderboard, single submission", 0.5968, ORANGE)]
    fig, ax = plt.subplots(figsize=(9, 3.9))
    y = np.arange(len(steps))[::-1]
    ax.barh(y, [s[1] for s in steps], color=[s[2] for s in steps], height=0.62)
    for yi, (lab, v, _) in zip(y, steps):
        ax.text(v + 0.008, yi, f"{v:.3f}", va="center", ha="left", color=INK, fontsize=10, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in steps], color=INK)
    ax.set_xlim(0, 0.7)
    style(ax, "From 0.16 to 0.60: what each idea added", "Spearman correlation (cross-validated unless stated)", grid="x")
    save(fig, "01_progress.png")


def fig_time_axis():
    rtr, rte = load_raw()
    tr, te, days = load()
    both = pd.concat([rtr.drop(columns="TARGET"), rte])
    by_day = both.drop_duplicates("DAY_ID").set_index("DAY_ID").sort_index()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.25, 1]})
    a = ax[0]
    seg = slice(400, 520)
    a.plot(np.arange(120), by_day.FR_CONSUMPTION.values[seg], color=NEUTRAL, linewidth=1.4)
    a.plot(np.arange(120), days.FR_CONSUMPTION.values[seg], color=BLUE, linewidth=2)
    a.text(119, days.FR_CONSUMPTION.values[seg][-1], "  sorted by ID", color=BLUE, va="center", fontsize=9.5, fontweight="bold")
    a.text(119, by_day.FR_CONSUMPTION.values[seg][-1], "  sorted by DAY_ID", color=MUTED, va="center", fontsize=9.5)
    a.set_xlim(0, 150)
    style(a, "French consumption, 120 consecutive rows", "row number", "standardised value")
    b = ax[1]
    lags = np.arange(1, 11)
    b.plot(lags, [days.FR_CONSUMPTION.autocorr(l) for l in lags], color=BLUE, linewidth=2, marker="o", markersize=6)
    b.plot(lags, [by_day.FR_CONSUMPTION.autocorr(l) for l in lags], color=NEUTRAL, linewidth=2, marker="o", markersize=6)
    b.text(10, days.FR_CONSUMPTION.autocorr(10) - 0.09, "ID order", color=BLUE, ha="right", fontsize=9.5, fontweight="bold")
    b.text(10, 0.07, "DAY_ID order", color=MUTED, ha="right", fontsize=9.5)
    b.set_ylim(-0.1, 1.05)
    style(b, "Does a row resemble the next one?", "distance between rows", "autocorrelation")
    save(fig, "02_time_axis.png")


def fig_de():
    tr, te, days = load()
    Y = tr.pivot(index="t", columns="COUNTRY", values="TARGET").reindex(days.index)
    x = days.DE_RESIDUAL_LOAD
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for a, z, xl, ttl in ((ax[0], x, "German residual load on the day", "Level of the day"),
                          (ax[1], x - x.shift(1), "German residual load, change since the previous day", "Change since the previous day")):
        m = Y.DE.notna() & z.notna()
        a.scatter(z[m], Y.DE[m].clip(-3, 3), s=13, color=BLUE, alpha=0.5, linewidths=0)
        style(a, f"{ttl}: Spearman {spearmanr(z[m], Y.DE[m]).correlation:.2f}", xl, "German target (clipped at ±3)" if a is ax[0] else None, grid="both")
    save(fig, "03_germany_level_vs_change.png")


def fig_fr():
    tr, te, days = load()
    Y = tr.pivot(index="t", columns="COUNTRY", values="TARGET").reindex(days.index)
    ks = list(range(-6, 7))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7), sharey=True)
    for a, var, name in ((ax[0], "FR_CONSUMPTION", "French consumption"), (ax[1], "FR_RESIDUAL_LOAD", "French residual load")):
        v = [spearmanr(days[var].shift(-k), Y.FR, nan_policy="omit").correlation for k in ks]
        a.bar(ks, v, color=[NEUTRAL if k < 0 else BLUE for k in ks], width=0.66)
        a.axhline(0, color=MUTED, linewidth=0.8)
        a.set_xticks(ks[::2])
        style(a, f"{name} on day t+k", "k (days before or after the target day)", "Spearman with the French target" if a is ax[0] else None)
    ax[0].text(-6.3, 0.028, "previous days", color=MUTED, fontsize=9.5)
    ax[0].text(0.7, 0.028, "target day and after", color=BLUE, fontsize=9.5, fontweight="bold")
    save(fig, "04_france_lead_lag.png")


def fig_target():
    tr, te, days = load()
    y = tr.TARGET.values
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    a = ax[0]
    bins = np.linspace(-4, 4, 81)
    a.hist(np.clip(y, -4, 4), bins=bins, color=BLUE, density=True)
    g = np.linspace(-4, 4, 300)
    a.plot(g, norm.pdf(g, y.mean(), y.std()), color=ORANGE, linewidth=2)
    a.text(1.25, 0.42, "normal curve with the\nsame mean and spread", color=ORANGE, fontsize=9.5, fontweight="bold")
    style(a, "The target: many tiny moves, a few huge ones", "target value (clipped at ±4)", "density")
    b = ax[1]
    r = rankdata(y) / len(y) - 0.5
    b.hist(r, bins=40, color=BLUE, density=True)
    style(b, "After the rank transform: equal weights", "rank of the target, rescaled to [-0.5, 0.5]", "density")
    save(fig, "05_target_and_ranks.png")


def fig_ridge():
    tr, te, days = load()
    alphas = [0.01, 0.1, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
    res = {}
    for c, blocks in (("FR", ["ret", "retlag1", "tr3"]), ("DE", ["ret", "d1"])):
        d = tr[tr.COUNTRY == c]
        X, y, t = build(blocks, days).loc[d.t].values, d.TARGET.values, d.t.values
        sc_cv, sc_in = [], []
        for a in alphas:
            oof = np.zeros((len(y), 2))
            for r, f, itr, iva in random_day_folds(t, 5, 2):
                m = make_model({"type": "ridge", "params": {"alpha": a}}).fit(X[itr], rankdata(y[itr]) / len(itr) - 0.5)
                oof[iva, r] = m.predict(X[iva])
            sc_cv.append(np.mean([spearmanr(y, oof[:, r]).correlation for r in range(2)]))
            m = make_model({"type": "ridge", "params": {"alpha": a}}).fit(X, rankdata(y) / len(y) - 0.5)
            sc_in.append(spearmanr(y, m.predict(X)).correlation)
        res[c] = (sc_cv, sc_in)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for a, c, name in ((ax[0], "FR", "France (weak signal)"), (ax[1], "DE", "Germany (strong signal)")):
        cv, ins = res[c]
        a.plot(alphas, ins, color=NEUTRAL, linewidth=2, marker="o", markersize=5)
        a.plot(alphas, cv, color=BLUE, linewidth=2, marker="o", markersize=5)
        k = int(np.argmax(cv))
        a.scatter([alphas[k]], [cv[k]], s=120, facecolor="none", edgecolor=INK, linewidth=1.5, zorder=5)
        a.annotate("on the data used for fitting", (alphas[1], ins[1]), xytext=(0, 9), textcoords="offset points", color=MUTED, fontsize=9.5)
        a.annotate("on unseen days", (alphas[1], cv[1]), xytext=(0, -17), textcoords="offset points", color=BLUE, fontsize=9.5, fontweight="bold")
        span = max(ins) - min(cv)
        a.set_ylim(min(cv) - 0.25 * span, max(ins) + 0.25 * span)
        a.set_xscale("log")
        style(a, name, "ridge penalty α (log scale): weak shrinkage → strong shrinkage", "Spearman" if c == "FR" else None, grid="both")
    save(fig, "06_regularisation.png")


def fig_leaderboard():
    lb = pd.read_csv(ROOT / "docs" / "leaderboard_scores_2026-11-10.csv").public_score.values
    fig, ax = plt.subplots(figsize=(9, 3.6))
    bins = np.arange(0.27, 0.64, 0.01)
    ax.hist(lb, bins=bins, color=NEUTRAL, rwidth=0.86)
    ax.axvline(0.5968, color=ORANGE, linewidth=2.5)
    ax.text(0.5968 - 0.005, ax.get_ylim()[1] * 0.86, "this project: 0.597\n5th of the snapshot", color=ORANGE, ha="right", fontsize=10, fontweight="bold")
    ax.text(0.295, ax.get_ylim()[1] * 0.86, "models that ignore\nthe order of the days", color=MUTED, ha="left", fontsize=9.5)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    style(ax, "Public leaderboard on 10 November 2026 (top 96 entries)", "public Spearman score", "number of participants")
    save(fig, "07_leaderboard.png")


def fig_pred_vs_actual():
    tr, te, days = load()
    oof = pd.read_csv(ROOT / "outputs" / "predictions" / "oof_K01_ensemble_main.csv").set_index("ID").loc[tr.ID].rep0.values
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, c, name in ((ax[0], "DE", "Germany"), (ax[1], "FR", "France")):
        m = (tr.COUNTRY == c).values
        rp, ry = rankdata(oof[m]) / m.sum(), rankdata(tr.TARGET.values[m]) / m.sum()
        a.scatter(rp, ry, s=11, color=BLUE, alpha=0.45, linewidths=0)
        a.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        style(a, f"{name}: Spearman {spearmanr(rp, ry).correlation:.2f}", "rank of the prediction (0 = lowest, 1 = highest)", "rank of the true value" if c == "DE" else None, grid="both")
    save(fig, "08_predicted_vs_actual_ranks.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_progress, fig_time_axis, fig_de, fig_fr, fig_target, fig_ridge, fig_leaderboard, fig_pred_vs_actual):
        f()
