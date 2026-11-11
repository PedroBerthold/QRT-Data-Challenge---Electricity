"""README banner. Run from the repo root: python -m docs.make_banner  -> docs/figures/00_banner.png"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from src.data import load, ROOT

for f in fm.findSystemFonts():
    if "Inter" in f or "DejaVuSansMono" in f:
        fm.fontManager.addfont(f)

BG, INK, MUTED, LINE = "#0b0e13", "#f1f3f5", "#8b93a0", "#1b212b"
GREEN, ORANGE, GREY = "#6fcf9f", "#f0a35e", "#9aa3b0"

_, _, days = load()[:3]
y = days["FR_NUCLEAR"].sort_index().dropna().to_numpy()   # chronological order (t), smooth by construction
lo, hi = np.percentile(y, [1, 99]); y = np.clip(y, lo, hi)
y = np.convolve(y, np.ones(5) / 5, mode="valid")        # light smoothing for readability

W, H = 16.4, 5.0
fig = plt.figure(figsize=(W, H), facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

for yy in np.linspace(0.3, 4.7, 10):                      # faint horizontal rules
    ax.plot([0, W], [yy, yy], color=LINE, lw=0.8, zorder=0)

ax.text(0.26, 4.52, "QRT DATA CHALLENGE  ·  CHALLENGE DATA ENS  ·  ELECTRICITY", color=MUTED, fontsize=11.5,
        family="DejaVu Sans Mono", va="center")
ax.text(0.26, 3.55, "Trend", color=INK, fontsize=50, fontweight="bold", family="Inter", va="center")
ax.text(2.38, 3.55, "beats level.", color=GREEN, fontsize=50, fontweight="bold", family="Inter", va="center")
ax.text(0.26, 2.52, "Explaining day-to-day electricity price moves in France and Germany.\n"
        "What is signal, what is order, what is noise.", color=MUTED, fontsize=15.5, family="Inter",
        va="center", linespacing=1.5)

# chronological series, right side
x = np.linspace(7.8, 15.5, len(y)); s = 1.85 + (y - y.min()) / (y.max() - y.min()) * 2.2
ax.plot(x, s, color="#3a4658", lw=1.6, zorder=2)
ax.plot([7.8, 15.5], [2.75, 2.75], color="#3f7d62", lw=1.1, ls=(0, (3, 4)), zorder=1)
ax.text(15.5, 1.52, "FR nuclear output, chronological order", color="#566072", fontsize=9, family="DejaVu Sans Mono",
        ha="right", va="center")

stats = [("59.68%", "public score, 5th", GREEN),
         ("+43.8 pt", "over the QRT benchmark", GREEN),
         ("0.98 vs 0", "autocorrelation: ID order vs DAY_ID", ORANGE),
         ("83", "experiments, failures kept", GREY)]
ax.plot([0.26, 15.9], [1.28, 1.28], color=LINE, lw=1.0)
for i, (big, small, col) in enumerate(stats):
    xx = 0.26 + i * 4.0
    ax.text(xx, 0.82 + 0.0, big, color=col, fontsize=29, fontweight="bold", family="Inter", va="center")
    ax.text(xx, 0.36, small, color=MUTED, fontsize=10.5, family="DejaVu Sans Mono", va="center")

fig.savefig(ROOT / "docs" / "figures" / "00_banner.png", dpi=130, facecolor=BG)
print("written 00_banner.png")
