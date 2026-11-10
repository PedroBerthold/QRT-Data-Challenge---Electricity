"""Data loading and the time axis.

Key structural fact (see outputs/diagnostics/audit.md): DAY_ID is shuffled, but
the ID column is chronological. DE rows carry IDs 0..931 and FR rows 932..2147,
each block sorted by date; every day has an FR row, so the rank of the FR row's
ID is the position of the day in time (t = 0..1215). Raw CSVs are never modified.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REDUNDANT = ["FR_DE_EXCHANGE", "DE_NET_IMPORT", "FR_NET_IMPORT"]  # exact negatives of other columns
META = ["ID", "DAY_ID", "COUNTRY"]
RETURNS = ["GAS_RET", "COAL_RET", "CARBON_RET"]


def load_raw():
    X = pd.read_csv(DATA / "X_train.csv")
    y = pd.read_csv(DATA / "y_train.csv")
    tr = X.merge(y, on="ID", how="left", validate="one_to_one")
    te = pd.read_csv(DATA / "X_test.csv")
    return tr, te


def numeric_columns(df):
    return [c for c in df.columns if c not in META + REDUNDANT + ["TARGET", "t", "set"]]


def time_index(tr, te):
    """DAY_ID -> t (chronological position), from the ID order of FR rows."""
    both = pd.concat([tr[META], te[META]])
    fr = both[both.COUNTRY == "FR"].sort_values("ID")
    assert fr.DAY_ID.is_unique and fr.DAY_ID.nunique() == both.DAY_ID.nunique()
    return pd.Series(np.arange(len(fr)), index=fr.DAY_ID.values, name="t")


def load():
    """Returns train rows, test rows (both with column t) and the day-level
    table of explanatory variables indexed by t (features are identical for
    the FR and DE rows of a day)."""
    tr, te = load_raw()
    tmap = time_index(tr, te)
    tr["t"] = tmap.loc[tr.DAY_ID].values
    te["t"] = tmap.loc[te.DAY_ID].values
    both = pd.concat([tr.drop(columns="TARGET"), te])
    days = both.drop_duplicates("t").set_index("t").sort_index()[numeric_columns(both)]
    return tr, te, days
