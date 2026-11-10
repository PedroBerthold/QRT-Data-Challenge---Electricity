"""Day-level feature blocks built on the true time axis (t).

A feature set is a list of block names; each block maps the day table (index t,
29 explanatory variables) to new columns. Blocks only use explanatory variables
(of train and test days alike, as available at prediction time), never targets.
"""
import numpy as np
import pandas as pd
from src.data import RETURNS

BLOCKS = {}


def block(name):
    def deco(fn):
        BLOCKS[name] = fn
        return fn
    return deco


def _phys(D):  # physical variables (everything but commodity returns)
    return D[[c for c in D.columns if c not in RETURNS]]


@block("level")
def level(D):
    return D.copy()


def _diff(k):
    return lambda D: (_phys(D) - _phys(D).shift(k)).add_prefix(f"D{k}_")


for _k in (1, 2, 3, 5):
    BLOCKS[f"d{_k}"] = _diff(_k)  # change since k days ago


@block("lead1")
def lead1(D):  # change from today to tomorrow
    return (_phys(D).shift(-1) - _phys(D)).add_prefix("LEAD1_")


@block("lead2")
def lead2(D):  # change from tomorrow to the day after
    return (_phys(D).shift(-2) - _phys(D).shift(-1)).add_prefix("LEAD2_")


@block("lag1d")
def lag1d(D):  # yesterday's change
    return (_phys(D).shift(1) - _phys(D).shift(2)).add_prefix("LAG1D_")


def _dev(w):
    def fn(D):
        P = _phys(D)
        past = sum(P.shift(k) for k in range(1, w + 1)) / w
        return (P - past).add_prefix(f"DEV{w}_")
    return fn


for _w in (3, 7, 14):
    BLOCKS[f"dev{_w}"] = _dev(_w)  # deviation from the mean of the previous w days


def _retlag(k):
    tag = f"RETLAG{k}_" if k > 0 else f"RETLEAD{-k}_"
    return lambda D: D[RETURNS].shift(k).add_prefix(tag)


for _k in (1, 2, 3, -1, -2):
    BLOCKS[f"retlag{_k}" if _k > 0 else f"retlead{-_k}"] = _retlag(_k)


def _lvlshift(k):
    tag = f"LVLLAG{k}_" if k > 0 else f"LVLLEAD{-k}_"
    return lambda D: _phys(D).shift(k).add_prefix(tag)


for _k in (1, 2, -1, -2, -3):
    BLOCKS[f"lvllag{_k}" if _k > 0 else f"lvllead{-_k}"] = _lvlshift(_k)  # levels of other days


@block("ret")
def ret(D):
    return D[RETURNS].copy()


@block("phys")
def phys(D):
    return _phys(D).copy()


def build(blocks, days):
    return pd.concat([BLOCKS[b](days) for b in blocks], axis=1)
