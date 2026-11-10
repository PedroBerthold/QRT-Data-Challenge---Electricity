"""Validation schemes.

Primary: repeated K-fold on days, random. Reason (audit): test days are randomly
interleaved with train days along time (527 train/test alternations, 511 expected
at random), and both rows of a day are always on the same side. A random day
split is therefore the faithful simulation of the competition split; a
chronological split would simulate a problem we do not have.

Supplementary: contiguous time blocks, to check stability across periods.
"""
import numpy as np

N_SPLITS, N_REPEATS = 5, 5


def random_day_folds(t, n_splits=N_SPLITS, n_repeats=N_REPEATS):
    t = np.asarray(t)
    days = np.unique(t)
    for r in range(n_repeats):
        perm = np.random.RandomState(r).permutation(days)
        for f, chunk in enumerate(np.array_split(perm, n_splits)):
            va = np.isin(t, chunk)
            yield r, f, np.where(~va)[0], np.where(va)[0]


def time_block_folds(t, n_blocks=5):
    t = np.asarray(t)
    days = np.unique(t)
    for f, chunk in enumerate(np.array_split(days, n_blocks)):
        va = np.isin(t, chunk)
        yield 0, f, np.where(~va)[0], np.where(va)[0]
