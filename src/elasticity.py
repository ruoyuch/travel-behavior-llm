"""Own-fare elasticities, measured the same way for observed choices and every model.

On the given rows, fit one MNL to the source's choices (observed) or predicted
probabilities (models): constants, one fare coefficient per mode, time, headway and a
current-mode term. The elasticity of a mode's share with respect to its own fare is

    E = sum_n b_fare * fare_n * P_n (1 - P_n) / sum_n P_n      (fares in the MNL are /100)

GA travel-pass holders pay no train or Swissmetro fare, so their rows are excluded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import mnl

FARES = {"train_fare": (0, "TRAIN_COST", "B_COST_TRAIN"), "sm_fare": (1, "SM_COST", "B_COST_SM")}


def fare_rows(df: pd.DataFrame) -> np.ndarray:
    return (df.GA == 0).to_numpy()


def fare_elasticities(df: pd.DataFrame, W: np.ndarray) -> dict[str, float]:
    """df: rows with a fare (see fare_rows); W: observed labels (n,) or probabilities (n, 3)."""
    m = mnl.fit(df, W, headway=True, current=True, alt_cost=True, se=False)
    P = mnl.predict(m, df)
    out = {}
    for name, (alt, col, coef) in FARES.items():
        p = P[:, alt]
        out[name] = float((m["params"][coef] * df[col].to_numpy() / 100 * p * (1 - p)).sum() / p.sum())
    return out
