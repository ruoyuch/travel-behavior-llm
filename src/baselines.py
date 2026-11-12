"""Data-trained baselines, all trained on the fixed pool (452 people).

MNL: all 9 questions, with headway and current-mode terms.
XGBoost: all 9 questions, attributes and profile only.
XGBoost+history: questions 7-9 as outcomes, with MEM_* features (share of the same
person's questions 1-6 choosing each mode), built exactly as for the test rows, so a
row's own choice never enters its features.
Unavailable alternatives get probability 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from src import mnl
from src.data import MEM_COLS, availability

FEATURES = ["TRAIN_TT", "TRAIN_COST", "TRAIN_HE", "SM_TT", "SM_COST", "SM_HE",
            "CAR_TT", "CAR_COST", "CAR_AV_SP", "AGE", "MALE", "INCOME", "PURPOSE",
            "LUGGAGE", "WHO", "GA", "FIRST", "SURVEY"]


def _xgb(train: pd.DataFrame, test: pd.DataFrame, features: list[str], seed: int) -> np.ndarray:
    x = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, objective="multi:softprob", random_state=seed, n_jobs=4)
    x.fit(train[features], train.y)
    P = np.where(availability(test), x.predict_proba(test[features]), 0.0)
    return P / P.sum(axis=1, keepdims=True)


def predictions(pool: pd.DataFrame, test: pd.DataFrame, n_memory: int, seed: int) -> dict[str, np.ndarray]:
    """Return {source name: (n_test, 3) probabilities}."""
    m = mnl.fit(pool, pool.y.to_numpy(), headway=True, current=True, se=False)
    return {"MNL": mnl.predict(m, test),
            "XGBoost": _xgb(pool, test, FEATURES, seed),
            "XGBoost+history": _xgb(pool[pool.task > n_memory], test, FEATURES + MEM_COLS, seed)}
