"""Vectorized multinomial logit (scipy), Biogeme 01logit specification plus optional terms.

V_TRAIN = ASC_TRAIN + B_TIME*TRAIN_TT + B_COST*TRAIN_COST
V_SM    =             B_TIME*SM_TT    + B_COST*SM_COST
V_CAR   = ASC_CAR   + B_TIME*CAR_TT   + B_COST*CAR_CO
(times and costs /100). Optional terms:
  B_HEADWAY * headway (train, SM)
  B_CURRENT * 1[alternative is the respondent's current mode]   (SURVEY)
  alt_cost: one cost coefficient per alternative (B_COST_TRAIN, B_COST_SM, B_COST_CAR)
fit() takes observed choices (integer labels) or predicted probabilities (n, 3).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from src.data import design

EXTRA = [("headway", "B_HEADWAY"), ("current", "B_CURRENT")]


def _feature_stack(X, alt_cost=False):
    """(k, n, 3) derivative of V wrt each parameter."""
    n = X["time"].shape[0]
    asc_t = np.zeros((n, 3)); asc_t[:, 0] = 1
    asc_c = np.zeros((n, 3)); asc_c[:, 2] = 1
    cost = [X["cost"] * np.eye(3)[k] for k in range(3)] if alt_cost else [X["cost"]]
    return np.stack([asc_t, asc_c, X["time"], *cost] + [X[k] for k, _ in EXTRA if k in X])


def _probs_from_F(beta, F, av):
    V = np.tensordot(beta, F, axes=1)
    V = np.where(av, V, -np.inf)
    V = V - V.max(axis=1, keepdims=True)
    e = np.exp(V)
    return e / e.sum(axis=1, keepdims=True)


def _negll_grad(beta, F, av, W):
    P = _probs_from_F(beta, F, av)
    ll = (W * np.log(np.where(av, P, 1.0))).sum()
    Fbar = (F * P[None]).sum(axis=2)  # (k, n)
    g = (W[None] * F).sum(axis=(1, 2)) - (Fbar * W.sum(axis=1)[None]).sum(axis=1)
    return -ll, -g


def fit(df, y, headway=False, current=False, alt_cost=False, se=True) -> dict:
    spec = {"headway": headway, "current": current}
    X = design(df, **spec)
    F = _feature_stack(X, alt_cost)
    y = np.asarray(y)
    W = y.astype(float) if y.ndim == 2 else np.eye(3)[y.astype(int)]
    k = F.shape[0]
    res = minimize(_negll_grad, np.zeros(k), args=(F, X["av"], W), jac=True, method="BFGS",
                   options={"gtol": 1e-7, "maxiter": 2000})
    cost = ["B_COST_TRAIN", "B_COST_SM", "B_COST_CAR"] if alt_cost else ["B_COST"]
    names = ["ASC_TRAIN", "ASC_CAR", "B_TIME", *cost] + [n for key, n in EXTRA if key in X]
    out = {"params": dict(zip(names, res.x)), "beta": res.x, "spec": spec, "alt_cost": alt_cost}
    if se:
        H = np.zeros((k, k)); h = 1e-5
        for i in range(k):
            e = np.zeros(k); e[i] = h
            H[:, i] = (_negll_grad(res.x + e, F, X["av"], W)[1]
                       - _negll_grad(res.x - e, F, X["av"], W)[1]) / (2 * h)
        out["se"] = dict(zip(names, np.sqrt(np.diag(np.linalg.inv((H + H.T) / 2)))))
    return out


def predict(model: dict, df) -> np.ndarray:
    X = design(df, **model["spec"])
    return _probs_from_F(model["beta"], _feature_stack(X, model["alt_cost"]), X["av"])
