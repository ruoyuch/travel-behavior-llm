"""Load, filter and split the Swissmetro SP data.

Alternatives (CHOICE coding): 1 = Train, 2 = Swissmetro (SM), 3 = Car.
Internal index: 0 = Train, 1 = SM, 2 = Car.

Design:
- tasks 1..n_memory are memory, the rest are test (default 6 + 3);
- respondents are shuffled once (seed); a sample of n is the first n of that order,
  so smaller samples are nested in larger ones;
- the training pool is every respondent outside the first cfg["n_respondents"]
  of the order, so it is the same for every sample size up to that.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
ALTS = ["TRAIN", "SM", "CAR"]
MEM_COLS = ["MEM_TRAIN", "MEM_SM", "MEM_CAR"]


def load_config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def load_filtered(cfg: dict | None = None) -> pd.DataFrame:
    """Biogeme 01logit filter: PURPOSE in {1,3}, CHOICE != 0. GA holders pay 0 for train and SM."""
    cfg = cfg or load_config()
    df = pd.read_csv(ROOT / cfg["data_path"], sep="\t")
    df = df[df.PURPOSE.isin([1, 3]) & (df.CHOICE != 0)].copy()
    df["row_id"] = df.index  # stable id from the raw file
    ga = df.GA == 1
    # Effective costs shown to the traveler (and used everywhere downstream).
    df["TRAIN_COST"] = np.where(ga, 0, df.TRAIN_CO)
    df["SM_COST"] = np.where(ga, 0, df.SM_CO)
    df["CAR_COST"] = df.CAR_CO
    df["TRAIN_AV_SP"] = df.TRAIN_AV * (df.SP != 0)
    df["CAR_AV_SP"] = df.CAR_AV * (df.SP != 0)
    df["y"] = df.CHOICE - 1  # 0 train, 1 SM, 2 car
    # SURVEY: 0 = recruited on the train, 1 = recruited among car users -> current mode.
    df["CURRENT"] = np.where(df.SURVEY == 1, 2, 0)
    # Task order within respondent follows file order.
    df["task"] = df.groupby("ID").cumcount() + 1
    df["n_tasks"] = df.groupby("ID").ID.transform("size")
    return df.reset_index(drop=True)


def availability(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([df.TRAIN_AV_SP, df.SM_AV, df.CAR_AV_SP]).astype(bool)


def design(df: pd.DataFrame, headway: bool = False, current: bool = False) -> dict[str, np.ndarray]:
    """Scaled (/100) attribute matrices, shape (n, 3), plus optional extra terms."""
    out = {
        "time": np.column_stack([df.TRAIN_TT, df.SM_TT, df.CAR_TT]) / 100.0,
        "cost": np.column_stack([df.TRAIN_COST, df.SM_COST, df.CAR_COST]) / 100.0,
        "av": availability(df),
    }
    if headway:
        out["headway"] = np.column_stack([df.TRAIN_HE, df.SM_HE, np.zeros(len(df))]) / 100.0
    if current:
        cur = np.zeros((len(df), 3))
        cur[np.arange(len(df)), df.CURRENT.to_numpy()] = 1.0
        out["current"] = cur
    return out


def respondent_order(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """All 9-task respondents in one fixed shuffled order."""
    ids = np.sort(df[df.n_tasks == 9].ID.unique())
    return np.random.default_rng(cfg["seed"]).permutation(ids)


def add_memory_features(df: pd.DataFrame, n_memory: int) -> pd.DataFrame:
    """Per respondent: share of memory tasks (1..n_memory) choosing each mode."""
    mem = df[df.task <= n_memory]
    shares = (pd.get_dummies(mem.y).reindex(columns=[0, 1, 2], fill_value=0)
              .groupby(mem.ID).mean())
    shares.columns = MEM_COLS
    return df.drop(columns=MEM_COLS, errors="ignore").merge(shares, left_on="ID", right_index=True, how="left")


def split(cfg: dict | None = None, n_respondents: int | None = None):
    """Return (memory, test, pool). All frames carry MEM_* features.

    memory: tasks 1..n_memory of sampled respondents; test: the remaining tasks.
    pool: all rows of respondents outside the first cfg["n_respondents"] (fixed training pool).
    """
    cfg = cfg or load_config()
    n = n_respondents or cfg["n_respondents"]
    if n > cfg["n_respondents"]:
        raise ValueError("n_respondents exceeds the configured maximum; the pool would overlap")
    df = add_memory_features(load_filtered(cfg), cfg["n_memory"])
    order = respondent_order(df, cfg)
    sample = set(order[:n])
    pool_ids = set(order[cfg["n_respondents"]:])
    s = df[df.ID.isin(sample)]
    memory = s[s.task <= cfg["n_memory"]].reset_index(drop=True)
    test = s[s.task > cfg["n_memory"]].reset_index(drop=True)
    pool = df[df.ID.isin(pool_ids)].reset_index(drop=True)
    return memory, test, pool


def option_order(resp_id: int, seed: int) -> list[int]:
    """Per-respondent random order of the three alternatives (same across all their prompts)."""
    return list(np.random.default_rng([seed, int(resp_id)]).permutation(3))
