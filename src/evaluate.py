"""Scores with a paired respondent bootstrap, the results table and the headline figure.

Every source is scored on the same held-out rows with the same bootstrap draws of
respondents. Each score is an array: index 0 is the point estimate, the rest are
bootstrap draws.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.elasticity import fare_elasticities, fare_rows

SOURCES = ["Observed", "MNL", "XGBoost", "XGBoost+history", "C1", "C2"]
LABEL = {"Observed": "Observed", "MNL": "MNL", "XGBoost": "XGBoost",
         "XGBoost+history": "XGBoost (+ history)", "C1": "LLM (profile)", "C2": "LLM (profile + history)"}
TRAINING = {"Observed": "—", "MNL": "452 people, Q1–9", "XGBoost": "452 people, Q1–9",
            "XGBoost+history": "452 people, Q1–6 → Q7–9", "C1": "none", "C2": "none"}
TRAINED = {"MNL", "XGBoost", "XGBoost+history"}


def ci(a: np.ndarray) -> tuple[float, float]:
    d = a[1:][np.isfinite(a[1:])]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def bootstrap_index(df: pd.DataFrame, n_boot: int, seed: int) -> list[np.ndarray]:
    """Respondent-level bootstrap: each draw is the row index of the resampled people."""
    rng = np.random.default_rng(seed)
    ids = df.ID.to_numpy()
    by_id = {i: np.flatnonzero(ids == i) for i in np.unique(ids)}
    keys = np.array(list(by_id))
    return [np.concatenate([by_id[i] for i in rng.choice(keys, len(keys))]) for _ in range(n_boot)]


def score(df: pd.DataFrame, P: np.ndarray | None, boots) -> dict[str, np.ndarray]:
    """Fare elasticities, plus accuracy, macro recall and log-loss for models.
    P=None scores the observed choices."""
    def stats(d, Pb):
        y = d.y.to_numpy()
        m = fare_rows(d)
        s = fare_elasticities(d[m], y[m] if Pb is None else Pb[m])
        if Pb is not None:
            pred = Pb.argmax(1)
            s["accuracy"] = float((pred == y).mean())
            s["recall"] = float(np.mean([(pred[y == k] == k).mean() for k in range(3)]))
            s["logloss"] = float(-np.log(np.clip(Pb[np.arange(len(y)), y], 1e-6, 1)).mean())
        return s

    rows = [stats(df, P)] + [stats(df.iloc[i], None if P is None else P[i]) for i in boots]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


# ---------- table ----------
def _f(a, signed=True):
    s = (lambda x: f"{x:+.2f}") if signed else (lambda x: f"{x:.2f}")
    lo, hi = ci(a)
    return f"{s(a[0])} [{s(lo)}, {s(hi)}]".replace("-", "−")


def table(S: dict, n_people: int, n_fare: int) -> tuple[pd.DataFrame, str]:
    """Long numeric table (every metric with CI) and the markdown results table."""
    long = pd.DataFrame([{"source": src, "metric": k, "est": a[0], "lo": ci(a)[0], "hi": ci(a)[1]}
                         for src in SOURCES for k, a in S[src].items()])
    rows = [[LABEL[src], TRAINING[src],
             *(["—", "—"] if src == "Observed" else [_f(S[src]["accuracy"], False), _f(S[src]["recall"], False)]),
             _f(S[src]["train_fare"]), _f(S[src]["sm_fare"])] for src in SOURCES]
    head = ["Source", "Training data", "Accuracy", "Recall", "Train fare elasticity", "Swissmetro fare elasticity"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows]
    note = (f"\n{n_people} people, {3 * n_people} held-out choices (Q7–9). Recall: mean over the three modes. "
            f"Elasticity: own-fare, from an MNL fit to each source on the {n_fare} rows with a fare (no GA pass). "
            f"[95% CI, paired resamples of people.]\n")
    return long, "\n".join(lines) + "\n" + note


# ---------- figure ----------
GRAY, ACCENT, HUMAN_C = "#8a8880", "#eb6834", "#2a78d6"
INK, MUTED, GRID = "#1f1e1c", "#6b6a63", "#ecebe6"
TITLE_FS, NAME_FS, SUB_FS, TEXT_FS = 13, 11, 10, 10.5


def _box(ax, x, y, w, h, title, sub="", fc="white", ec=INK, tc=INK, sc=MUTED, ls="-"):
    """Rounded box; bold name, optional regular gray subtitle, block centered vertically."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.3, ls=ls, zorder=2))
    t_h, s_h, gap = 0.040, 0.037, 0.010            # line heights in axes units
    n_t, n_s = title.count("\n") + 1, (sub.count("\n") + 1) if sub else 0
    block = n_t * t_h + (gap + n_s * s_h if sub else 0)
    top = y + block / 2
    ax.text(x, top, title, ha="center", va="top", fontsize=NAME_FS, color=tc, fontweight="bold",
            zorder=3, linespacing=1.15)
    if sub:
        ax.text(x, top - n_t * t_h - gap, sub, ha="center", va="top", fontsize=SUB_FS, color=sc,
                zorder=3, linespacing=1.15)


def _arrow(ax, a, b, ls="-", color=MUTED):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.5, color=color,
                                 ls=ls, shrinkA=0, shrinkB=0, zorder=1))


def _diagram(ax):
    """Which information each source gets, and how all are scored."""
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1)
    ax.axis("off")
    # inputs
    _box(ax, 0.5, 0.925, 0.98, 0.11, "Traveler profile", "age, income, trip purpose, current mode",
         fc="#f4f3ef", ec="#c9c7bf")
    _box(ax, 0.5, 0.715, 0.44, 0.15, "History", "same person's choices,\nquestions 1–6", ls="--")
    # sources
    _box(ax, 0.17, 0.49, 0.34, 0.15, "LLM agent", "zero-shot", fc=ACCENT, ec=ACCENT,
         tc="white", sc="white")
    _box(ax, 0.83, 0.49, 0.34, 0.15, "MNL · XGBoost", "trained on\n452 people", fc=GRAY, ec=GRAY,
         tc="white", sc="white")
    # scoring
    _box(ax, 0.5, 0.27, 0.72, 0.15, "Held-out choices: questions 7–9",
         "same 300 people and trip, new fare,\ntime and headway for each mode",
         fc="#eaf2fc", ec=HUMAN_C)
    _box(ax, 0.25, 0.06, 0.44, 0.14, "Accuracy", "share predicted correctly", ec=HUMAN_C)
    _box(ax, 0.75, 0.06, 0.46, 0.14, "Train fare elasticity", "train share shift as\nthe train fare changes", ec=HUMAN_C)

    _arrow(ax, (0.12, 0.87), (0.12, 0.565))
    _arrow(ax, (0.88, 0.87), (0.88, 0.565))
    _arrow(ax, (0.34, 0.64), (0.28, 0.565), ls="--", color=INK)
    _arrow(ax, (0.66, 0.64), (0.72, 0.565), ls="--", color=INK)
    for x, ha in ((0.29, "right"), (0.71, "left")):
        ax.text(x, 0.605, "optional", ha=ha, va="center", fontsize=SUB_FS, color=MUTED, style="italic")
    _arrow(ax, (0.17, 0.415), (0.30, 0.345))
    _arrow(ax, (0.83, 0.415), (0.70, 0.345))
    _arrow(ax, (0.38, 0.195), (0.29, 0.13))
    _arrow(ax, (0.62, 0.195), (0.71, 0.13))
    ax.set_title("(a) Study design", fontsize=TITLE_FS, color=INK, loc="center", pad=10)


def _scatter(ax, S):
    """x = accuracy, y = train fare elasticity; observed value as the reference band."""
    from matplotlib.patches import FancyArrowPatch
    h = S["Observed"]["train_fare"]
    ax.axhspan(*ci(h), color=HUMAN_C, alpha=0.10, lw=0)
    ax.axhline(h[0], color=HUMAN_C, lw=1.4, ls="--")
    ax.text(0.99, h[0] - 0.02, f"observed ({h[0]:+.2f})".replace("-", "−"), color=HUMAN_C,
            fontsize=TEXT_FS, va="top", ha="right")
    ax.axhline(0, color=INK, lw=1.0, zorder=1)
    ax.text(0.99, 0.015, "no reaction", color=INK, fontsize=TEXT_FS, va="bottom", ha="right")

    pts = {}
    for src in SOURCES[1:]:
        x, y = S[src]["accuracy"], S[src]["train_fare"]
        c = GRAY if src in TRAINED else ACCENT
        (xl, xh), (yl, yh) = ci(x), ci(y)
        pts[src] = (x[0], y[0])
        ax.errorbar(x[0], y[0], xerr=[[x[0] - xl], [xh - x[0]]], yerr=[[y[0] - yl], [yh - y[0]]],
                    fmt="o", ms=9, color=c, mec="white", mew=1.6, elinewidth=1.6, capsize=0, zorder=3)
        ax.annotate(LABEL[src], textcoords="offset points", fontsize=TEXT_FS, color=c, **LABEL_POS[src](x, y))
    ax.add_patch(FancyArrowPatch(pts["C1"], pts["C2"], connectionstyle="arc3,rad=0.3", arrowstyle="-|>",
                                 mutation_scale=14, lw=1.6, color=ACCENT, shrinkA=9, shrinkB=9,
                                 ls=(0, (4, 2.5)), zorder=4))
    ax.text(0.618, -0.76, "+ history", color=ACCENT, fontsize=SUB_FS, ha="right", va="center")

    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(*Y_LIM)
    ax.set_xlabel("Accuracy on held-out choices", fontsize=TEXT_FS + 1, color=INK)
    ax.set_ylabel("Train fare elasticity", fontsize=TEXT_FS + 1, color=INK)
    ax.tick_params(colors=MUTED, labelsize=SUB_FS, length=0, pad=6)
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("(b) Accuracy vs. train fare elasticity", fontsize=TITLE_FS, color=INK, loc="center", pad=10)


Y_LIM = (-1.6, 0.15)
_right = lambda x, y: dict(xy=(ci(x)[1], y[0]), xytext=(5, 0), ha="left", va="center")
_left = lambda x, y: dict(xy=(ci(x)[0], y[0]), xytext=(-5, 0), ha="right", va="center")
_above = lambda x, y: dict(xy=(x[0], ci(y)[1]), xytext=(0, 5), ha="center", va="bottom")
_below_left = lambda x, y: dict(xy=(ci(x)[0], y[0]), xytext=(-5, -14), ha="right", va="top")
LABEL_POS = {"MNL": _above, "XGBoost": _right, "XGBoost+history": _right, "C1": _left, "C2": _below_left}


def figure(S: dict, path):
    """Headline: (a) study design and (b) accuracy vs train fare elasticity."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"wspace": 0.25})
    _diagram(a)
    _scatter(b, S)
    fig.savefig(path, dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
