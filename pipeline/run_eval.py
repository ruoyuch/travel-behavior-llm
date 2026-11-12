"""Score every source on the held-out choices; write the results table and figure.

Outputs (results/): table.md (the table), metrics.csv (every number with its CI),
fig_headline.png (the figure). No LLM calls: reads results/agent_probs.csv.

Usage:
  uv run pipeline/run_eval.py                      # full design
  uv run pipeline/run_eval.py --n-respondents 30   # demo (must match pipeline/run_agents.py)
"""
from __future__ import annotations

import argparse
import pathlib

import pandas as pd

from src import baselines, data, evaluate
from src.elasticity import fare_rows

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-respondents", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    args = ap.parse_args()
    cfg = data.load_config()
    n_boot = args.n_boot or cfg["n_boot"]
    out = ROOT / cfg["results_dir"]

    _, test, pool = data.split(cfg, args.n_respondents)
    P = baselines.predictions(pool, test, cfg["n_memory"], cfg["seed"])
    agent = pd.read_csv(out / "agent_probs.csv").set_index(["condition", "row_id"])
    for c in cfg["conditions"]:
        try:
            P[c] = agent.loc[c].loc[test.row_id, ["p_train", "p_sm", "p_car"]].to_numpy()
        except KeyError:
            raise SystemExit(f"{c}: rows missing; rerun pipeline/run_agents.py")

    boots = evaluate.bootstrap_index(test, n_boot, cfg["seed"])
    S = {}
    for src in evaluate.SOURCES:
        print(f"scoring {src} ...", flush=True)
        S[src] = evaluate.score(test, P.get(src), boots)

    long, md = evaluate.table(S, test.ID.nunique(), int(fare_rows(test).sum()))
    long.to_csv(out / "metrics.csv", index=False)
    (out / "table.md").write_text(md)
    evaluate.figure(S, out / "fig_headline.png")
    print("\n" + md)
    print(f"wrote {cfg['results_dir']}/table.md, metrics.csv, fig_headline.png")


if __name__ == "__main__":
    main()
