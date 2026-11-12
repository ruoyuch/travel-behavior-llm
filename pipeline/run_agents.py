"""Query LLM agents on every test row for each condition.

Resumable: every call is cached in cache/llm_calls.jsonl keyed by prompt hash.
Writes results/agent_probs.csv (one row per test row x condition).

Usage:
  uv run pipeline/run_agents.py                      # full design from config.yaml
  uv run pipeline/run_agents.py --n-respondents 30   # demo
  uv run pipeline/run_agents.py --pilot 50           # first 50 test rows only, prints timing
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

from src import data, prompts
from src.llm import LLM

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-respondents", type=int, default=None)
    ap.add_argument("--pilot", type=int, default=None, help="only the first N test rows")
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--show-prompts", action="store_true")
    args = ap.parse_args()

    cfg = data.load_config()
    conds = args.conditions or cfg["conditions"]
    memory, test, _ = data.split(cfg, args.n_respondents)
    if args.pilot:
        test = test.head(args.pilot)
    mem_by_id = {i: g.sort_values("task") for i, g in memory.groupby("ID")}

    jobs = [(c, r) for c in conds for _, r in test.iterrows()]

    llm = LLM(cfg, ROOT)
    if args.show_prompts:
        for c in conds:
            r = test.iloc[0]
            p = prompts.build(c, r, mem_by_id[r.ID], data.option_order(r.ID, cfg["seed"]))
            print(f"===== {c} =====\n[system] {p['messages'][0]['content']}\n[user]\n{p['messages'][1]['content']}\n")

    out, new_secs, t0 = [], [], time.time()
    for k, (c, r) in enumerate(jobs, 1):
        order = data.option_order(r.ID, cfg["seed"])
        p = prompts.build(c, r, mem_by_id[r.ID], order)
        cached = llm.key(p["messages"]) in llm.cache
        rec = llm.choose(p["messages"], p["letters"])
        if not cached:
            new_secs.append(rec["seconds"])
        pr = [0.0, 0.0, 0.0]  # by alternative: train, SM, car
        for L, a in p["letter_to_alt"].items():
            pr[a] = rec["probs"].get(L, 0.0)
        raw_alt = p["letter_to_alt"].get(rec["raw"].strip().upper()[:1])
        out.append({"condition": c, "row_id": int(r.row_id), "ID": int(r.ID), "task": int(r.task),
                    "p_train": pr[0], "p_sm": pr[1], "p_car": pr[2], "method": rec["method"],
                    "raw": rec["raw"], "raw_mode": None if raw_alt is None else prompts.MODE_NAME[raw_alt],
                    "order": "".join(prompts.MODE_NAME[a][0] for a in order),
                    "seconds": rec["seconds"]})
        if k % 25 == 0 or k == len(jobs):
            el = time.time() - t0
            print(f"[{k}/{len(jobs)}] {el:.0f}s elapsed, new calls {len(new_secs)}"
                  + (f", {np.mean(new_secs):.2f}s/call" if new_secs else ""), flush=True)

    res = pd.DataFrame(out)
    if not args.pilot:
        (ROOT / cfg["results_dir"]).mkdir(exist_ok=True)
        res.to_csv(ROOT / cfg["results_dir"] / "agent_probs.csv", index=False)
    # Pilot report
    print("\nmethod counts:", res.method.value_counts().to_dict())
    print("answer letter:", res.raw.value_counts().to_dict())
    print("answer mode:", res.raw_mode.value_counts().to_dict())
    print("mean max prob by condition:", res.groupby("condition")[["p_train", "p_sm", "p_car"]]
          .apply(lambda d: d.max(axis=1).mean()).round(3).to_dict())
    if new_secs:
        spc = float(np.median(new_secs))
        full = len(cfg["conditions"]) * cfg["n_respondents"] * (9 - cfg["n_memory"])
        print(f"median s/call (uncached): {spc:.2f}; projected full run ({full} calls): {full * spc / 3600:.2f} h")


if __name__ == "__main__":
    sys.exit(main())
