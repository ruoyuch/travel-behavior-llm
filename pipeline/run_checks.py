"""Pipeline checks (no LLM calls): MNL vs Biogeme, observed fare elasticities on other
samples, baseline accuracy.

Usage:
  uv run pipeline/run_checks.py [--n-respondents 30]
"""
from __future__ import annotations

import argparse

from src import baselines, data, evaluate, mnl

BIOGEME = {"ASC_TRAIN": -0.701, "ASC_CAR": -0.155, "B_TIME": -1.28, "B_COST": -1.08}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-respondents", type=int, default=None)
    args = ap.parse_args()
    cfg = data.load_config()

    df = data.load_filtered(cfg)
    print(f"filtered rows {len(df)}, respondents {df.ID.nunique()}")
    m = mnl.fit(df, df.y.to_numpy())
    print("\n1) MNL on the full filtered sample vs Biogeme 01logit")
    ok = True
    for k, ref in BIOGEME.items():
        v = m["params"][k]
        match = abs(v - ref) < 0.01
        ok &= match
        print(f"   {k:10s} {v:+.4f} (se {m['se'][k]:.4f})   Biogeme {ref:+.3f}   {'OK' if match else 'MISMATCH'}")
    print(f"   -> {'matches Biogeme' if ok else 'DOES NOT MATCH Biogeme'}")

    _, test, pool = data.split(cfg, args.n_respondents)
    print(f"\n2) Observed fare elasticities by sample ({cfg['n_boot']} respondent bootstraps)")
    test_people = df[df.ID.isin(set(test.ID))]
    for name, d in [("test people, Q7-9 (benchmark)", test), ("test people, Q1-9", test_people),
                    ("pool people, Q1-9", pool), ("all people, Q1-9", df)]:
        d = d.reset_index(drop=True)
        s = evaluate.score(d, None, evaluate.bootstrap_index(d, cfg["n_boot"], cfg["seed"]))
        print(f"   {name:30s} train {evaluate._f(s['train_fare'])}   Swissmetro {evaluate._f(s['sm_fare'])}")

    print("\n3) Baseline prediction on test rows")
    y = test.y.to_numpy()
    for name, p in baselines.predictions(pool, test, cfg["n_memory"], cfg["seed"]).items():
        print(f"   {name:16s} accuracy {(p.argmax(1) == y).mean():.3f}")


if __name__ == "__main__":
    main()
