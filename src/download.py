"""
Download swissmetro.dat (Biogeme example data).
No raw data is committed.
More details can be found at https://transp-or.epfl.ch/biogeme-2.5/swissmetro.pdf.
"""
import pathlib
import sys
import urllib.request

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    out = ROOT / cfg["data_path"]
    if out.exists():
        print(f"exists: {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {cfg['data_url']} -> {out}")
    urllib.request.urlretrieve(cfg["data_url"], out)
    n = sum(1 for _ in out.open())
    if n < 10000:
        sys.exit(f"unexpected file length ({n} lines)")
    print(f"ok: {n} lines")


if __name__ == "__main__":
    main()
