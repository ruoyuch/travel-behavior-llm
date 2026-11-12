"""Ollama client: single-token answer, log-probs over the option letters, jsonl cache.

Every call is cached in a jsonl file keyed by a hash of (model, messages, options),
so reruns are resumable and free.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import time

import requests

FLOOR_LOGPROB = -30.0  # for letters absent from top_logprobs


class LLM:
    def __init__(self, cfg: dict, root: pathlib.Path):
        self.cfg = cfg
        self.url = cfg["endpoint"].rstrip("/") + "/api/chat"
        self.cache_path = root / cfg["cache_path"]
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.open():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec

    def _payload(self, messages):
        return {"model": self.cfg["model"], "messages": messages, "stream": False, "think": False,
                "options": {"temperature": self.cfg["temperature"], "num_predict": 1, "seed": 0},
                "logprobs": True, "top_logprobs": self.cfg["top_logprobs"]}

    def key(self, messages) -> str:
        blob = json.dumps({"model": self.cfg["model"], "messages": messages,
                           "temperature": self.cfg["temperature"], "n": 1}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _post(self, payload):
        r = requests.post(self.url, json=payload, timeout=600)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _parse(resp: dict, letters: list[str]) -> dict[str, float] | None:
        lp = resp.get("logprobs")
        if not lp:
            return None
        best = {L: FLOOR_LOGPROB for L in letters}
        for cand in lp[0].get("top_logprobs", []):
            t = cand["token"].strip().strip(".):").upper()
            if t in best:
                best[t] = max(best[t], cand["logprob"])
        m = max(best.values())
        z = sum(math.exp(v - m) for v in best.values())
        return {L: math.exp(v - m) / z for L, v in best.items()}

    def choose(self, messages, letters) -> dict:
        """Return cached record with 'probs' over letters."""
        k = self.key(messages)
        if k in self.cache:
            return self.cache[k]
        t0 = time.time()
        resp = self._post(self._payload(messages))
        probs = self._parse(resp, letters)
        if probs is None:
            raise RuntimeError("model returned no logprobs; this pipeline requires them")
        raw = resp["message"]["content"]
        rec = {"key": k, "probs": probs, "raw": raw, "method": "logprobs",
               "seconds": round(time.time() - t0, 3), "model": self.cfg["model"]}
        with self.cache_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        self.cache[k] = rec
        return rec
