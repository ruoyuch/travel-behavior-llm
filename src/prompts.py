"""Prompt builders for agent conditions.

C1: traveler profile (incl. current mode) + today's options (time, cost, headway),
    with mode names and Swiss context.
C2: C1 + the respondent's memory tasks (options shown + choice made).

Option order is randomized once per respondent (src.data.option_order) and kept
across all of that person's prompts, so a letter means the same mode throughout.
Codebook follows the Swissmetro data description (Biogeme docs).
"""
from __future__ import annotations

import pandas as pd

LETTERS = ["A", "B", "C"]
MODE_NAME = {0: "Train", 1: "Swissmetro", 2: "Car"}

AGE = {1: "24 or younger", 2: "25-39", 3: "40-54", 4: "55-65", 5: "over 65"}
INCOME = {0: "under 50,000 CHF", 1: "under 50,000 CHF", 2: "50,000-100,000 CHF", 3: "over 100,000 CHF"}
PURPOSE = {1: "commuting", 3: "business"}
LUGGAGE = {0: "no luggage", 1: "one piece of luggage", 3: "several pieces of luggage"}
WHO = {1: "you pay yourself", 2: "your employer pays", 3: "you and your employer split the cost"}

SYSTEM = ("You are role-playing a specific traveler answering a travel survey. "
          "Choose the option this traveler would most likely choose. "
          "Reply with a single letter only.")


def letter_map(row: pd.Series, order: list[int]) -> dict[str, int]:
    """Letters assigned to available alternatives in the respondent's order."""
    alts = [a for a in order if a != 2 or row.CAR_AV_SP == 1]
    return {LETTERS[i]: a for i, a in enumerate(alts)}


def _context(r: pd.Series) -> str:
    lines = []
    if r.AGE in AGE:
        lines.append(f"- Age: {AGE[r.AGE]}")
    lines.append(f"- Sex: {'male' if r.MALE == 1 else 'female'}")
    inc = INCOME.get(r.INCOME)
    if inc:
        lines.append(f"- Annual household income: {inc}")
    lines.append(f"- Trip purpose: {PURPOSE.get(r.PURPOSE, 'other')}")
    lines.append(f"- Luggage: {LUGGAGE.get(r.LUGGAGE, 'no luggage')}")
    if r.WHO in WHO:
        lines.append(f"- Who pays: {WHO[r.WHO]}")
    lines.append("- Holds a GA (Swiss annual rail pass, covers train and Swissmetro fares): "
                 + ("yes" if r.GA == 1 else "no"))
    lines.append(f"- You currently make this trip by {MODE_NAME[int(r.CURRENT)].lower()}.")
    return "\n".join(lines)


def _option_line(r: pd.Series, alt: int, letter: str) -> str:
    tt, co, he = {0: (r.TRAIN_TT, r.TRAIN_COST, r.TRAIN_HE),
                  1: (r.SM_TT, r.SM_COST, r.SM_HE),
                  2: (r.CAR_TT, r.CAR_COST, None)}[alt]
    s = f"{letter}) {MODE_NAME[alt]}: travel time {int(tt)} min, cost {int(co)} CHF"
    return s + (f", departs every {int(he)} min" if he is not None else "")


def _options(r: pd.Series, lmap: dict[str, int]) -> str:
    return "\n".join(_option_line(r, a, L) for L, a in lmap.items())


def _memory(mem: pd.DataFrame, order: list[int]) -> str:
    blocks = []
    for k, (_, m) in enumerate(mem.iterrows(), 1):
        lmap = letter_map(m, order)
        chosen = {a: L for L, a in lmap.items()}[int(m.y)]
        blocks.append(f"Past choice {k}:\n{_options(m, lmap)}\nYou chose: {chosen}")
    return "\n\n".join(blocks)


def build(condition: str, row: pd.Series, mem: pd.DataFrame, order: list[int]) -> dict:
    """Return {'messages', 'letters', 'letter_to_alt'} for one test row."""
    lmap = letter_map(row, order)
    parts = [f"You are a traveler making a trip in Switzerland. About you:\n{_context(row)}"]
    if condition == "C2":
        parts.append("Earlier in this survey you made these choices:\n\n" + _memory(mem, order))
    parts.append("Now choose for this trip:\n" + _options(row, lmap))
    parts.append(f"Which option do you choose? Answer with one letter: {', '.join(lmap)}.")
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": "\n\n".join(parts)}],
            "letters": list(lmap), "letter_to_alt": lmap}
