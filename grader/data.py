"""Read every ACE batch under Data/ into one list of cards.

Each batch is a folder holding one CSV (columns number, surface, centering,
edges, corners, grade, and optionally release_year, parse_status, ...) and
images/card_<number>.png. A row whose picture is not there yet is skipped.

The cards are divided once, by certificate number, so the division never
changes as batches are added:
  exam            1 card in 10. No experiment may use these; they are opened once, at the end.
  test groups     the rest, in N_FOLDS groups ("folds").
  Score examples  the cards Score published in its starter pack. Never trained on.
"""
import csv
import hashlib
from pathlib import Path

from scoring import FIELDS

DATA = Path(__file__).resolve().parents[1] / "Data"
EXCLUDED = Path(__file__).resolve().parent / "excluded.csv"   # cert,reason: cards step 2 found unusable
BAD_CUT = Path(__file__).resolve().parent / "bad_cut.csv"     # cert,reason: the card finder misses these; tested but not trained on
KINDS = Path(__file__).resolve().parent / "kinds.csv"         # cert,back_group,kind: written by check_data.py
N_FOLDS = 5
AGES = ("to 2002", "2003-2015", "2016-2020", "2021 on")       # release-year bands; "unknown" when the CSV has no year
EXAM_ONE_IN = 10
SCORE_EXAMPLES = {"225539", "226435", "226491", "450736", "451692", "452012"}


def fold_of(cert: str) -> int:
    """Test fold of a certificate. Fixed by its number, so it never changes as batches are added."""
    return int(hashlib.md5(cert.encode()).hexdigest(), 16) % N_FOLDS


def is_exam(cert: str) -> bool:
    """Whether a certificate belongs to the locked final exam. Independent of its fold."""
    return int(hashlib.md5(f"exam:{cert}".encode()).hexdigest(), 16) % EXAM_ONE_IN == 0


def year_of(row: dict) -> int | None:
    """Release year from a CSV row; None when the column is missing, empty or not a sensible year."""
    text = (row.get("release_year") or "").strip()
    return int(text) if text.isdigit() and 1900 <= int(text) <= 2100 else None


def age_of(year: int | None) -> str:
    if year is None:
        return "unknown"
    return AGES[0] if year <= 2002 else AGES[1] if year <= 2015 else AGES[2] if year <= 2020 else AGES[3]


def _listed(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path, newline="") as handle:
        return {row["cert"]: row["reason"] for row in csv.DictReader(handle)}


def load_cards(data: Path = DATA, verbose: bool = True, include_exam: bool = False,
               include_examples: bool = False) -> list[dict]:
    """One dict per usable card: cert, path, fold, exam, bad_cut, back_group, kind, year, age and the five grades.

    year is the card's release year from the CSV column release_year (None when absent) and age
    its band, one of AGES or "unknown". Neither is ever shown to the model: the miner does not
    receive a year. They are used to measure the age shortcut and to balance training.

    kind is "modern Pokemon", "older Pokemon" or "other cards", judged from the look of the
    card back in step 2 ("unknown" for a card check_data.py has not seen yet).

    A card marked bad_cut is one whose views are cut wrongly. Such cards stay in
    the tests, because the miner will meet them too, but train.py does not learn from them.

    Exam cards and Score's example cards are left out unless asked for, so that
    no training or experiment can touch them by accident.
    """
    cards = {}
    skipped = {"bad row": 0, "no picture": 0, "duplicate": 0, "excluded": 0, "exam": 0, "Score example": 0}
    unusable, bad_cut = _listed(EXCLUDED), _listed(BAD_CUT)
    kinds = {}
    if KINDS.exists():
        with open(KINDS, newline="") as handle:
            kinds = {row["cert"]: (int(row["back_group"]), row["kind"]) for row in csv.DictReader(handle)}
    for table in sorted(data.glob("*/*.csv")):
        with open(table, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                cert = row.get("number", "").strip()
                try:
                    values = [float(row[f]) for f in FIELDS]
                    grades = {f: int(v) for f, v in zip(FIELDS, values)} if all(v == int(v) for v in values) else None
                except (KeyError, TypeError, ValueError):
                    grades = None
                if not cert or grades is None or row.get("parse_status", "ok") != "ok" \
                        or not all(1 <= g <= 10 for g in grades.values()):
                    skipped["bad row"] += 1
                    continue
                path = table.parent / "images" / f"card_{cert}.png"
                if cert in unusable:
                    skipped["excluded"] += 1
                elif not path.exists():
                    skipped["no picture"] += 1
                elif cert in cards:
                    skipped["duplicate"] += 1
                    if cards[cert]["year"] is None:   # an earlier batch had no year column; take it from this one
                        cards[cert].update(year=year_of(row), age=age_of(year_of(row)))
                else:
                    cards[cert] = {"cert": cert, "path": path, "fold": fold_of(cert), "exam": is_exam(cert),
                                   "bad_cut": cert in bad_cut, "back_group": kinds.get(cert, (-1, "unknown"))[0],
                                   "kind": kinds.get(cert, (-1, "unknown"))[1], "year": year_of(row),
                                   "age": age_of(year_of(row)), **grades}
    kept = []
    for card in cards.values():
        if card["cert"] in SCORE_EXAMPLES and not include_examples:
            skipped["Score example"] += 1
        elif card["exam"] and not include_exam:
            skipped["exam"] += 1
        else:
            kept.append(card)
    if verbose:
        print(f"{len(kept)} cards loaded; left out {skipped}")
    return kept
