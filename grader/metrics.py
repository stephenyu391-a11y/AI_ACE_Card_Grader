"""Turn the model's probabilities into answers, scores and things to look at.

evaluate()     all the numbers for a set of test cards
print_tables() the same numbers as readable tables
save_tables()  true grade against answered grade, one CSV per field
worst_sheet()  a picture sheet of the cards the model got most wrong
"""
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scoring import FIELDS, REWARD, best_answer, card_score, field_score
from views import to_rgb

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def font(size: int) -> ImageFont.ImageFont:
    """DejaVu Sans when the machine has it; otherwise Pillow's own font, so a missing font never stops a run."""
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default(size)
SHORT = {"surface": "S", "edges": "E", "corners": "C", "centering": "Ce", "grade": "G"}


def answers_from(probabilities: np.ndarray) -> dict:
    """Probabilities (5, 10) -> the answer with the highest expected score for each field."""
    return {f: best_answer(probabilities[i]) for i, f in enumerate(FIELDS)}


def grade_by_rule(answer: dict) -> int:
    """ACE's pattern: lowest sub-grade + 1 when it is lowest alone, the lowest itself when tied."""
    subs = sorted(answer[f] for f in FIELDS if f != "grade")
    return subs[0] if subs[0] == subs[1] else min(10, subs[0] + 1)


def _even_mix(scores: np.ndarray, grades: np.ndarray) -> float:
    """Average where every true final grade counts the same, however many cards it has."""
    return float(np.mean([scores[grades == g].mean() for g in np.unique(grades)]))


def shortcut_key(card: dict) -> tuple:
    """What the age-shortcut rival knows about a card: its release-year band and the look of its back."""
    return card.get("age", "unknown"), card.get("back_group", -1)


def answers_by_kind(cards: list[dict], fewest: int = 20) -> dict:
    """The age shortcut as a rival: for each age band and kind of back, the one answer that earns most on these cards.

    It never looks at damage. A model that scores no higher has only learned what kind of card it sees.
    Combinations with fewer than `fewest` cards fall back to the age band alone (key (age, None)).
    """
    def best(members):
        return {f: int(np.argmax(np.bincount([c[f] for c in members], minlength=11)[1:] @ REWARD)) + 1 for f in FIELDS}

    groups = {}
    for card in cards:
        groups.setdefault(shortcut_key(card), []).append(card)
        groups.setdefault((card.get("age", "unknown"), None), []).append(card)
    return {key: best(members) for key, members in groups.items() if key[1] is None or len(members) >= fewest}


# The cards on which the age shortcut is tested: (name, key of the card, value, lowest and highest final grade).
SHORTCUT_SETS = (("released to 2002, all", "age", "to 2002", 1, 10), ("released to 2002, grade 9-10", "age", "to 2002", 9, 10),
                 ("released 2021 on, grade 1-6", "age", "2021 on", 1, 6), ("not Pokemon, all", "kind", "other cards", 1, 10))


def evaluate(cards: list[dict], probabilities: np.ndarray, by_kind: dict | None = None) -> tuple[dict, list[dict]]:
    """Returns (summary numbers, one record per card with its answers and score).

    by_kind maps a test fold to answers_by_kind() of the cards outside that fold; with it the
    summary also holds what the age shortcut alone would score on the same cards.
    """
    records = []
    for card, p in zip(cards, probabilities):
        answer = answers_from(p)
        rival = (by_kind or {}).get(card["fold"], {})
        shortcut = rival.get(shortcut_key(card)) or rival.get((card.get("age", "unknown"), None))
        records.append({
            "card": card, "answer": answer, "score": card_score(answer, card),
            "score_kind_only": card_score(shortcut, card) if shortcut else float("nan"),
            "score_rule": card_score({**answer, "grade": grade_by_rule(answer)}, card),
            "score_always_10": card_score(dict.fromkeys(FIELDS, 10), card),
        })
    grades = np.array([c["grade"] for c in cards])
    column = lambda key: np.array([r[key] for r in records])
    summary = {
        "cards": len(cards),
        "score": float(column("score").mean()),
        "score_even_mix": _even_mix(column("score"), grades),
        "score_grade_by_rule": float(column("score_rule").mean()),
        "always_10": float(column("score_always_10").mean()),
        "always_10_even_mix": _even_mix(column("score_always_10"), grades),
    }
    for f in FIELDS:
        summary[f"field_{f}"] = float(np.mean([field_score(r["answer"][f], r["card"][f]) for r in records]))
    summary["kind_only"] = float(column("score_kind_only").mean())
    summary["shortcut"] = {}
    for name, key, value, low, high in SHORTCUT_SETS:
        chosen = np.array([c.get(key, "unknown") == value for c in cards]) & (grades >= low) & (grades <= high)
        if chosen.any():
            summary["shortcut"][name] = {"cards": int(chosen.sum()), "score": float(column("score")[chosen].mean()),
                                         "kind_only": float(column("score_kind_only")[chosen].mean()),
                                         "always_10": float(column("score_always_10")[chosen].mean())}
    # score for each true final grade: shows whether the model is blind to some grades
    summary["by_grade"] = {int(g): {"cards": int((grades == g).sum()), "score": float(column("score")[grades == g].mean()),
                                    "always_10": float(column("score_always_10")[grades == g].mean())}
                           for g in np.unique(grades)}
    # table[field][true - 1][answered - 1] = number of cards
    table = {f: np.zeros((10, 10), dtype=int) for f in FIELDS}
    for r in records:
        for f in FIELDS:
            table[f][r["card"][f] - 1, r["answer"][f] - 1] += 1
    summary["true_vs_answered"] = {f: t.tolist() for f, t in table.items()}
    return summary, records


def print_tables(summary: dict) -> None:
    print(f"  cards {summary['cards']}")
    print(f"  score, natural mix   {summary['score']:.3f}   (always 10: {summary['always_10']:.3f})")
    print(f"  score, even mix      {summary['score_even_mix']:.3f}   (always 10: {summary['always_10_even_mix']:.3f})")
    print(f"  final grade by rule  {summary['score_grade_by_rule']:.3f}")
    print("  per field            " + "  ".join(f"{f} {summary[f'field_{f}']:.3f}" for f in FIELDS))
    if summary.get("shortcut"):
        print(f"  age shortcut         a rival that knows only the release year and kind of card scores {summary['kind_only']:.3f} on these cards")
        print("  on the cards that test it (cards, model, rival, always 10)")
        for name, row in summary["shortcut"].items():
            print(f"    {name:29s}{row['cards']:5d}  {row['score']:.3f}  {row['kind_only']:.3f}  {row['always_10']:.3f}")
    print("  per true final grade (cards, model, always 10)")
    for grade, row in summary["by_grade"].items():
        print(f"    grade {int(grade):2d}  {row['cards']:5d}  {row['score']:.3f}  {row['always_10']:.3f}")


def save_tables(summary: dict, folder: Path) -> None:
    """One CSV per field: rows are the true grade, columns the answered grade."""
    for f, table in summary["true_vs_answered"].items():
        with open(folder / f"true_vs_answered_{f}.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true \\ answered"] + list(range(1, 11)))
            for grade, row in enumerate(table, start=1):
                writer.writerow([grade] + row)


def worst_sheet(records: list[dict], folder: Path, count: int = 50, columns: int = 5) -> None:
    """Picture sheet and CSV of the cards with the lowest score, worst first."""
    worst = sorted(records, key=lambda r: r["score"])[:count]
    if not worst:
        return
    tile_w, tile_h, text_h = 470, 285, 86
    rows = (len(worst) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_w, rows * (tile_h + text_h)), "white")
    draw = ImageDraw.Draw(sheet)
    text_font = font(19)
    grades = lambda values: " ".join(f"{SHORT[f]}{values[f]}" for f in FIELDS)
    with open(folder / "worst_cards.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cert", "score"] + [f"true_{f}" for f in FIELDS] + [f"answer_{f}" for f in FIELDS])
        for i, r in enumerate(worst):
            card, x, y = r["card"], (i % columns) * tile_w, (i // columns) * (tile_h + text_h)
            sheet.paste(to_rgb(Image.open(card["path"])).resize((tile_w - 6, tile_h - 4), Image.BILINEAR), (x + 3, y + 2))
            draw.text((x + 6, y + tile_h + 2), f"{card['cert']}   score {r['score']:.2f}", font=text_font, fill=(0, 0, 0))
            draw.text((x + 6, y + tile_h + 26), f"true    {grades(card)}", font=text_font, fill=(0, 0, 0))
            draw.text((x + 6, y + tile_h + 50), f"answer  {grades(r['answer'])}", font=text_font, fill=(170, 0, 0))
            writer.writerow([card["cert"], f"{r['score']:.3f}"] + [card[f] for f in FIELDS] + [r["answer"][f] for f in FIELDS])
    sheet.save(folder / "worst_cards.jpg", quality=88)
