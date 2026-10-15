"""One table of finished runs, to compare them on the same test cards.

  python compare_runs.py r1-base r0-frozen r5-1000 ...     (folder names under out/)
"""
import csv
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
FIELDS = ["surface", "edges", "corners", "centering", "grade"]


def main(names: list[str]) -> None:
    print(f"{'run':14s} {'train':>6s} {'best':>9s} {'natural':>8s} {'even':>6s} {'rival':>6s} {'by rule':>8s}  "
          + " ".join(f"{f[:7]:>7s}" for f in FIELDS) + "   min/pass  last-4 mean")
    for name in names:
        result = OUT / name / "result.json"
        if not result.exists():
            print(f"{name:14s} not finished")
            continue
        r = json.loads(result.read_text())
        history = list(csv.DictReader(open(OUT / name / "fold0" / "history.csv"))) if (OUT / name / "fold0" / "history.csv").exists() else []
        minutes = sum(float(h["seconds"]) for h in history) / max(1, len(history)) / 60
        # the best pass is picked on the test cards and flatters; the mean of the last four passes does not
        last = [float(h["score"]) for h in history[-4:]]
        steady = f"{sum(last) / len(last):.4f}" if last else "-"
        best = f"{r['best_pass'].get('0', '-')}/{r['epochs']}"
        train = r.get("train_cards") or "all"
        print(f"{name:14s} {str(train):>6s} {best:>9s} {r['score']:8.4f} {r['score_even_mix']:6.3f} {r.get('kind_only', float('nan')):6.3f} "
              f"{r['score_grade_by_rule']:8.4f}  " + " ".join(f"{r['field_' + f]:7.3f}" for f in FIELDS) + f"   {minutes:5.1f}     {steady}")
    print("\nage shortcut (cards, model, rival, always 10)")
    for name in names:
        result = OUT / name / "result.json"
        if result.exists():
            for set_name, row in json.loads(result.read_text()).get("shortcut", {}).items():
                print(f"  {name:14s} {set_name:30s} {row['cards']:5d}  {row['score']:.3f}  {row['kind_only']:.3f}  {row['always_10']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
