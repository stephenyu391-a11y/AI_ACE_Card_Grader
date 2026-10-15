"""Step 2: check every card before any training, fix the split and compute the baselines.

  python check_data.py scan      open every picture once (the slow part; saved in out/step2/scan.npz)
  python check_data.py report    all checks, tables and picture sheets, from the saved scan

The checks are lettered as in the plan (turn 12 of the work log):
  A pictures open        B picture format      C grades valid       D grades make sense together
  E duplicates           F cards are found     G same as Score's examples
  H grade table          I kinds of card back  J change over certificate number
  K release year
"""
import csv
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data import AGES, DATA, KINDS, N_FOLDS, SCORE_EXAMPLES, age_of, fold_of, is_exam, load_cards, year_of
from metrics import _even_mix, grade_by_rule
from metrics import font as metrics_font
from scoring import FIELDS, REWARD, WEIGHTS, card_score
from views import REFERENCE_BOXES, REFERENCE_SIZE, _detect, find_cards, to_rgb

OUT = Path(__file__).resolve().parent / "out" / "step2"
EXAMPLES = Path(__file__).resolve().parents[1] / "score_examples"
USUAL = [tuple(box) for box in REFERENCE_BOXES]   # the checks assume 1412x857 pictures; check B confirms it
THUMB = (24, 32)            # width, height of the small card pictures used to group cards by look
BACK_GROUPS = 12
BACK_CENTRES = Path(__file__).resolve().parent / "back_centres.npy"
# Which back groups are which kind of card, named by eye in step 2 (out/step2/back_group_names.json).
KIND_OF_GROUP = {0: "modern Pokemon", 1: "modern Pokemon", 2: "modern Pokemon", 3: "older Pokemon", 8: "older Pokemon"}


def look(path: str) -> dict:
    """Everything the checks need from one picture."""
    out = {"path": path, "error": "", "width": 0, "height": 0, "mode": "", "pixel_md5": "", "min_alpha": 255,
           "detected": [2, 2], "box": [(0, 0, 0, 0)] * 2, "thumb": np.zeros((2, THUMB[1], THUMB[0], 3), np.uint8)}
    try:
        image = Image.open(io.BytesIO(Path(path).read_bytes()))
        image.load()
    except Exception as error:  # a picture that does not open is a finding, not a crash
        out["error"] = f"{type(error).__name__}: {error}"[:120]
        return out
    out.update(width=image.size[0], height=image.size[1], mode=image.mode)
    if image.mode == "RGBA":
        out["min_alpha"] = int(np.asarray(image)[..., 3].min())
    rgb = to_rgb(image)
    out["pixel_md5"] = hashlib.md5(rgb.tobytes()).hexdigest()
    found = _detect(rgb)
    out["box"] = find_cards(rgb)
    # how each side was decided: 0 found in the picture, 1 copied from the other side, 2 the usual place
    usual = all(tuple(a) == b for a, b in zip(out["box"], USUAL))
    out["detected"] = [0 if found and tuple(found[i]) == tuple(out["box"][i]) else 2 if usual else 1 for i in range(2)]
    out["thumb"] = np.stack([np.asarray(rgb.crop(box).resize(THUMB, Image.BILINEAR)) for box in out["box"]])
    return out


def scan(workers: int = 8) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(str(p) for p in DATA.glob("*/images/*.png"))
    results = []
    with Pool(workers) as pool:
        for i, result in enumerate(pool.imap(look, paths, chunksize=16), start=1):
            results.append(result)
            if i % 500 == 0 or i == len(paths):
                print(f"  {i}/{len(paths)} pictures", flush=True)
    np.savez_compressed(
        OUT / "scan.npz",
        path=[r["path"] for r in results], error=[r["error"] for r in results],
        size=np.array([(r["width"], r["height"]) for r in results]), mode=[r["mode"] for r in results],
        pixel_md5=[r["pixel_md5"] for r in results], min_alpha=np.array([r["min_alpha"] for r in results]),
        detected=np.array([r["detected"] for r in results]), box=np.array([r["box"] for r in results]),
        thumb=np.stack([r["thumb"] for r in results]))
    print(f"saved {OUT / 'scan.npz'}")


def read_rows() -> tuple[list[dict], Counter]:
    """Every CSV row with the reason it cannot be used (empty when it can)."""
    rows, columns = [], Counter()
    for table in sorted(DATA.glob("*/*.csv")):
        with open(table, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns[(table.parent.name, tuple(reader.fieldnames or ()))] += 1
            for row in reader:
                cert, problem, grades = (row.get("number") or "").strip(), "", {}
                for f in FIELDS:
                    try:
                        value = float(row[f])
                    except (KeyError, TypeError, ValueError):
                        problem = f"{f} is not a number: {row.get(f)!r}"
                        break
                    if value != int(value) or not 1 <= value <= 10:
                        problem = f"{f} is not a whole grade 1 to 10: {row[f]!r}"
                        break
                    grades[f] = int(value)
                if not cert.isdigit():
                    problem = f"certificate is not a number: {cert!r}"
                elif row.get("parse_status", "ok") != "ok":
                    problem = f"parse_status {row['parse_status']!r}"
                if not any((row.get(f) or "").strip() for f in FIELDS):
                    problem = "empty row: no grades at all"
                rows.append({"cert": cert, "batch": table.parent.name, "problem": problem, "year": year_of(row),
                             "year_text": (row.get("release_year") or "").strip(), "has_year_column": "release_year" in row,
                             "path": str(table.parent / "images" / f"card_{cert}.png"), **grades})
    return rows, columns


def sheet(items: list[tuple], target: Path, columns: int = 5, tile_w: int = 470) -> None:
    """Picture sheet: items are (picture path, caption) or (picture path, caption, boxes to draw)."""
    if not items:
        return
    tile_h, text_h = round(tile_w * 857 / 1412), 30
    lines = (len(items) + columns - 1) // columns
    page = Image.new("RGB", (columns * tile_w, lines * (tile_h + text_h)), "white")
    draw, font = ImageDraw.Draw(page), metrics_font(17)
    for i, (path, caption, *boxes) in enumerate(items):
        x, y = (i % columns) * tile_w, (i // columns) * (tile_h + text_h)
        picture = to_rgb(Image.open(path))
        for card_box in (boxes[0] if boxes else ()):
            ImageDraw.Draw(picture).rectangle([int(v) for v in card_box], outline=(0, 190, 0), width=5)
        page.paste(picture.resize((tile_w - 6, tile_h - 4), Image.BILINEAR), (x + 3, y + 2))
        draw.text((x + 6, y + tile_h + 3), caption, font=font, fill=(0, 0, 0))
    page.save(target, quality=88)


def group_backs(thumbs: np.ndarray, groups: int = BACK_GROUPS, rounds: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Plain k-means on the small pictures of the card backs. Returns (group of each card, distance to its centre).

    The centres are found once and saved in back_centres.npy. Later runs, with more
    cards, put each card in the nearest saved group, so a group keeps its number and its name.
    """
    x = thumbs.reshape(len(thumbs), -1).astype(np.float32) / 255.0
    if BACK_CENTRES.exists():
        centres = np.load(BACK_CENTRES)
        nearest = ((x ** 2).sum(1)[:, None] - 2 * x @ centres.T + (centres ** 2).sum(1)[None]).argmin(1)
        return nearest, np.sqrt(((x - centres[nearest]) ** 2).sum(1))
    rng = np.random.default_rng(0)
    centres = x[rng.choice(len(x), 1)]
    for _ in range(groups - 1):  # start from cards far from the centres chosen so far
        distance = ((x[:, None] - centres[None]) ** 2).sum(-1).min(1)
        centres = np.vstack([centres, x[rng.choice(len(x), p=distance / distance.sum())]])
    for _ in range(rounds):
        nearest = ((x ** 2).sum(1)[:, None] - 2 * x @ centres.T + (centres ** 2).sum(1)[None]).argmin(1)
        centres = np.stack([x[nearest == k].mean(0) if (nearest == k).any() else centres[k] for k in range(groups)])
    order = np.argsort(-np.bincount(nearest, minlength=groups))  # group 0 is the largest
    rank = np.empty(groups, int)
    rank[order] = np.arange(groups)
    np.save(BACK_CENTRES, centres[order])
    return rank[nearest], np.sqrt(((x - centres[nearest]) ** 2).sum(1))


def best_fixed(cards: list[dict]) -> dict:
    """The one answer per field that earns the most on these cards."""
    answer = {}
    for f in FIELDS:
        share = np.bincount([c[f] for c in cards], minlength=11)[1:] / len(cards)
        answer[f] = int(np.argmax(share @ REWARD)) + 1
    return answer


def score_of(cards: list[dict], answer_for) -> tuple[float, float]:
    """(natural mix, even mix) for a function that gives the answer for a card."""
    scores = np.array([card_score(answer_for(c), c) for c in cards])
    return float(scores.mean()), _even_mix(scores, np.array([c["grade"] for c in cards]))


def grade_table(cards: list[dict]) -> str:
    counts = {f: np.bincount([c[f] for c in cards], minlength=11)[1:] for f in FIELDS}
    lines = ["  grade " + "".join(f"{f:>11}" for f in FIELDS)]
    lines += [f"  {g:5d} " + "".join(f"{counts[f][g - 1]:11d}" for f in FIELDS) for g in range(1, 11)]
    return "\n".join(lines)


def report() -> None:
    scan_file = np.load(OUT / "scan.npz")
    seen = {p: i for i, p in enumerate(scan_file["path"].tolist())}
    error, size, mode = scan_file["error"], scan_file["size"], scan_file["mode"]
    pixel_md5, detected, box, thumb = scan_file["pixel_md5"], scan_file["detected"], scan_file["box"], scan_file["thumb"]
    rows, columns = read_rows()
    result = {}

    print("FILES")
    for (batch, names), _ in columns.items():
        print(f"  {batch}: columns {', '.join(names)}")
    print(f"  {len(rows)} rows, {len(seen)} pictures")

    print("\nA  PICTURES OPEN")
    no_picture = [r for r in rows if r["path"] not in seen]
    unreadable = [r for r in rows if r["path"] in seen and error[seen[r["path"]]]]
    row_paths = {r["path"] for r in rows}
    no_row = [p for p in seen if p not in row_paths]
    print(f"  rows without a picture: {len(no_picture)}   pictures without a row: {len(no_row)}   "
          f"pictures that do not open: {len(unreadable)}")
    for r in unreadable[:10]:
        print(f"    {r['cert']}: {error[seen[r['path']]]}")
    result["A"] = {"no_picture": [r["cert"] for r in no_picture], "no_row": no_row, "unreadable": [r["cert"] for r in unreadable]}

    if no_picture:   # which pictures still have to be brought to this machine
        numbers = sorted(int(r["cert"]) for r in no_picture if r["cert"].isdigit())
        with open(OUT / "A_rows_without_picture.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["cert", "batch"] + FIELDS)
            writer.writerows([r["cert"], r["batch"]] + [r.get(f, "") for f in FIELDS] for r in no_picture)
        print(f"    certificates {numbers[0]} to {numbers[-1]}; list in A_rows_without_picture.csv")
        low = sum(1 for r in no_picture if r.get("grade", 10) <= 6)
        print(f"    of these, final grade 6 or lower: {low}")

    print("\nB  PICTURE FORMAT")
    opened = [i for i in range(len(error)) if not error[i]]
    formats = Counter((int(size[i][0]), int(size[i][1]), str(mode[i])) for i in opened)
    for (w, h, m), n in formats.most_common():
        print(f"  {w}x{h} {m}: {n}")
    print(f"  pictures with see-through pixels: {int((scan_file['min_alpha'][opened] < 255).sum())}")
    odd_format = [r for r in rows if r["path"] in seen and not error[seen[r["path"]]]
                  and tuple(size[seen[r["path"]]]) != REFERENCE_SIZE]
    result["B"] = {f"{w}x{h} {m}": n for (w, h, m), n in formats.items()}

    print("\nC  GRADES VALID")
    bad = [r for r in rows if r["problem"]]
    print(f"  rows with an invalid value: {len(bad)}")
    for r in bad[:10]:
        print(f"    {r['cert']}: {r['problem']}")
    result["C"] = {r["cert"]: r["problem"] for r in bad}

    print("\nE  DUPLICATES")
    by_cert = defaultdict(list)
    for r in rows:
        if not r["problem"]:
            by_cert[r["cert"]].append(r)
    for rs in by_cert.values():   # a card keeps the year of whichever batch gives one, as in data.py
        rs[0]["year"] = next((r["year"] for r in rs if r["year"] is not None), None)
    twice = {c: rs for c, rs in by_cert.items() if len(rs) > 1}
    other_grades = [c for c, rs in twice.items() if any(any(r[f] != rs[0][f] for f in FIELDS) for r in rs[1:])]
    other_pixels = [c for c, rs in twice.items() if len({pixel_md5[seen[r["path"]]] for r in rs if r["path"] in seen}) > 1]
    print(f"  certificates in more than one batch: {len(twice)}   with different grades: {len(other_grades)}   "
          f"with different pixels: {len(other_pixels)}")
    usable = [rs[0] for rs in by_cert.values() if rs[0]["path"] in seen and not error[seen[rs[0]["path"]]]]
    by_pixels = defaultdict(list)
    for r in usable:
        by_pixels[pixel_md5[seen[r["path"]]]].append(r["cert"])
    same_picture = [certs for certs in by_pixels.values() if len(certs) > 1]
    print(f"  same picture under different certificates: {len(same_picture)} sets")
    for certs in same_picture[:10]:
        print(f"    {', '.join(certs)}")
    result["E"] = {"twice": len(twice), "other_grades": other_grades, "other_pixels": other_pixels, "same_picture": same_picture}
    print(f"  usable cards (one per certificate, picture opens, grades valid): {len(usable)}")

    print("\nD  GRADES MAKE SENSE TOGETHER")
    gap = Counter(r["grade"] - min(r[f] for f in FIELDS if f != "grade") for r in usable)
    for g in sorted(gap):
        print(f"  final grade minus lowest sub-grade = {g:+d}: {gap[g]} cards")
    by_rule = sum(r["grade"] == grade_by_rule(r) for r in usable)
    print(f"  final grade equals the sub-grade rule (lowest + 1 alone, lowest when tied): {by_rule} of {len(usable)}")
    odd_grades = [r for r in usable if r["grade"] - min(r[f] for f in FIELDS if f != "grade") not in (0, 1)]
    grades_of = lambda r: " ".join(f"{f[:2]}{r[f]}" for f in FIELDS)
    with open(OUT / "D_odd_grades.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cert"] + FIELDS)
        writer.writerows([r["cert"]] + [r[f] for f in FIELDS] for r in odd_grades)
    sheet([(r["path"], f"{r['cert']}  {grades_of(r)}") for r in odd_grades[:30]], OUT / "D_odd_grades.jpg")
    result["D"] = {"gap": {int(k): v for k, v in gap.items()}, "by_rule": by_rule, "odd": len(odd_grades)}

    print("\nF  CARDS ARE FOUND IN THE PICTURE")
    index = np.array([seen[r["path"]] for r in usable])
    how = detected[index]
    for side, name in enumerate(("front", "back")):
        print(f"  {name}: found in the picture {int((how[:, side] == 0).sum())}, copied from the other side "
              f"{int((how[:, side] == 1).sum())}, usual place {int((how[:, side] == 2).sum())}")
    boxes = box[index].astype(float)
    width, height = boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1]
    for side, name in enumerate(("front", "back")):
        q = lambda values: " / ".join(f"{v:.0f}" for v in np.percentile(values[:, side], [0, 1, 50, 99, 100]))
        print(f"  {name}: left {q(boxes[..., 0])}   top {q(boxes[..., 1])}   width {q(width)}   height {q(height)}"
              "   (lowest / 1% / middle / 99% / highest)")
    shift = np.abs(boxes - np.array(REFERENCE_BOXES, float)[None]).max(-1).max(-1)   # pixels away from the usual place
    odd_shape = ((width / height < 0.66) | (width / height > 0.78)).any(1)
    print(f"  cards more than 25 pixels from the usual place: {int((shift > 25).sum())}   not card-shaped: {int(odd_shape.sum())}")
    rng = np.random.default_rng(0)
    for name, members in (("copied", np.flatnonzero((how == 1).any(1))), ("usual_place", np.flatnonzero((how == 2).any(1))),
                          ("far_or_odd_shape", np.flatnonzero((shift > 25) | odd_shape))):
        picks = members if len(members) <= 60 else np.sort(rng.choice(members, 60, replace=False))
        sheet([(usable[i]["path"], f"{usable[i]['cert']}", box[index[i]]) for i in picks], OUT / f"F_{name}.jpg")
    result["F"] = {"how": {name: np.bincount(how[:, side], minlength=3).tolist() for side, name in enumerate(("front", "back"))},
                   "far": int((shift > 25).sum()), "odd_shape": int(odd_shape.sum()),
                   "usual_place_certs": [usable[i]["cert"] for i in np.flatnonzero((how == 2).any(1))]}

    print("\nG  SAME AS SCORE'S EXAMPLES")
    path_of = {r["cert"]: r for r in usable}
    result["G"] = {}
    for truth_file in sorted(EXAMPLES.glob("*.json")):
        cert = truth_file.stem.rsplit("-", 1)[1]
        truth = json.loads(truth_file.read_text())
        theirs = {"grade": truth["Header"]["card_grade"], **{f: truth["Grading_Features"][f"subgrade_{f}"] for f in FIELDS[:4]}}
        if cert not in path_of:
            print(f"  {cert}: not in our cards")
            result["G"][cert] = "not in our cards"
            continue
        ours = path_of[cert]
        a, b = Image.open(truth_file.with_suffix(".png")), Image.open(ours["path"])
        line = f"  {cert}: grades {'equal' if all(ours[f] == int(theirs[f]) for f in FIELDS) else 'DIFFERENT'}, "
        if a.size != b.size or a.mode != b.mode:
            line += f"format differs ({a.size} {a.mode} against {b.size} {b.mode})"
        else:
            difference = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16))
            line += (f"pixels: largest difference {int(difference.max())} of 255, average {difference.mean():.3f}, "
                     f"{(difference > 8).any(-1).mean():.2%} of pixels differ by more than 8")
        print(line)
        result["G"][cert] = line.strip()

    print("\nH  GRADE TABLE (cards per grade)")
    print(grade_table(usable))
    low = sum(r["grade"] <= 6 for r in usable)
    print(f"  final grade 6 or lower: {low} ({low / len(usable):.1%})   9 or 10: "
          f"{sum(r['grade'] >= 9 for r in usable) / len(usable):.1%}")
    result["H"] = {f: np.bincount([r[f] for r in usable], minlength=11)[1:].tolist() for f in FIELDS}

    print("\nI  KINDS OF CARD BACK (grouped by look)")
    group, distance = group_backs(thumb[index, 1])
    names = json.loads((OUT / "back_group_names.json").read_text()) if (OUT / "back_group_names.json").exists() else {}
    for k in range(BACK_GROUPS):
        members = np.flatnonzero(group == k)
        if not len(members):
            continue
        grades = np.array([usable[i]["grade"] for i in members])
        print(f"  group {k:2d}  {len(members):5d} cards  mean grade {grades.mean():.2f}  6 or lower {np.mean(grades <= 6):.0%}"
              f"  {names.get(str(k), '')}")
        typical = members[np.argsort(distance[members])]
        picks = list(typical[:10]) + list(typical[-10:])                # the most and the least typical of the group
        sheet([(usable[i]["path"], f"{usable[i]['cert']}  group {k}") for i in picks], OUT / f"I_back_group_{k:02d}.jpg",
              tile_w=330)
    result["I"] = {str(k): int((group == k).sum()) for k in range(BACK_GROUPS)}
    with open(KINDS, "w", newline="") as handle:   # read by data.py, for the age-shortcut numbers of every run
        writer = csv.writer(handle)
        writer.writerow(["cert", "back_group", "kind"])
        writer.writerows([usable[i]["cert"], int(group[i]), KIND_OF_GROUP.get(int(group[i]), "other cards")] for i in range(len(usable)))

    print("\nJ  CHANGE OVER CERTIFICATE NUMBER")
    numbers = np.array([int(r["cert"]) for r in usable])
    final = np.array([r["grade"] for r in usable])
    span = numbers.max() - numbers.min() + 1
    print(f"  certificates {numbers.min()} to {numbers.max()}: {len(numbers)} of {span} numbers present ({len(numbers) / span:.0%})")
    result["J"] = []
    for lo, hi in zip(*(lambda e: (e[:-1], e[1:]))(np.linspace(numbers.min(), numbers.max() + 1, 9).astype(int))):
        part = final[(numbers >= lo) & (numbers < hi)]
        print(f"  {lo}-{hi - 1}: {len(part):5d} cards  mean grade {part.mean():.2f}  6 or lower {np.mean(part <= 6):5.1%}"
              f"  grade 10 {np.mean(part == 10):5.1%}")
        result["J"].append({"from": int(lo), "to": int(hi - 1), "cards": len(part), "mean": float(part.mean()),
                            "low": float(np.mean(part <= 6)), "ten": float(np.mean(part == 10))})

    print("\nK  RELEASE YEAR")
    with_column = [r for r in rows if r["has_year_column"]]
    odd_year = [r for r in with_column if not r["problem"] and r["year"] is None]
    print(f"  rows in a CSV with a release_year column: {len(with_column)} of {len(rows)}   "
          f"valid rows there with no usable year: {len(odd_year)}")
    for r in odd_year[:10]:
        print(f"    {r['cert']}: {r['year_text']!r}")
    years = np.array([r["year"] or 0 for r in usable])
    print(f"  usable cards with a year: {int((years > 0).sum())} of {len(usable)}   "
          f"years {years[years > 0].min() if (years > 0).any() else '-'} to {years.max() if (years > 0).any() else '-'}")
    result["K"] = {}
    for age in AGES + ("unknown",):
        part = final[np.array([age_of(r["year"]) == age for r in usable])]
        if len(part):
            print(f"  released {age:10s} {len(part):5d} cards  mean grade {part.mean():.2f}  6 or lower {int((part <= 6).sum()):4d}"
                  f"  7 to 8 {int(((part >= 7) & (part <= 8)).sum()):4d}  9 to 10 {int((part >= 9).sum()):5d}")
            result["K"][age] = {"cards": len(part), "mean": float(part.mean()), "low": int((part <= 6).sum()),
                                "high": int((part >= 9).sum())}
    print("  kind of back (step 2, by eye) against release year:")
    for kind in sorted({KIND_OF_GROUP.get(int(g), "other cards") for g in group}):
        members = [age_of(usable[i]["year"]) for i in range(len(usable)) if KIND_OF_GROUP.get(int(group[i]), "other cards") == kind]
        print(f"    {kind:15s} " + "  ".join(f"{age} {members.count(age)}" for age in AGES + ("unknown",) if members.count(age)))

    print("\nTHE SPLIT (as train.py will see it)")
    everything = load_cards(include_exam=True, include_examples=True)
    exam = [c for c in everything if c["exam"]]
    example = [c for c in everything if c["cert"] in SCORE_EXAMPLES]
    working = load_cards()
    assert not {c["cert"] for c in working} & ({c["cert"] for c in exam} | SCORE_EXAMPLES), "exam cards leaked into the working set"
    assert all(is_exam(c["cert"]) == c["exam"] and fold_of(c["cert"]) == c["fold"] for c in everything)
    parts = [("exam", exam)] + [(f"group {k}", [c for c in working if c["fold"] == k]) for k in range(N_FOLDS)]
    print(f"  {len(everything)} cards = {len(exam)} exam + {len(working)} working + {len(example)} Score examples (never trained on)")
    for name, part in parts:
        grades = np.array([c["grade"] for c in part])
        print(f"  {name:8s} {len(part):5d} cards  mean grade {grades.mean():.2f}  6 or lower {np.mean(grades <= 6):5.1%}"
              f"  grade 10 {np.mean(grades == 10):5.1%}  always 10 scores {score_of(part, lambda c: dict.fromkeys(FIELDS, 10))[0]:.3f}")
    result["split"] = {name: len(part) for name, part in parts}

    print("\nBASELINES WITHOUT A MODEL (working set; natural mix / even mix)")
    fixed = best_fixed(working)
    baselines = {
        "always 10": score_of(working, lambda c: dict.fromkeys(FIELDS, 10)),
        f"best fixed answer {fixed}": score_of(working, lambda c: fixed),
    }
    for name, (natural, even) in baselines.items():
        print(f"  {name}: {natural:.4f} / {even:.4f}")
    for f in FIELDS:
        share = np.bincount([c[f] for c in working], minlength=11)[1:] / len(working)
        print(f"    {f:9s} weight {WEIGHTS[f]:.2f}  always 10 earns {share @ REWARD[:, 9]:.3f}  best fixed ({fixed[f]}) earns "
              f"{(share @ REWARD).max():.3f}")
    rule = np.mean([REWARD[c["grade"] - 1, grade_by_rule(c) - 1] for c in working])
    exact = np.mean([c["grade"] == grade_by_rule(c) for c in working])
    print(f"  sub-grade rule for the final grade, given the true sub-grades: earns {rule:.4f} of the final-grade field "
          f"(exactly right on {exact:.1%})")
    result["baselines"] = {"always_10": baselines["always 10"], "best_fixed": {"answer": fixed, "score": list(baselines.values())[1]},
                           "rule_given_true_subgrades": float(rule), "rule_exact": float(exact)}

    with open(OUT / "cards.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cert", "batch", "part", "back_group", "release_year"] + FIELDS)
        group_of = {usable[i]["cert"]: int(group[i]) for i in range(len(usable))}
        for c in sorted(everything, key=lambda c: int(c["cert"])):
            part = "score example" if c["cert"] in SCORE_EXAMPLES else "exam" if c["exam"] else f"group {c['fold']}"
            writer.writerow([c["cert"], c["path"].parents[1].name, part, group_of.get(c["cert"], ""), c["year"] or ""]
                            + [c[f] for f in FIELDS])
    (OUT / "report.json").write_text(json.dumps(result, indent=1, default=str))
    print(f"\nfiles in {OUT}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "scan":
        scan()
    else:
        report()
