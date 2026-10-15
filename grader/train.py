"""Train the grader on the ACE cards and measure it on cards it has not seen.

Two ways to run it:
  --freeze   the backbone only describes the views, and just the small head learns.
             Quick, works without a graphics card; used to prove every step runs.
  (default)  the backbone is trained too. This is the real training; needs a graphics card.

Each card belongs to one of 5 folds (fixed by its certificate number). A run with
--fold k trains on the other four and tests on fold k; --fold all does all five
and reports the score over every card.

During the real training the test cards are scored after every pass over the
data, and the best pass is the one that is kept. A run that was interrupted
continues from its last finished pass when the same command is given again
(--fresh starts over instead).

What a run leaves in out/<name>/
  result.json                 all numbers of the run
  true_vs_answered_<f>.csv    true grade against answered grade, per field
  worst_cards.jpg / .csv      the 50 cards the model got most wrong
  test_probabilities.npz      raw output, for averaging models later
  fold<k>/history.csv         loss and scores after every pass
  fold<k>/best.pt             the model of the best pass

Examples
  python train.py --freeze --fold all                                      # this machine
  python train.py --backbone dinov2-base --fold 0 --epochs 12 --batch 8    # graphics card
  python train.py --limit 16 --epochs 2 --batch 2 --fold 0 --name smoke    # smoke test
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from data import N_FOLDS, load_cards
from metrics import answers_by_kind, evaluate, print_tables, save_tables, worst_sheet
from model import Grader, views_to_tensors
from scoring import FIELDS, REWARD, WEIGHTS
from views import make_views

OUT = Path(__file__).resolve().parent / "out"
# Soft target: each row is the score every answer would earn for that true grade.
SOFT_TARGET = torch.tensor(REWARD / REWARD.sum(1, keepdims=True), dtype=torch.float32)


class Cards(Dataset):
    def __init__(self, cards: list[dict], train: bool):
        self.cards, self.train = cards, train

    def __len__(self) -> int:
        return len(self.cards)

    def __getitem__(self, i: int):
        card = self.cards[i]
        rng = np.random.default_rng() if self.train else None  # test pictures are never changed
        tensors = views_to_tensors(make_views(Image.open(card["path"]), rng))
        if self.train:
            gain = float(rng.uniform(0.95, 1.05))  # small brightness change
            tensors = tuple(t * gain for t in tensors)
        return (*tensors, torch.tensor([card[f] - 1 for f in FIELDS]))


def sampler_for(cards: list[dict]) -> WeightedRandomSampler:
    """Show rare final grades more often (square-root balancing: helps without overdoing it)."""
    counts = np.bincount([c["grade"] for c in cards], minlength=11)
    weights = [1.0 / np.sqrt(counts[c["grade"]]) for c in cards]
    return WeightedRandomSampler(weights, num_samples=len(cards), replacement=True)


def loss_for(scores: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against the soft targets, fields weighted as in the subnet's score."""
    log_p = torch.log_softmax(scores.float(), dim=-1)                 # (B, 5, 10)
    per_field = -(SOFT_TARGET.to(scores.device)[truth] * log_p).sum(-1)  # (B, 5)
    weights = torch.tensor([WEIGHTS[f] for f in FIELDS], device=scores.device)
    return (per_field * weights).sum(1).mean()


def save_safely(state: dict, path: Path) -> None:
    """Write to a side file first, so a crash while saving never leaves a broken file."""
    torch.save(state, path.with_suffix(".tmp"))
    os.replace(path.with_suffix(".tmp"), path)


@torch.inference_mode()
def describe(model: Grader, cards: list[dict], device: str, workers: int) -> torch.Tensor:
    """Frozen backbone: features (N, 18, dim) for every card, computed once."""
    model.backbone.eval()
    loader = DataLoader(Cards(cards, train=False), batch_size=4, num_workers=workers)
    out = []
    for i, (full, corners, edges, _) in enumerate(loader):
        out.append(model.backbone(full.to(device), corners.to(device), edges.to(device)).float().cpu())
        print(f"  described {min(len(cards), (i + 1) * 4)}/{len(cards)} cards", end="\r", flush=True)
    print()
    return torch.cat(out)


def train_head(model: Grader, features: torch.Tensor, truth: torch.Tensor, epochs: int, lr: float) -> None:
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=0.05)
    model.head.train()
    for _ in range(epochs):
        for index in torch.randperm(len(features)).split(32):
            optimizer.zero_grad()
            loss_for(model.head(features[index]), truth[index]).backward()
            optimizer.step()
    model.head.eval()


@torch.inference_mode()
def predict(model: Grader, cards: list[dict], device: str, workers: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(Cards(cards, train=False), batch_size=4, num_workers=workers)
    out = [torch.softmax(model(full.to(device), corners.to(device), edges.to(device)).float(), -1).cpu()
           for full, corners, edges, _ in loader]
    return torch.cat(out).numpy()


def train_all(model: Grader, train_cards: list[dict], test_cards: list[dict], args, device: str, folder: Path) -> np.ndarray:
    """The real training. Tests after every pass, keeps the best pass, can continue after an interruption.

    Returns the test probabilities of the best pass.
    """
    loader = DataLoader(Cards(train_cards, train=True), batch_size=args.batch, sampler=sampler_for(train_cards),
                        num_workers=args.workers, drop_last=len(train_cards) > args.batch)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.head_lr},
    ], weight_decay=0.05)
    steps = max(1, args.epochs * len(loader))
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[args.lr, args.head_lr], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    select = "score_even_mix" if args.select == "even" else "score"

    first_epoch, best, history = 0, {"value": -1.0, "epoch": 0}, []
    last = folder / "last.pt"
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        if state["epochs"] != args.epochs or state["steps_per_epoch"] != len(loader) \
                or state.get("own_views", False) != args.own_views:
            raise SystemExit(f"{last} was made with other settings; use the same command or add --fresh")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        scaler.load_state_dict(state["scaler"])
        first_epoch, best, history = state["epoch"], state["best"], state["history"]
        print(f"  continuing after pass {first_epoch} (best so far {best['value']:.3f} at pass {best['epoch']})")

    for epoch in range(first_epoch, args.epochs):
        model.train()
        total, started = 0.0, time.time()
        for full, corners, edges, truth in loader:
            optimizer.zero_grad()
            with torch.autocast(device, enabled=device == "cuda"):
                scores = model(full.to(device), corners.to(device), edges.to(device))
            loss = loss_for(scores, truth.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            total += loss.item()

        probabilities = predict(model, test_cards, device, args.workers)
        summary, _ = evaluate(test_cards, probabilities)
        row = {"pass": epoch + 1, "train_loss": round(total / len(loader), 4), "score": round(summary["score"], 4),
               "score_even_mix": round(summary["score_even_mix"], 4),
               **{f: round(summary[f"field_{f}"], 4) for f in FIELDS}, "seconds": round(time.time() - started)}
        history.append(row)
        is_best = summary[select] > best["value"]
        if is_best:
            best = {"value": summary[select], "epoch": epoch + 1}
            save_safely(model.state_dict(), folder / "best.pt")
            np.save(folder / "best_probabilities.npy", probabilities)
        with open(folder / "history.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        save_safely({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(),
                     "scaler": scaler.state_dict(), "epoch": epoch + 1, "epochs": args.epochs, "own_views": args.own_views,
                     "steps_per_epoch": len(loader), "best": best, "history": history}, last)
        print(f"  pass {epoch + 1}/{args.epochs}  loss {row['train_loss']:.4f}  score {row['score']:.3f}  "
              f"even mix {row['score_even_mix']:.3f}  {row['seconds']}s{'  <- best' if is_best else ''}", flush=True)

    print(f"  best pass: {best['epoch']} of {args.epochs}")
    last.unlink(missing_ok=True)  # only needed for continuing; it is three times the size of the model
    return np.load(folder / "best_probabilities.npy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="dinov2-base")
    parser.add_argument("--fold", default="all", help="0..4 or 'all'")
    parser.add_argument("--freeze", action="store_true", help="train only the head")
    parser.add_argument("--epochs", type=int, default=None, help="passes over the data")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5, help="backbone learning rate")
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--own-views", action="store_true",
                        help="corners, edges and centering are judged only from their own views (against the age shortcut)")
    parser.add_argument("--select", choices=["natural", "even"], default="natural", help="which score picks the best pass")
    parser.add_argument("--train-cards", type=int, default=0, help="train on only this many cards (data-amount test)")
    parser.add_argument("--limit", type=int, default=0, help="use only this many cards in total (smoke test)")
    parser.add_argument("--fresh", action="store_true", help="ignore earlier progress of this run and start over")
    parser.add_argument("--name", default=None, help="folder under out/ for this run")
    args = parser.parse_args()
    args.epochs = args.epochs or (60 if args.freeze else 12)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    cards = load_cards()
    if args.limit:
        cards = cards[:args.limit]
    folds = range(N_FOLDS) if args.fold == "all" else [int(args.fold)]
    run = OUT / (args.name or f"{args.backbone}-{'frozen' if args.freeze else 'finetune'}")
    run.mkdir(parents=True, exist_ok=True)
    print(f"{len(cards)} cards, backbone {args.backbone}, device {device}, folds {list(folds)}, run {run.name}")

    features = None
    if args.freeze:
        features = describe(Grader(args.backbone, args.own_views).to(device), cards, device, args.workers)
    truth = torch.tensor([[c[f] - 1 for f in FIELDS] for c in cards])

    tested, probabilities, best_passes, by_kind = [], [], {}, {}
    for fold in folds:
        by_kind[fold] = answers_by_kind([c for c in cards if c["fold"] != fold])
        train_index = [i for i, c in enumerate(cards) if c["fold"] != fold and not c["bad_cut"]]
        test_index = [i for i, c in enumerate(cards) if c["fold"] == fold]
        if not train_index or not test_index:
            continue
        if args.train_cards:
            order = np.random.default_rng(args.seed).permutation(len(train_index))[:args.train_cards]
            train_index = [train_index[i] for i in sorted(order)]
        fold_cards = [cards[i] for i in test_index]
        folder = run / f"fold{fold}"
        if args.fresh and folder.exists():
            for old in folder.iterdir():
                old.unlink()
        folder.mkdir(exist_ok=True)
        print(f"fold {fold}: train {len(train_index)}, test {len(test_index)}")

        done = folder / "done.json"
        if done.exists():   # this fold was finished in an earlier, interrupted run
            p = np.load(folder / "best_probabilities.npy")
            best_passes[fold] = json.loads(done.read_text())["best_pass"]
            print("  already finished, results reused")
        else:
            torch.manual_seed(args.seed + fold)
            model = Grader(args.backbone, args.own_views).to(device)
            if args.freeze:
                train_head(model, features[train_index].to(device), truth[train_index].to(device), args.epochs, args.head_lr)
                with torch.inference_mode():
                    p = torch.softmax(model.head(features[test_index].to(device)), -1).cpu().numpy()
                torch.save(model.head.state_dict(), folder / "head.pt")
                np.save(folder / "best_probabilities.npy", p)
                best_passes[fold] = args.epochs
            else:
                p = train_all(model, [cards[i] for i in train_index], fold_cards, args, device, folder)
                best_passes[fold] = max(csv.DictReader(open(folder / "history.csv")),
                                        key=lambda r: float(r["score_even_mix" if args.select == "even" else "score"]))["pass"]
            done.write_text(json.dumps({"best_pass": best_passes[fold]}))
        tested += fold_cards
        probabilities.append(p)

    probabilities = np.concatenate(probabilities)
    summary, records = evaluate(tested, probabilities, by_kind)
    summary.update(backbone=args.backbone, own_views=args.own_views, frozen=args.freeze, epochs=args.epochs, folds=list(folds), seed=args.seed,
                   lr=args.lr, head_lr=args.head_lr, batch=args.batch, train_cards=args.train_cards,
                   best_pass={str(k): v for k, v in best_passes.items()})
    (run / "result.json").write_text(json.dumps(summary, indent=1))
    np.savez(run / "test_probabilities.npz", certs=[c["cert"] for c in tested], probabilities=probabilities)
    save_tables(summary, run)
    worst_sheet(records, run)
    print("\nALL TESTED CARDS")
    print_tables(summary)
    print(f"  results in {run}")


if __name__ == "__main__":
    main()
