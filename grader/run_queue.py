"""Run several experiments one after another, unattended.

The list of experiments is a text file. Each line is one run: the options for
train.py, starting with --name. Empty lines and lines starting with # are skipped.

    --name r1-base   --fold 0 --epochs 12 --lr 1e-5
    --name r2-lr-low --fold 0 --epochs 12 --lr 3e-6

What the queue does for every line:
  - skips it if out/<name>/result.json already exists (the run is finished);
  - otherwise starts train.py; an interrupted run continues where it stopped;
  - writes everything the run prints to out/<name>/log.txt;
  - goes on with the next line even if this one failed;
  - with --copy, sends the results (without the large model files) to another machine.

Start it inside tmux so that it keeps running after you log out:

    python run_queue.py runs.txt
    python run_queue.py runs.txt --copy sky@my-server:/home/sky/Work/44/grader/out/
    python run_queue.py runs.txt --then "sudo shutdown -h now"     # stop paying when the list is done
"""
import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def name_of(options: list[str]) -> str:
    if "--name" not in options or options.index("--name") + 1 >= len(options):
        raise SystemExit(f"every line needs --name: {' '.join(options)}")
    return options[options.index("--name") + 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("list", help="text file with one run per line")
    parser.add_argument("--copy", default="", help="rsync target for results, for example user@host:/path/out/")
    parser.add_argument("--with-models", action="store_true", help="also copy the model files (large)")
    parser.add_argument("--then", default="", help="shell command to run when the whole list is finished")
    args = parser.parse_args()

    lines = [l.strip() for l in Path(args.list).read_text().splitlines()]
    runs = [shlex.split(l) for l in lines if l and not l.startswith("#")]
    names = [name_of(r) for r in runs]
    if len(set(names)) != len(names):
        raise SystemExit("two lines have the same --name")
    print(f"{len(runs)} runs in the list: {', '.join(names)}", flush=True)

    outcome = {}
    for options, name in zip(runs, names):
        folder = OUT / name
        if (folder / "result.json").exists():
            outcome[name] = "already finished"
            print(f"[{name}] already finished, skipped", flush=True)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        print(f"[{name}] started {time.strftime('%Y-%m-%d %H:%M')}", flush=True)
        started = time.time()
        with open(folder / "log.txt", "a") as log:
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  train.py {' '.join(options)}\n")
            log.flush()
            code = subprocess.run([sys.executable, str(HERE / "train.py"), *options], stdout=log, stderr=subprocess.STDOUT, cwd=HERE).returncode
        minutes = (time.time() - started) / 60
        outcome[name] = f"finished in {minutes:.0f} min" if code == 0 else f"FAILED (exit code {code}), see {folder / 'log.txt'}"
        print(f"[{name}] {outcome[name]}", flush=True)

        if args.copy:
            command = ["rsync", "-a", "--relative"] + ([] if args.with_models else ["--exclude", "*.pt"]) + [f"./{name}", args.copy]
            copied = subprocess.run(command, cwd=OUT).returncode == 0
            print(f"[{name}] results {'copied to ' + args.copy if copied else 'NOT copied: rsync failed'}", flush=True)

    print("\nSummary")
    for name, text in outcome.items():
        print(f"  {name:24s} {text}")
    if args.then:
        print(f"running: {args.then}", flush=True)
        subprocess.run(args.then, shell=True)


if __name__ == "__main__":
    main()
