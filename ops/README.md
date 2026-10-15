# ops: preparing the server and renting a graphics card

The commands used by hand in turns 35 to 38 of the work log, as scripts. Run them from the 44 project folder.

## Once, on a new server

    VAST_API_KEY=<your key> ops/setup_local.sh

Python environment, PyTorch (processor version), the grader's libraries, the Vast.ai tool, the
pre-trained models if missing, the Vast.ai key (readable only by you) and an SSH key.

## A training session

| Step | Command | Note |
|---|---|---|
| 1. Find a machine | `ops/vast_find.sh` | Cheapest per unit of work first |
| 2. Rent it | `ops/vast_rent.sh <offer id>` | Billing starts. Waits until it runs, sets up SSH alias `gpu` |
| 3. Upload | `ops/upload.sh` | Code, models, libraries, 29 GB of pictures; checks every file |
| 4. Start the runs | `ops/start_runs.sh grader/session2.txt` | In tmux there; a watcher here copies results back and stops the machine at the end |
| 5. Watch | `ops/status.sh session2` | Passes so far, card use, credit |
| 6. Compare | `ops/report.sh r1-base own-views` | Uses the copies already here |
| 7. End | `ops/vast_end.sh destroy` | Or `stop` to keep the disk for a next session |

After a code fix during a session: `ops/upload.sh code`.

## Things learned in step 3

- A host can fail to start (turn 35, South Korea). If it is still loading after 10 minutes, `ops/vast_end.sh destroy` and rent another.
- A stopped machine may not restart: its card can be rented by someone else meanwhile (turn 36). Then destroy it and rent another; the upload takes 10 to 35 minutes.
- An RTX 5090 needs the CUDA 12.8 image set in `config.sh`.
- The picture sheets need a font; `grader/metrics.py` falls back to Pillow's own.
