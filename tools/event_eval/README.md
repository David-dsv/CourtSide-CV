# event_eval — bounce↔shot confusion thermometer

`tools/bounce_eval` scores bounces in isolation. `event_eval` measures the thing
the user actually cares about: **confusion** between floor BOUNCEs and racket HITs
— a hit emitted as a bounce (`confusion_H→B`, the bug) or vice-versa.

## What's here

- **`event_eval.py`** — pure comparator. Takes predicted events (each tagged
  `BOUNCE`/`HIT` with a frame) + the two GTs (a bounce file + a shots file) and
  runs ONE greedy 1-1 cross matcher (tol `round(0.15*fps)`), producing the 2×3
  confusion matrix. Deterministic §5.2 disambiguation (closest frame; tie → the
  predicted label decides; one-to-one). `confusion_H→B` is metric #1.
- **`run_demo3.py`** — BEFORE/AFTER driver: baseline (prod Kalman path) vs the
  methodology (`vision.events.classify_events`) on the committed demo3 cache.
- **`sweep_demo3.py`** — config sweep + the 0/0-invariance check (how many configs
  structurally hold confusion 0/0 — the firewall is not threshold-tuned).

## GT format

- bounces: `tests/fixtures/bounces/<clip>.bounces.json` (bounce_io schema).
- shots:   `tests/fixtures/shots/<clip>.shots.json` —
  `{fps, width, height, shots:[{frame, x, y, type}]}`.
- `frame` is the clip-local `demo_frame` (== the pipeline's segment frame index).

## Usage

```bash
# BEFORE/AFTER on the demo3 benchmark
python tools/event_eval/run_demo3.py

# find the best config under the hard constraint confusion_H->B == 0
python tools/event_eval/sweep_demo3.py

# score a hand-written / exported predictions file
python tools/event_eval/event_eval.py \
    --pred preds.json \
    --truth-bounces tests/fixtures/bounces/tennis_demo3.bounces.json \
    --truth-shots   tests/fixtures/shots/tennis_demo3.shots.json
```

Regression: `tests/test_event_confusion_regression.py` asserts H→B==0, B→H==0,
bounce F1 ≥ 0.50, hit F1 ≥ 0.60 on the committed cache (no GPU/video).

See `docs/research/bounce-vs-shot-methodo-CR.md` for the full methodology + the
iteration journal.
