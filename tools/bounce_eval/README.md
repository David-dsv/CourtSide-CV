# Bounce-detection ground-truth tooling

The objective measuring stick for bounce detection. Before tuning
`detect_bounces_from_trajectory()` in `run_pipeline_8s.py`, we need a way to
score it — this is that thermometer.

Three pieces:

| File | Role |
|---|---|
| `bounce_io.py` | Shared JSON schema + read/write. Single source of truth for the format. |
| `annotate_bounces.py` | Hand-annotate true bounces in a clip (OpenCV frame-stepper). |
| `eval_bounces.py` | Compare predictions vs. ground truth → precision / recall / F1. |

## File format

One JSON file per clip (see `bounce_io.py` docstring for the full schema). Only
`frame` is required per bounce; `x`, `y`, `depth` (`deep`/`mid`/`short`) are
optional. `frame` is absolute in the source video.

- **Ground truth** → `tests/fixtures/bounces/<clip>.bounces.json` (committed —
  it's the calibration of the thermometer, don't lose it).
- **Predictions** → `data/preds/<clip>.bounces.json` (gitignored — regenerable).

## 1. Annotate

```bash
python tools/bounce_eval/annotate_bounces.py data/clips/match1.mp4
```

Steps through the clip in an OpenCV window. Keys (also drawn on-screen):

```
d / →  next      a / ←  prev      f  +10      b  -10      g  jump-to-frame
SPACE  mark bounce        click  set ball x,y        1/2/3  depth deep/mid/short
x      remove bounce at/nearest current frame        u  undo last add
s      save               q  quit (prompts if unsaved)
```

Removal is position-independent: navigate back to any marked frame (the HUD
lists nearby marks) and press `x`. Re-running on an existing output file loads
it so you can resume or fix mistakes.

## 2. Evaluate

```bash
python tools/bounce_eval/eval_bounces.py \
  --pred  data/preds/match1.bounces.json \
  --truth tests/fixtures/bounces/match1.bounces.json
```

A predicted bounce counts as correct if within ±N frames of a true one
(one-to-one, greedy nearest). Default `N = round(0.15 * fps)`; override with
`--tolerance-frames`. Tune N once you've seen real results.

## Predictions source (future, separate step)

`eval_bounces.py` is a pure file comparator — it does not run the pipeline.
Predictions will come from a gated, additive `--dump-bounces PATH` export added
to `run_pipeline_8s.py` (off by default, reads the already-computed
`bounce_events`). That change is intentionally **not** part of this tooling and
will be proposed as its own diff. Until then, hand-write a predictions file to
exercise the evaluator.
