# Shot-guard **CUT** — integration contract

> Module owner: session **S2** (`feat/shot-guard-cut`).
> Files owned: `vision/shot_guard.py`, `vision/annot_clip.py`, their tests.
> **S2 does NOT touch `run_pipeline_8s.py`** — this doc tells S1 how to wire it.

## What changed vs `feat/shot-guard`

`feat/shot-guard` computes a court-visibility mask and **keeps every frame**
(timeline 1:1), drawing "clean" (no overlay) on non-court frames. The PM
decision overrides that: we **actually CUT** the non-court frames out of the
annotated video, so:

- the annotation loop does pose/ball/overlay/**write only on court frames** →
  **0 CPU** spent on replays/close-ups/graphics;
- the annotated output is **shorter** than the source (frame↔time mapping
  changes) → we emit a **clip-index sidecar** so the front (S5) can re-align;
- the **original video is never rewritten** (we only read the input).

This module (`vision/annot_clip.py`) is pure list/dict math — no video, no I/O
(except the optional sidecar writer). It is fully unit-tested without a GPU.

---

## API — `vision/annot_clip.py`

### Index conventions (read first)
- The mask is over the **processed window** only → mask index `i` is
  **window-relative** (`0 … len(mask)-1`).
- `start_frame = int(args.start * fps)` is the absolute index of the window's
  first frame. **Absolute original index of mask `i` = `start_frame + i`.**
- `kept_original_frames` holds **ABSOLUTE** original indices (so the front
  aligns straight onto the source timeline).
- `cut_spans` stay **window-relative** (they mirror `non_court_spans(mask)`
  1:1). `start_frame` is recorded so either space converts to the other.

### Functions
```python
frames_to_annotate(mask) -> list[int]
    # window-relative indices where mask[i] is True (temporal order).
    # The annotation loop iterates ONLY these.

build_clip_index(mask, fps, start_frame=0) -> dict
    # the sidecar payload (see schema below).

original_to_annotated(clip_index, original_frame) -> int | None
    # absolute original frame -> annotated index, or None if it was CUT.
    # (doubles as the "was this frame cut?" test). O(log n).

annotated_to_original(clip_index, annotated_frame) -> int
    # annotated index -> absolute original frame. Raises IndexError if out of
    # range (every annotated frame is kept by construction).

write_clip_index(output_path, mask, fps, start_frame=0) -> (path|None, dict)
    # OPTIONAL helper: writes <output-without-ext>_clipindex.json next to the
    # video / _stats.json. Never raises on I/O (returns (None, clip_index) on
    # failure), mirroring the pipeline's stats-emission policy.
```

### Clip-index JSON schema (`<out>_clipindex.json`)
```jsonc
{
  "fps": 50.0,
  "original_total": 400,            // len(mask) — frames in the processed window
  "annotated_total": 256,           // sum(mask) — frames actually kept/annotated
  "kept_original_frames": [60, 61, 64, ...],  // ABSOLUTE original idx, annotated order
  "cut_spans": [[2, 14], [110, 130]],         // window-RELATIVE inclusive non-court runs
  "start_frame": 60                 // absolute idx of mask[0]; bridges the two spaces
}
```
`kept_original_frames[k]` = absolute original frame index of the **k-th
annotated frame** — the re-alignment mapping.

---

## How S1 wires it into `run_pipeline_8s.py`

S1 already computes `court_visible_mask` in **Pass 0.5** (the streaming
shot-guard block). The CUT changes are downstream of that.

### 1. Pass 0.5 — compute the keep-list (unchanged mask, new keep-list)
```python
from vision.annot_clip import frames_to_annotate, write_clip_index

# ... existing Pass 0.5 produces court_visible_mask (window-relative) ...
keep = frames_to_annotate(court_visible_mask)   # window-relative idx to annotate
keep_set = set(keep)                             # O(1) membership in the loop
logger.info(f"shot-guard CUT: annotating {len(keep)}/{max_frames} frames "
            f"({100*len(keep)/max(1,max_frames):.1f}%), "
            f"{max_frames-len(keep)} cut")
```

### 2. The annotation loop — SKIP cut frames entirely
This is the core difference from `feat/shot-guard`. There, the loop runs on
every `i` and *nulls* ball/pose on non-court frames but still **writes** the
frame. Here, the loop **`continue`s before any work and before the write** on a
cut frame, so the writer only receives court frames:

```python
for i, frame in enumerate(window_frames):     # i is window-relative
    if shot_guard_enabled and i not in keep_set:
        continue                              # CUT: no pose/ball/overlay, no write
    # ... pose / ball / overlay as today ...
    writer.write(annotated_frame)             # only court frames reach the writer
```
> ⚠️ This applies to **every pass that writes or detects per-frame**: Pass 1
> ball detection, Pass 1.5 pose, and Pass 2 annotation should all skip the
> same cut frames (don't burn CPU detecting on a frame you'll never write).
> The cheapest correct shape: gate at the top of each per-frame loop on
> `i not in keep_set`.

> 🔗 **Frame-index bookkeeping stays unchanged.** Downstream data
> (`_stats.json` bounces/shots/etc.) keeps using **absolute original** indices
> (`start_frame + i`) exactly as today — the cut does **not** renumber those.
> Only the *annotated video's* internal frame numbering compresses, and that
> compression is captured entirely by the clip-index. So a shot recorded at
> absolute frame `F` in `_stats.json` is located in the annotated video via
> `original_to_annotated(clip_index, F)`.

### 3. Write the sidecar next to `_stats.json`
Right where `_stats.json` is written, add:
```python
clip_path, clip_index = write_clip_index(output_path, court_visible_mask,
                                          fps, start_frame)
if clip_path:
    logger.info(f"Clip-index written to: {clip_path} "
                f"({clip_index['annotated_total']}/{clip_index['original_total']} kept)")
```
(or inline `build_clip_index` + your own `json.dump` if you prefer; the helper
just packages the path convention + the never-raise policy.)

### 4. Original video is never rewritten
The pipeline only ever **reads** `video_path` (Pass 0.5 `VideoReader`, Pass 1/2
readers) and **writes** a *new* `output_path`. The CUT does not change that:
no code path opens the input for writing. Guarantee holds by construction.

### Disable switch
Keep `feat/shot-guard`'s `--no-shot-guard`. When disabled, `keep ==
range(max_frames)` (mask all-True) → `build_clip_index` yields the identity
mapping and `cut_spans == []`, so the sidecar is still emitted, just a no-op
(annotated == original). The front can treat "no clip-index" and "identity
clip-index" the same.

---

## Contract for S5 (front)

The front reads `<video>_annotated_clipindex.json` alongside
`<video>_annotated_stats.json`.

**Re-align an annotated-video frame → original/time:**
```js
const origFrame = clip.kept_original_frames[annotatedFrame];  // absolute
const origSeconds = origFrame / clip.fps;
```

**Locate a stats event (absolute original frame `F`) in the annotated video:**
binary-search `kept_original_frames` for `F`.
- found → that array index is the annotated frame;
- not found → `F` was CUT (it's inside a `cut_spans` run, shifted by
  `start_frame`); show it on the original timeline only, or snap to the nearest
  kept neighbour.

**Original-video scrubbing:** `cut_spans` (window-relative; add `start_frame`
for absolute) marks the greyed-out / "replay" regions on the source timeline so
the UI can show where annotation was skipped.

**Identity / disabled case:** `annotated_total == original_total`,
`cut_spans == []` → annotated and original are frame-aligned; no remapping
needed.

---

## Honest limits

- Detection is `vision/shot_guard.py`'s HoughLinesP line-count gate
  (≥40 long lines) + dwell hysteresis. It can **miss** on amateur framings
  (a tight phone angle where the full court fills the frame but few long lines
  show, or a low court-level grazing angle). It keys on *court-line geometry*,
  not content, so a graphics-heavy broadcast frame with strong straight edges
  could read as court. These are the gate's known limits, inherited as-is.
- On **felix** the clip is **100 % court** → the mask is all-True → **0 cut**
  (`annotated == original`, identity clip-index). That is **correct**, not a
  bug: there are no replays/close-ups to remove. The cut only fires on
  broadcast-style footage with out-of-play segments (e.g. `tennis.mp4`,
  ~36 % non-court).
- `annot_clip.py` is exact and total (round-trip-tested, bounds-tested); the
  only fuzziness is upstream, in whether the mask itself is right.
