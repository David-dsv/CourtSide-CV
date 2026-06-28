"""Unit test for the FAR-detection safety net ``vision.pose._roi_redetect`` (S2).

Fast (no GPU / no video): a FAKE detector returns a known box in the upscaled-crop
frame; the test asserts the helper maps it back to full-frame coordinates EXACTLY and
applies the above-net filter. This is the committed coverage the bypass-gate probe
(``tools/pose_gt/_probes/probe_roi_net.py``) gave only informally.

Two properties are locked:
  1. Coordinate round-trip: box_fullframe == [x0 + bx/scale, y0 + by/scale, ...] where
     (x0, y0) is the crop origin and scale the upscale factor — no off-by-one/scale bug.
  2. Above-net filter: a returned box whose mapped centre falls BELOW net_y is dropped
     (the helper only yields far-side candidates).

Run:  python tests/test_roi_redetect.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vision.pose import _roi_redetect  # noqa: E402


class _FakeBoxes:
    def __init__(self, xyxy, conf):
        self._xyxy = np.asarray(xyxy, dtype=float)
        self._conf = np.asarray(conf, dtype=float)

    def __len__(self):
        return len(self._xyxy)

    @property
    def xyxy(self):
        return _Tensorish(self._xyxy)

    @property
    def conf(self):
        return _Tensorish(self._conf)


class _Tensorish:
    """Mimics the .cpu().numpy() chain ultralytics boxes expose."""
    def __init__(self, arr):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def _make_fake_detector(boxes_in_upscaled_crop, box_confs):
    """Return a callable matching the detector(...) → [result] convention. It records
    the imgsz it was called with so the test can assert imgsz is sized to the crop.
    ``box_confs`` are the per-box confidences the fake detector returns (distinct from
    the ``conf`` THRESHOLD argument the helper passes in)."""
    state = {}

    def detector(img, conf=None, imgsz=None, classes=None, device=None, verbose=None):
        state["imgsz"] = imgsz
        state["img_shape"] = img.shape
        return [_FakeResult(_FakeBoxes(boxes_in_upscaled_crop, list(box_confs)))]

    detector.state = state
    return detector


def test_coordinate_roundtrip_and_above_net_filter():
    fw, fh = 1920, 1080
    net_y = 0.5 * fh                      # 540
    scale = 2.0
    # crop origin used by _roi_redetect (must mirror the helper's geometry)
    x0, y0 = int(0.10 * fw), int(max(0.0, 0.20 * net_y))   # (192, 108)
    # Box A: a far player. Place its full-frame target at center (800, 200) → above net.
    # In the upscaled crop, that maps to ((800 - x0)*scale, (200 - y0)*scale) center.
    ax1f, ay1f, ax2f, ay2f = 770, 160, 830, 240          # full-frame target box A
    a_up = [(ax1f - x0) * scale, (ay1f - y0) * scale,
            (ax2f - x0) * scale, (ay2f - y0) * scale]
    # Box B: a below-net box (centre at y=520*... ) — its mapped centre must be < net_y
    # to survive; we craft it to map BELOW net_y so it is DROPPED.
    bx1f, by1f, bx2f, by2f = 900, 520, 960, 600           # full-frame, centre y=560>540
    b_up = [(bx1f - x0) * scale, (by1f - y0) * scale,
            (bx2f - x0) * scale, (by2f - y0) * scale]

    det = _make_fake_detector([a_up, b_up], [0.77, 0.40])
    frame = np.zeros((fh, fw, 3), dtype=np.uint8)
    out = _roi_redetect(det, frame, net_y, fw, fh, "cpu", conf=0.08, imgsz=1280,
                        scale=scale)

    # Box B (below net) dropped → only box A returned
    assert len(out) == 1, f"expected only the above-net box, got {len(out)}"
    box, conf = out[0]
    assert abs(conf - 0.77) < 1e-6
    # Exact coordinate round-trip
    for got, want in zip(box, [ax1f, ay1f, ax2f, ay2f]):
        assert abs(got - want) < 1e-6, f"coord map off: got {box} want {[ax1f, ay1f, ax2f, ay2f]}"


def test_imgsz_sized_to_upscaled_crop():
    """The inference imgsz must be >= the full-frame imgsz (the upscale must not be
    silently undone by a smaller letterbox)."""
    fw, fh = 1920, 1080
    net_y = 0.5 * fh
    det = _make_fake_detector([], [])
    frame = np.zeros((fh, fw, 3), dtype=np.uint8)
    _roi_redetect(det, frame, net_y, fw, fh, "cpu", conf=0.08, imgsz=1280, scale=2.0)
    assert det.state["imgsz"] >= 1280, (
        f"ROI imgsz {det.state['imgsz']} < full-frame 1280 — upscale wasted")
    assert det.state["imgsz"] % 32 == 0, "imgsz must be a multiple of the YOLO stride"


def test_degenerate_roi_returns_empty():
    """A net_y so small the band collapses must return [] without calling resize/det."""
    fw, fh = 1920, 1080
    det = _make_fake_detector([[0, 0, 10, 10]], [0.9])
    frame = np.zeros((fh, fw, 3), dtype=np.uint8)
    out = _roi_redetect(det, frame, net_y=1.0, fw=fw, fh=fh, device="cpu",
                        conf=0.08, imgsz=1280, scale=2.0)
    assert out == [], "degenerate ROI should short-circuit to []"


def test_empty_detection_returns_empty():
    fw, fh = 1920, 1080
    det = _make_fake_detector([], [])
    frame = np.zeros((fh, fw, 3), dtype=np.uint8)
    out = _roi_redetect(det, frame, net_y=0.5 * fh, fw=fw, fh=fh, device="cpu",
                        conf=0.08, imgsz=1280, scale=2.0)
    assert out == []


if __name__ == "__main__":
    for name in sorted(globals()):
        if name.startswith("test_"):
            globals()[name]()
            print(f"  PASS {name}")
    print("PASS")
