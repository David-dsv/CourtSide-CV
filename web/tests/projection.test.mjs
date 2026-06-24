// Standalone assert-based test for lib/court/projection.ts.
// No test runner required — run with:
//   node --experimental-strip-types --no-warnings web/tests/projection.test.mjs
// (Node >= 22.6 strips TS types natively; we import the real module so the
// assertions exercise the production code, not a copy.)
import assert from "node:assert/strict";
import {
  pxToMeters,
  projectMeters,
  onRadar,
  depthFromMeters,
  clampToCourt,
  COURT,
} from "../src/lib/court/projection.ts";

let passed = 0;
const check = (name, fn) => {
  fn();
  passed++;
  console.log(`  ✓ ${name}`);
};

const DW = COURT.DOUBLES_HALF_W; // 5.485
const HL = COURT.HALF_LEN; // 11.885
const PAD = COURT.PAD_M;

// ── With a homography H: metric-exact, known points map to known meters ──────
// A pure affine H: Xm = 0.01*x − 9.6 , Ym = −0.02*y + 12  (image→court meters).
const H = [
  [0.01, 0.0, -9.6],
  [0.0, -0.02, 12.0],
  [0.0, 0.0, 1.0],
];

check("pxToMeters(exact=true) when H is provided", () => {
  const r = pxToMeters(960, 600, H, 1920, 1080);
  assert.equal(r.exact, true);
  // Xm = 0.01*960 − 9.6 = 0.0 ; Ym = −0.02*600 + 12 = 0.0 → court center.
  assert.ok(Math.abs(r.Xm) < 1e-9, `Xm=${r.Xm}`);
  assert.ok(Math.abs(r.Ym) < 1e-9, `Ym=${r.Ym}`);
});

check("pxToMeters applies H linearly across the frame", () => {
  const a = pxToMeters(100, 100, H, 1920, 1080);
  const b = pxToMeters(1820, 100, H, 1920, 1080); // +1720 in x
  // affine: ΔXm = 0.01 * Δx = 17.2
  assert.ok(Math.abs((b.Xm - a.Xm) - 0.01 * 1720) < 1e-6);
});

// ── Without H: geometric fallback keeps points inside the court ─────────────
check("pxToMeters(exact=false) without H, central px land in-court", () => {
  // a point near image center → near court center
  const r = pxToMeters(960, 540, undefined, 1920, 1080);
  assert.equal(r.exact, false);
  assert.ok(onRadar(r.Xm, r.Ym), `center off-radar: ${r.Xm},${r.Ym}`);
});

check("fallback maps the full frame band inside court bounds", () => {
  // sample a grid across the calibrated band [top..bot]; every interior point
  // must satisfy |Xm| ≤ doubles-half-width+pad and |Ym| ≤ half-length+pad.
  for (let fy = 0.34; fy <= 0.85; fy += 0.05) {
    for (let fx = 0.1; fx <= 0.9; fx += 0.1) {
      const r = pxToMeters(fx * 1920, fy * 1080, undefined, 1920, 1080);
      assert.ok(
        Math.abs(r.Xm) <= DW + PAD,
        `Xm out of bounds at fy=${fy}, fx=${fx}: ${r.Xm}`,
      );
      assert.ok(
        Math.abs(r.Ym) <= HL + PAD,
        `Ym out of bounds at fy=${fy}, fx=${fx}: ${r.Ym}`,
      );
    }
  }
});

check("fallback depth axis: top of frame = far baseline (+Ym)", () => {
  const top = pxToMeters(960, 0.34 * 1080, undefined, 1920, 1080);
  const bot = pxToMeters(960, 0.85 * 1080, undefined, 1920, 1080);
  assert.ok(top.Ym > bot.Ym, `top should be farther (+Ym): top=${top.Ym} bot=${bot.Ym}`);
});

// ── projectMeters: court meters → tile px, court center → tile center ────────
check("projectMeters maps (0,0) to the tile center", () => {
  const [px, py] = projectMeters(0, 0, 360, 760);
  assert.ok(Math.abs(px - 180) < 1e-6);
  assert.ok(Math.abs(py - 380) < 1e-6);
});

check("projectMeters +Ym (far) maps toward the TOP of the tile", () => {
  const [, pyFar] = projectMeters(0, HL, 360, 760);
  const [, pyNear] = projectMeters(0, -HL, 360, 760);
  assert.ok(pyFar < pyNear, "far baseline should render above near baseline");
});

// ── onRadar bounds the pad-expanded court ───────────────────────────────────
check("onRadar accepts in-court and rejects far-out points", () => {
  assert.ok(onRadar(0, 0));
  assert.ok(onRadar(DW, HL));
  assert.ok(!onRadar(DW + PAD + 10, 0));
  assert.ok(!onRadar(0, HL + PAD + 10));
});

// ── depthFromMeters: ITF bands from |Ym| ────────────────────────────────────
check("depthFromMeters classifies the ITF depth bands", () => {
  assert.equal(depthFromMeters(0), "short"); // on the net
  assert.equal(depthFromMeters(-3.9), "short");
  assert.equal(depthFromMeters(4), "mid"); // boundary → mid
  assert.equal(depthFromMeters(-6.9), "mid");
  assert.equal(depthFromMeters(7), "deep"); // boundary → deep
  assert.equal(depthFromMeters(HL), "deep"); // on a baseline
});

check("depthFromMeters is symmetric in the sign of Ym (both sides)", () => {
  // a bounce deep on the near side (-Ym) is just as "deep" as one on the far side
  for (const d of [1, 3, 5, 8, 11]) {
    assert.equal(depthFromMeters(d), depthFromMeters(-d), `mismatch at |Ym|=${d}`);
  }
});

// ── clampToCourt: keeps synthetic pucks on the map ──────────────────────────
check("clampToCourt pulls out-of-bounds points back to the pad edges", () => {
  const over = clampToCourt(DW + 50, HL + 50);
  assert.ok(over.Xm <= DW + PAD + 1e-6);
  assert.ok(over.Ym <= HL + PAD + 1e-6);
  const under = clampToCourt(-DW - 50, -HL - 50);
  assert.ok(under.Xm >= -(DW + PAD) - 1e-6);
  assert.ok(under.Ym >= -(HL + PAD) - 1e-6);
});

check("clampToCourt leaves in-court points untouched", () => {
  const c = clampToCourt(1.2, -3.4);
  assert.ok(Math.abs(c.Xm - 1.2) < 1e-9 && Math.abs(c.Ym + 3.4) < 1e-9);
});

console.log(`\n${passed} passed`);
