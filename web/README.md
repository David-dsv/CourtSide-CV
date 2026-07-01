# CourtSide — Frontend SaaS (web/)

Premium dark-glass frontend for the CourtSide tennis-analysis platform.
**Next.js 15/16 (App Router) · TypeScript · Tailwind v4 · shadcn (base-ui) · pnpm.**

This sprint ships the full SaaS shell on **mock data** (no backend yet). The mock
imports the **real pipeline JSON** from `data/output/*.json`, so every stat, bounce,
shot and confidence flag on screen is genuine pipeline output.

## Run

```bash
cd web
pnpm install        # uses .npmrc local overrides (trust-policy=report) — global pnpm is hardened
pnpm dev            # http://localhost:3000 (or next free port)
pnpm build          # production build + type-check
pnpm lint
```

> npm is blocked on this machine (shell alias). **Always use pnpm.**

## What's here

- **Marketing**: `/landing` (hero + 3 pillars + pipeline + honest-stats demo), `/pricing` (Free/Starter/Coach + FAQ).
- **Auth** (mock cookie session): `/login`, `/signup` + demo prefill. Guard in `src/middleware.ts`.
- **App** (protected): `/projects` (grid), `/projects/new` (mock upload → Pass 1 → Pass 2 → ready),
  and 7 project sub-pages: `overview`, `stats`, `shots`, `bounces`, `minimap`, `player-compare`, `coach-plan`.

## Key design decisions

- **Calibrated confidence is a first-class design element.** `ConfidenceBadge`
  (`src/components/core/confidence-badge.tsx`) maps `speed_source` / `homography_confidence`
  to `homography(high)` / `scalar` / `~approx` — the product never lies about precision.
- **Frame-as-truth.** Every stat/insight is a `FrameJumpLink` that drives the shared
  `VideoFrameContext`; the mock `VideoScrubber` and `CourtMinimap` react to the current frame.
- **CourtMinimap** (`src/components/court/court-minimap.tsx`) is an SVG port of `vision/minimap.py`
  (ITF court geometry in `src/lib/court/`, bounce colors by depth, P1/P2 pucks). The INFERNO
  `PlacementHeatmap` is Canvas (port of `vision/fx.py`).
- **Glassmorphism** utilities (`@utility glass` in `globals.css`) mirror `vision/fx.py` for
  visual continuity between the annotated clip and the dashboard.

## Mock data

Five real fixtures cover the full confidence spectrum (`src/lib/mock/raw/`, checked against the
actual JSON — keep this table in sync):

| id | fichier | speed_source / confiance | contenu |
|---|---|---|---|
| `acapulco-hero` | `hero.json` | **`homography(high)`** — H projectif RÉEL émis par le pipeline (`homography_H`) | le cas vitrine : 10 bounces / 9 shots (service inclus), 17/17 vs GT humaine, vitesses métriques |
| `tennis-demo` | `tennis.json` | `scalar` / `none` | 9 bounces / 14 shots, clip-index synthétique (démo cut-timeline) |
| `tennis-full` | `tennis-full.json` | `scalar` / `none`, `match_mode` | 38 bounces / 149 shots / 15 rallies avec outcomes + 19 side-swaps |
| `felix-sota` | `sota.json` | `homography(low)` (grazing calibré) | 20 bounces / 7 shots |
| `felix` | `felix.json` | `homography(low)` → fallback géométrique `~` côté radar (pas de H) | 6 bounces / 5 shots, le cas pathologique honnête |

The dataviz series colors (`--viz-*` in `globals.css`) are validated per theme (lightness band,
chroma floor, CVD ΔE, 3:1 contrast) — if you change them, re-run a palette validator before
shipping.

## Wiring a real backend later

Replace `src/lib/mock/{projects,auth,coach-insights}.ts` with `fetch` calls to the FastAPI
service. The Pydantic schemas map 1:1 to `src/lib/types.ts`. Swap the cookie session for NextAuth
without touching layouts (`getSession`/`requireSession` contract).
