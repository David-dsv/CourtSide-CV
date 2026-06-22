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

Three real fixtures cover the full confidence spectrum (`src/lib/mock/raw/`):
felix = `scalar`/`none` (grazing), tennis-demo = `homography(high)`, felix-sota = `homography(low)`.

## Wiring a real backend later

Replace `src/lib/mock/{projects,auth,coach-insights}.ts` with `fetch` calls to the FastAPI
service. The Pydantic schemas map 1:1 to `src/lib/types.ts`. Swap the cookie session for NextAuth
without touching layouts (`getSession`/`requireSession` contract).
