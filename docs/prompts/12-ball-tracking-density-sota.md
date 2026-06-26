# PROMPT — Densifier le tracking de balle (le vrai mur SOTA) : filtre player-aware + chaîne Kalman (Backend R&D mesurée)

> **Session R&D + IMPLÉMENTATION mesurée.** Effort recommandé : **ultracode**.
> **Objectif utilisateur : SOTA, la meilleure app du marché.** Itérer
> mesure→fix→re-mesure jusqu'à un rappel de balle nettement meilleur **sur le
> benchmark**, **sans jamais régresser felix**.

## ⛔ NE PAS refaire les impasses déjà mesurées (lire les mémoires AVANT)
La cause-racine est **déjà diagnostiquée** (mémoire `courtside-bounce-broadcast-multifactor`, + `scripts/diag_wasb_demo3.py` qui l'a prouvée). **Tu PARS de ce diagnostic, tu ne le refais pas :**
1. **WASB est AVEUGLE sur ce type de clip (broadcast/highlights)** : il met son pic de heatmap (0.9) sur les **JOUEURS**, jamais sur la balle (`hm@GT ≈ 0.00-0.07`). C'est un **domain shift**, PAS un sous-tir. → **Multi-scale WASB = mort, mesuré. NE PAS le retenter.**
2. **YOLO26 localise mieux la balle (5/9 GT à conf 0.1)** MAIS fire aussi sur un **parasite joueur** : un hotspot récurrent à ~(1760-1840, 680-1000) = la boîte/raquette du joueur near P2, détecté comme `tennis_ball` 50-78% des frames.
3. **Le Kalman perd la balle au rebond** (réversion brutale) et **se réinitialise sur le parasite** → trous + sauts que la détection d'event ne peut pas lire.

## Le problème à résoudre (chiffré)
La méthodo classify+alternance (déjà mergée, `vision/events.py`) atteint `confusion_H→B=0` mais **plafonne à un rappel 15/17** car **2 rebonds GT (120, 369) ne sont dans AUCUN candidat** : la balle n'y est pas trackée. **Le mur n'est pas la classification — c'est la DENSITÉ/QUALITÉ du tracking de balle.** Sur demo3 : WASB 10% (aveugle), Kalman 68% (troué aux events).

## Pistes PROMETTEUSES nommées par le diagnostic (par où commencer)
La mémoire désigne explicitement les leviers **non encore essayés**, sans GPU :
- **(PRIORITÉ 1) Filtre balle player-aware** : rejeter toute détection de balle **à l'intérieur (ou très près) d'une boîte personne** — le parasite EST la raquette/boîte du joueur. Physiquement fondé (la balle au contact est près du joueur, mais le parasite PERSISTE 50-78% des frames au même endroit alors que la vraie balle bouge → distinguer le parasite STATIQUE-near-player de la balle mobile). **Zéro hardcoding** (boîtes personnes déjà détectées par le pose-pass / generic YOLO). Pas encore construit.
- **(PRIORITÉ 2) Chaîne Kalman au rebond** : le gating (`gate = base + growth*misses`, reinit) lâche la balle quand elle inverse violemment (le rebond/la frappe). Revoir la logique de gating/reinit pour **survivre à une réversion** (c'est précisément aux events qu'on a besoin de la balle) sans réinitialiser sur le parasite. ⚠️ felix est le garde-fou : toute modif Kalman doit garder felix ≥ 0.72 (Kalman) / 0.865 (WASB).
- **(PRIORITÉ 3, exploratoire) Fusion détecteurs** : YOLO26 (meilleur localisateur, 5/9) + WASB (dense quand il voit) + le filtre player-aware → une trajectoire plus dense que chaque source seule. Mesurer que la fusion ne dégrade pas felix.
- **(NE PAS, sauf preuve)** : multi-scale WASB (mort), tuning de seuils de détecteur de rebond (mort — c'est la trajectoire qui est bruitée, pas les gates).

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-balltrack feat/ball-tracking-density
```
Branche `feat/ball-tracking-density` (depuis `feat/accuracy-overhaul`). Commit/push après chaque étape.
⚠️ [[courtside-edit-targets-main-tree]] : édite sous `/Users/vuong/Documents/CourtSide-CV-balltrack/...`.

## Le benchmark + les outils (déjà là — utilise-les)
- GT : `tests/fixtures/bounces/tennis_demo3.bounces.json` (9 rebonds) + `tests/fixtures/shots/tennis_demo3.shots.json` (8 frappes).
- Évaluateur de confusion : `tools/event_eval/` (matrice rebond↔frappe, run_demo3.py).
- Cache : `tests/fixtures/cache/demo3_event.json` (Kalman pré-spline + WASB + poses). Régénérable via `scripts/build_demo3_event_cache.py`.
- Diagnostic existant : `scripts/diag_wasb_demo3.py` (ne PAS le refaire — le lire).
- Clip : `data/output/tennis_demo3.mp4` (régénérable : `tennis.mp4 -s 73 -d 13`).

## La MÉTRIQUE de succès (ce qu'on cherche à faire monter)
La cause profonde est le **rappel de candidats**. Mesure d'abord, à chaque itération :
1. **Couverture trajectoire balle** : % de frames avec une balle trackée (vs 68% Kalman / 10% WASB actuels) — surtout **autour des 17 events GT** (fenêtre ±8 frames).
2. **Rappel du POOL de candidats** : combien des 9 rebonds + 8 frappes GT ont **au moins un candidat** dans ±8 frames (aujourd'hui plafonné à 15/17 — les 120 & 369 manquent).
3. **Puis** le rappel/F1 final via `tools/event_eval/` (rebond F1 0.533 / frappe 0.667 aujourd'hui — viser nettement plus haut une fois le pool débouché).
4. **Cohérence WASB** rapportée (signal de santé).

## Anti-régression (NON négociable)
- `tests/test_bounce_regression.py` (felix Kalman ≥ 0.72), `tests/test_bounce_wasb_regression.py` (felix WASB ≥ 0.80), `tests/test_vx_veto.py` (0.914), `tests/test_event_confusion_regression.py` (H→B=0/B→H=0), `tests/test_shots_regression.py` : **TOUS verts**. felix est le garde-fou universel.
- **Zéro hardcoding** (CLAUDE.md) : seuils = ratios de dimensions/fps/géométrie détectée (boîtes joueurs), jamais des pixels tirés de demo3. Le parasite se distingue par sa **persistance statique près d'un joueur**, pas par ses coordonnées.
- **Anti-surapprentissage** : valider sur felix ET demo3 ; le filtre player-aware ne doit pas tuer de vraies balles près d'un joueur sur felix.

## Validation + ITÉRATION (le cœur)
- `py_compile` partout.
- À chaque itération : **mesure AVANT/APRÈS** la couverture balle + le rappel de pool + la matrice de confusion. Journal exigé dans le CR.
- **Itère jusqu'à** : pool de candidats ≥ 16/17 (récupérer au moins 120 OU 369), couverture balle en hausse nette autour des events, F1 rebond+frappe en hausse — **ou preuve mesurée qu'un event est irrécupérable** (balle physiquement invisible) et documentation du plafond.

## CR à la fin (OBLIGATOIRE)
- **Journal d'itérations** (baseline couverture/rappel/F1 → chaque fix → résultat).
- Le(s) levier(s) qui ont marché (filtre player-aware ? chaîne Kalman ? fusion ?) avec les chiffres.
- La couverture balle + rappel de pool + matrice de confusion finale sur demo3.
- **Preuve felix non régressé** (tous les tests verts, valeurs exactes).
- Honnête : events irrécupérables (balle invisible), surapprentissage mono-clip, ce qui demanderait un GPU/re-train (piste (a) YOLO26 broadcast).
- Quel chemin de câblage prod (derrière quel flag) si le gain est prouvé.
Pousse sur `feat/ball-tracking-density`. Fichiers probables : `models/yolo_detector.py` (filtre player-aware), `run_pipeline_8s.py` (chaîne ball/Kalman), `models/wasb_ball_tracker.py`, `vision/bounce.py`, `tests/`, `scripts/`. **Ne touche pas `vision/events.py`** (méthodo figée) sauf pour consommer une meilleure trajectoire.
