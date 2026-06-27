# PROMPT — S1 : pousser le tracking balle vers le SOTA (Backend R&D+impl)

> **Session AUTONOME ULTRACODE.** Lance en `/ultracode`.
> ⛔ **NE POSE JAMAIS DE QUESTION. Aucun outil de clarification. Décide, documente, livre.**
> ✅ **RECHERCHE WEB ENCOURAGÉE** : cherche activement les meilleures technos/stratégies
> 2024-2026 de tracking de balle de tennis (TrackNet v2/v3, WASB, monotrack, ballnet,
> heatmap-temporel, tiny-object detection, SAHI tiling, test-time augmentation…).
> Évalue leur licence (MIT/Apache OK, AGPL/no-license = bloqueur SaaS), leur faisabilité
> CPU/MPS, et leur gain plausible. Implémente ce qui est mesurablement meilleur.

## Le problème (mesuré, GT humaine)
La couverture du tracking balle est **86.2%** (cold-start 85.3%) après le fix conf-0.16, mais le recall des événements reste plafonné par les trous de trajectoire. Cible : monter la couverture **et le CORRECT** sur la GT, surtout au cold-start et sur les balles rapides/petites/hautes.

## Le thermomètre (GT + scorer — ZÉRO triche, mesure LIVE)
- GT : `tests/fixtures/ball_gt/tennis_demo3.ball_gt.json` (311 points balle humains + 39 occluded).
- Scorer : `python tools/ball_gt/measure_ball_coverage.py [--ball-tracker kalman|wasb] [--ball-conf X]` → coverage % + CORRECT % (overall + cold-start). Baseline kalman conf 0.16 = 86.2%/79.1%.
- Replay instantané sur cache : `tools/ball_gt/replay_strat.py --cache tests/fixtures/ball_gt/demo3_det_cache_1280.json --sweep`.
- ⚠️ **Mesure sur du LIVE / la GT, jamais un chiffre de cache figé sans le revalider** (leçon : nos caches ont menti 4×). Vérifie tout gain sur ≥2 configs.

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-s1ball feat/s1-ball-sota
```
Branche `feat/s1-ball-sota` (depuis `feat/accuracy-overhaul`). Édite SOUS `/Users/vuong/Documents/CourtSide-CV-s1ball/...` (mémoire `courtside-edit-targets-main-tree`). **Fichiers : `models/wasb_ball_tracker.py` + un nouveau module si une techno externe est intégrée** (ex. `models/<tracker>.py`). NE touche PAS `vision/events.py`, `vision/pose.py`, `vision/shots.py`.

## Pistes (mesurer, pas supposer)
1. **WASB** : il existe (`--ball-tracker wasb`). Mesure sa couverture sur la GT vs Kalman. Le fixer (frame_step pour 50fps ? normalisation ImageNet — cf. mémoire `wasb-bounce-f1-865`) pourrait densifier.
2. **Tiling / upscale** (SAHI-like) : la balle haute/petite est ratée — détecter sur des crops upscalés (la moitié haute) puis fusionner. Mesure le gain cold-start.
3. **Techno externe SOTA** (recherche web) : si un tracker récent licence-clean bat WASB/Kalman sur ce type de broadcast, intègre-le derrière un flag et mesure.
4. **Fusion multi-source** : YOLO26 + WASB + continuité Kalman → trajectoire plus dense que chaque source.

## Contraintes dures
- **NE régresse PAS felix** : `test_bounce_regression` (≥0.72), `test_bounce_wasb_regression` (≥0.80), `test_ball_conf_regression`, + les autres — verts à chaque itération.
- **Zéro hardcoding** (CLAUDE.md) ; anti-surapprentissage (vérifie felix). **Zéro triche** : pas de seuil tuné sur demo3, pas de lecture de la GT par le code de prod.
- Licence : tout code/poids externe doit être MIT/Apache (pas AGPL/no-license — bloqueur SaaS). Documente la licence.

## Boucle + livrable
Itère mesure→change→re-mesure SANS plafond. CR `docs/research/s1-ball-sota-CR.md` : journal (couverture/cold-start/CORRECT à chaque étape), technos web évaluées (+ licence + verdict), ce qui a gagné, preuve felix intact, l'idée la plus réutilisable. Commit + push. VRAIS chiffres mesurés LIVE.
