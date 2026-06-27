# PROMPT — S2 : pousser la far-pose au-delà de 84.9% (Backend R&D+impl)

> **Session AUTONOME ULTRACODE.** Lance en `/ultracode`.
> ⛔ **NE POSE JAMAIS DE QUESTION. Décide, documente, livre.**
> ✅ **RECHERCHE WEB ENCOURAGÉE** : cherche les meilleures stratégies 2024-2026 de
> détection de PETITES personnes lointaines (small-object detection, SAHI tiling,
> high-res inference, RTMDet/RTMW, ViTPose, sport-player tracking)… Évalue licence
> (MIT/Apache OK), faisabilité CPU/MPS, gain plausible. Implémente ce qui mesure mieux.

## Le problème (mesuré, GT humaine)
La sélection du joueur far a été corrigée à **84.9%** far-CORRECT (was 0.9%), via le score `hcv` (hauteur/centralité/conf/validité). Le plafond restant = **89.3%** (24/225 GT sans candidat = trou de DÉTECTION du petit joueur far). Cible : monter le far-CORRECT vers ~89% (récupérer les frames sans candidat) SANS casser felix.

## Le thermomètre (GT + scorer — mesure LIVE, zéro triche)
- GT : `tests/fixtures/pose_gt/tennis_demo3.pose_gt.json` (225 points far humains).
- Scorer : `python tools/pose_gt/measure_far_coverage.py` → far-CORRECT % vs GT. Baseline 84.9%.
- ⚠️ Le clip `data/output/tennis_demo3.mp4` est VIERGE (corrigé) — ne le régénère pas en annoté.

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-s2pose feat/s2-far-pose-sota
```
Branche `feat/s2-far-pose-sota` (depuis `feat/accuracy-overhaul`). Édite SOUS le worktree. **Fichiers : `vision/pose.py` UNIQUEMENT** (+ un module détecteur séparé si une techno externe est intégrée). NE touche PAS `models/yolo_detector.py` (S1), `vision/events.py`, `vision/shots.py`.

## Pistes (mesurer, pas supposer)
1. **Le plafond = détection** : les 24 frames sans candidat far. Le far est-il détectable à conf encore plus basse / imgsz plus élevé / sur un crop upscalé de la moitié haute (SAHI tiling) ? Mesure combien des 24 deviennent récupérables.
2. **Techno externe SOTA** (recherche web) : un détecteur de personnes plus fort sur les petites cibles lointaines (RTMDet, un modèle sport-player), licence-clean, testé MPS/CPU.
3. **Pose backend** : RTMW (déjà tenté, n'aidait pas la sélection — mais avec le far mieux détecté, re-mesure). Documente.
4. **Sélection** : raffiner `hcv` (le score gagnant) si des far mal sélectionnés subsistent, sans surapprendre demo3.

## Contraintes dures
- **NE régresse PAS felix** : `test_far_coverage`, `test_bounce_regression`, `test_vx_veto`, `test_event_confusion_regression`, `test_sharp_turns` — verts. felix far-CORRECT doit rester ≥ ~85% (la centralité pure le cassait — ne refais pas ce piège).
- **Zéro hardcoding / zéro triche** : seuils géométriques généralisables, pas de pixels demo3, pas de lecture GT par le prod. Licence externe MIT/Apache.

## Boucle + livrable
Itère mesure→change→re-mesure SANS plafond. CR `docs/research/s2-far-pose-sota-CR.md` : journal far-CORRECT (demo3 + felix), technos web évaluées (+licence+verdict), gain, preuve felix intact, idée réutilisable. Commit + push. Chiffres LIVE mesurés.
