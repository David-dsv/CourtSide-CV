# PROMPT — ROI dynamique balle : crop zoomé autour de la prédiction (Backend) — SESSION

> **Session AUTONOME ULTRACODE.** Lance en `/ultracode`.
> ⛔ **NE POSE JAMAIS DE QUESTION. Décide, documente, livre.**
> ✅ **RECHERCHE WEB ENCOURAGÉE** : search-region tracking, ROI cropping pour
> tiny-object/ball tracking (SiamRPN, TrackNet+ROI, local search window)…
> Évalue + adopte ce qui mesure mieux. Licence MIT/Apache si code externe.

## L'idée (proposée + PRÉ-MESURÉE par le PM — elle marche)
Quand la détection balle pleine image rate, **croper une fenêtre autour de la dernière position connue / la prédiction Kalman, l'upscaler modérément, et re-détecter dedans** — la balle y est plus grosse et plus nette, moins de parasites (gradins/scoreboard/minimap ignorés). C'est du *search-region tracking* (SOTA classique).

**Pré-mesuré (PM, /tmp/roi_test.py) sur la GT balle :** sur les **71 frames** où la config mergée (1920/conf0.05) rate la balle ALORS qu'une prédiction est dispo, un **ROI crop 320px + zoom 2× + conf 0.05** en **récupère 20/71 (28%)**. C'est un vrai gain (≠ 0/71 = pas un model-ceiling), qui pousserait la couverture ~95.5% → ~98%.

⚠️ **C'est DIFFÉRENT du SAHI/tiling déjà rejeté** (0/17) : SAHI = grille FIXE aveugle sur toute l'image ; ici = UN crop DYNAMIQUE guidé par la prédiction (on sait où chercher). Ne confonds pas.

## Le réglage (validé par le test — respecte la prudence "pas trop puissant")
- **Fenêtre LARGE** (~320px = ±160 autour de la prédiction) : assez pour absorber l'erreur de prédiction quand la balle change brutalement de direction (rebond/frappe). Un crop trop serré rate la balle au rebond.
- **Zoom MODÉRÉ ~2×** : assez pour grossir la balle, pas trop pour ne pas la rendre floue/pixelisée (au-delà, le modèle ne la reconnaît plus). Mesure 1.5×/2×/2.5× et garde le meilleur sur la GT.
- **conf basse (0.05)** dans le crop + remap des coords vers le plein cadre (`cx = cx_crop/zoom + x0`).

## Où l'insérer (le point d'ancrage existe déjà)
`run_pipeline_8s.py` Pass 1 : `BallKalmanFilter` (~l.46) expose `predict()` → la position prédite = le centre du ROI. Le ROI est un **FALLBACK** : ne s'active que sur une frame où la détection pleine image n'a donné aucun candidat dans le gate Kalman (sinon coût inutile). Quand le Kalman n'a pas encore d'état (toute 1ère frame, cold-start), pas de prédiction → pas de ROI (ou ROI centré sur la dernière détection brute).

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-roi feat/ball-roi-dynamic
```
Branche `feat/ball-roi-dynamic` (depuis `feat/accuracy-overhaul` — qui a DÉJÀ le density-adaptive S1). Édite SOUS le worktree. **Fichiers : `models/yolo_detector.py` (un helper `detect_ball_roi(frame, cx, cy, half, zoom)`) + `run_pipeline_8s.py` (le fallback ROI dans Pass 1, zone balle).** NE touche PAS `vision/`, `models/wasb_ball_tracker.py`.

## Le thermomètre (GT + scorer — mesure LIVE, zéro triche)
- GT : `tests/fixtures/ball_gt/tennis_demo3.ball_gt.json` (311 points).
- Scorer : `python tools/ball_gt/measure_ball_coverage.py` → coverage % + CORRECT % (overall + cold-start). Baseline (1920-Kalman density-adaptive) = 95.5% / CORRECT 92.3%.
- **Cible : pousser coverage + CORRECT au-dessus de 95.5% / 92.3%** en récupérant les trous via le ROI, SANS introduire de parasites (le Kalman gating les rejette — vérifie que le ROI ne casse pas le CORRECT).
- ⚠️ Mesure sur du LIVE / la GT, ≥2 configs de zoom. Pas de chiffre de cache non-revalidé.

## Contraintes dures
- **NE régresse PAS felix** : `test_bounce_regression` (0.800/0.774), `test_bounce_wasb_regression` (0.865), `test_ball_conf_regression`, `test_event_confusion_regression`, `test_vx_veto`, `test_far_coverage` — verts. felix est parasite-heavy → vérifie que le ROI ne lui injecte pas de faux positifs (il tourne sur le chemin 1280/0.16 ; le ROI fallback doit rester sûr là aussi, ou être gardé par la même logique density-adaptive).
- **Coût** : le ROI est un FALLBACK (seulement sur les trous) → coût marginal. Mesure-le, garde-le raisonnable.
- **Zéro hardcoding** (fenêtre/zoom = ratios de frame/vitesse balle, pas pixels demo3) ; **zéro triche** (pas de GT lue par le prod) ; anti-surapprentissage (felix).

## Boucle + livrable
Itère mesure→change→re-mesure SANS plafond. CR `docs/research/ball-roi-dynamic-CR.md` : journal (coverage/CORRECT/cold-start à chaque zoom/fenêtre testé), combien de trous le ROI récupère, coût, preuve felix intact, l'idée la plus réutilisable. Commit + push. Chiffres LIVE mesurés.
