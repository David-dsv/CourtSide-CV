# PROMPT — Densifier le tracking balle : conf basse + gating anti-parasite (Backend) — SESSION

> **Session AUTONOME ULTRACODE.** Lance en `/ultracode`.
> ⛔ **NE POSE JAMAIS DE QUESTION à l'utilisateur. Aucun outil de clarification.**
> Décide seule, documente, ne bloque jamais. Une question = run raté.

## Contexte — le diagnostic est DÉJÀ FAIT (mesuré sur une GT humaine)
Repo CourtSide-CV. Lire `CLAUDE.md`, `PROJET.md`, `docs/pm.md`, et la mémoire `courtside-ball-track-conf-not-detection` (le diagnostic clé), + `courtside-events-plateau-ball-tracking-wall`, `courtside-far-player-selection-not-detection`.

**Le vrai problème (mesuré contre la GT balle humaine `tests/fixtures/ball_gt/tennis_demo3.ball_gt.json`, 311 points balle vérifiés) :** le tracking balle ne couvre que **45.3%** des frames (YOLO26 ball à conf 0.2-0.25) → trajectoire trouée → événements (rebonds/frappes) ratés, surtout au DÉBUT du clip. C'est LE mur du recall que l'utilisateur voit.

**MAIS la balle EST détectable** (test mesuré) : à `conf=0.05, imgsz=1280` sur le modèle ball (`detector.custom_model`), les balles ratées en haut de cadre (f12, f18 — manquées à conf 0.2) sont retrouvées **à 3px de la GT**. → **C'est un problème de SEUIL DE CONFIANCE + GATING, pas une détection impossible** (même pattern que le far-player : détectable à conf basse, mais filtré). Le piège : à conf 0.05, des PARASITES apparaissent (détections à ~1200px de la vraie balle sur certaines frames) → il faut baisser la conf ET rejeter les parasites.

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-balltrack feat/ball-track-density
```
Branche `feat/ball-track-density` (depuis `feat/accuracy-overhaul`). Commit/push par itération.
⚠️ Mémoire `courtside-edit-targets-main-tree` : édite SOUS `/Users/vuong/Documents/CourtSide-CV-balltrack/...`.
Fichiers probables : `models/yolo_detector.py` (conf balle SÉPARÉE de la conf person) + `run_pipeline_8s.py` (Pass 1 : collecte candidats balle + gating). NE touche PAS `vision/events.py` (figé).

## Le chemin actuel (mesuré)
- `models/yolo_detector.py` : le modèle ball (`self.custom_model`, YOLO26 single-class) tourne à `self.conf_threshold` (= `--conf` 0.2, le MÊME que les persons). C'est ce seuil unique qui jette la petite balle faible.
- `run_pipeline_8s.py` Pass 1a (~l.780-830) : collecte tous les candidats balle par frame → static-FP filter → Kalman (`BallKalmanFilter` ~l.46) → spline. Le static-FP filter et le Kalman gating existent déjà pour rejeter les FP.

## Ton job — densifier le tracking SANS noyer de parasites
1. **Conf balle séparée + basse** : donner au modèle ball son propre seuil (ex. `ball_conf` ~0.05-0.10), distinct de la conf person (qui reste 0.2). Le far-select a fait pareil pour le détecteur far. Ajoute un flag `--ball-conf` (défaut à déterminer par la mesure).
2. **Rejeter les parasites** introduits par la conf basse : exploiter/renforcer le static-FP filter + le Kalman gating + la continuité de trajectoire (un vrai ball suit une parabole ; un parasite saute). Possible : préférer le candidat le plus proche de la prédiction Kalman, pas le plus confiant. Tester aussi `--ball-tracker wasb` (peut couvrir différemment — mesure-le).
3. **Cold-start** (priorité utilisateur) : les ratés du DÉBUT (balle haute, petite, fond chargé) — assure-toi qu'ils sont récupérés (le Kalman démarre froid : peut-être seeder la trajectoire plus tôt, ou imgsz plus élevé au début).

## Le benchmark + le scorer
- GT : `tests/fixtures/ball_gt/tennis_demo3.ball_gt.json` (311 points balle vérifiés + 39 occluded).
- **Scorer** : `python tools/ball_gt/measure_ball_coverage.py [--ball-tracker kalman|wasb]` → **coverage % + CORRECT % (overall ET cold-start)**. Baseline kalman = 45.3% overall / 44% cold-start. **Cible : monter ces deux chiffres nettement** (viser ≥70% overall, et fermer les trous du cold-start).
- Clip VIERGE : `data/output/tennis_demo3.mp4` (déjà corrigé — c'est bien la source brute sans overlay ; ne le régénère pas en annoté).

## La boucle (itère SANS plafond)
mesure (coverage + cold-start) → change UNE chose (conf, gating, tracker) → re-mesure → garde si mieux, annule si pire. **Logge coverage overall + cold-start à chaque itération.**

## Contraintes dures
- **NE régresse PAS** : `test_bounce_regression` (felix Kalman ≥0.72), `test_bounce_wasb_regression` (≥0.80), `test_vx_veto`, `test_event_confusion_regression`, `test_sharp_turns`, `test_far_coverage` — verts à chaque itération. La balle felix est le garde-fou (baisser la conf ne doit pas noyer felix de parasites).
- **Zéro hardcoding** : seuils = ratios/paramètres généralisables, pas tunés sur demo3. Anti-surapprentissage (1 clip — vérifie que felix ne se dégrade pas).
- Mesure le gain EN LIVE sur la GT, pas un cache.

## Livrable — CR (OBLIGATOIRE)
`docs/research/ball-track-density-CR.md` : journal d'itérations (coverage overall + cold-start à chaque étape), ce qui a marché (conf ? gating ? wasb ?), le compromis conf-basse-vs-parasites, preuve felix non régressé, l'idée la plus réutilisable. Commit + push. VRAIS chiffres mesurés.
