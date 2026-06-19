# SOTA — Pose, classification de coups, qualité de frappe (2026-06-20)

> Recherche vérifiée (deep-research, claims confirmés contre sources primaires). Contexte : pas de GPU pour entraîner, inférence Mac/MPS, modèles pré-entraînés, angle amateur.

## Brique 1 — Pose joueurs

| Modèle | COCO AP | Vitesse | MPS ? | Verdict |
|---|---|---|---|---|
| **RTMPose-m** (via `rtmlib`) | **75.8** | 90+ FPS CPU i7 | ✅ `device='cpu'/'cuda'/'mps'` | **Upgrade recommandé** — lib légère (onnxruntime, pas de mmcv), MPS natif |
| RTMO-l (one-stage) | 74.8 | 141 FPS GPU | via rtmlib | latence indép. du nb de personnes |
| YOLO26x-pose | 71.6 | — | ✅ ultralytics | actuel (YOLOv8m-pose plus bas) |
| Sapiens (Meta) | très haut | lourd (0.3-2B, 1024²) | possible | trop lourd pour notre cas |
| ViTPose-H | 79.1 | lourd | — | meilleur AP mais lourd |

- **Verdict pose** : notre approche **top-down (détection→crop→pose hi-res)** déjà implémentée (`vision/pose.py`) résout le joueur lointain (16-17/17 kps). Upgrade possible : **`rtmlib`+RTMPose-m** (MPS, +4 AP vs YOLO26x, lib légère). À tester si la pose actuelle reste insuffisante.
- Le motion blur est un mode d'échec quantifié (arXiv 2309.01010) — le crop hi-res l'atténue.

## Brique 2 — Classification coup droit/revers

| Approche | Perf | Réutilisable sans GPU ? |
|---|---|---|
| **`antoinekeller/tennis_shot_recognition`** | ~80% (FC sur pose MoveNet), 4 classes (FH/BH/serve/neutral) | ✅ pose features → classifieur léger |
| Fuzzy ST-GCN (Sensors 2020, THETIS) | **82.2%+** | ML plus lourd |
| THETIS dataset | 8374 séquences, 12 classes | données d'entraînement si besoin |

- **Verdict FH/BH** : approche **géométrique pose-based** = chemin pragmatique SOTA, sans GPU. Le repo keller prouve ~80% avec de simples features de pose. **Discriminant clé** : côté de la balle au contact relatif au corps du joueur (poignet/épaule via pose) + main dominante (droitier par défaut). FH = balle du côté de la main dominante, BH = côté opposé.

## Brique 3 — Score de qualité de frappe

- **Standard pro = ATP/TennisViz "Shot Quality"** (vérifié) : note **0-10** sur serve/return/FH/BH, calculée à partir de **vitesse, spin, profondeur (depth), largeur/placement (width), hauteur d'impact**. Calibrée sur la probabilité de gagner le point (qualité 2 → 22% ; qualité 8 → 77%).
- **Verdict qualité** : remplacer le `Q = 0.4·vitesse + 0.6·profondeur` actuel par une formule multi-facteurs inspirée d'ATP : **vitesse + profondeur + placement (largeur/distance aux lignes) + dégagement du filet**. Spin non mesurable sans haute fréquence → l'omettre. Normaliser 0-100. Plus défendable, et exploite l'homographie (mètres réels) quand dispo.

## Plan d'implémentation (ordre de dépendance)
1. **Attribution coup→joueur + frame de contact** : segmenter par rebonds, hit = reversal vertical de la balle near un joueur (pose donne la position). 
2. **FH/BH géométrique** : au contact, comparer position balle vs corps du joueur (pose).
3. **Qualité multi-facteurs** : vitesse + profondeur + placement + filet, normalisé 0-100.
4. **Panneau stats récap** + affichage par coup.

Sources clés : RTMPose arXiv 2303.07399 · rtmlib `Tau-J/rtmlib` · `antoinekeller/tennis_shot_recognition` · THETIS `github.com/THETIS-dataset/dataset` · Fuzzy ST-GCN Sensors 2020 PMC7662764 · ATP Shot Quality atptour.com/insights-shot-quality.
