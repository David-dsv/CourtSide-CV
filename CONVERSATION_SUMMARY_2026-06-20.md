# CourtSide-CV — Résumé de session (2026-06-20)

> Document de reprise après refresh de conversation. Résume l'état du projet, ce qui a été fait, et ce qui reste à faire. Tout le travail est sur la branche **`feat/accuracy-overhaul`** (poussée sur GitHub `David-dsv/CourtSide-CV`).

---

## 1. Contexte & objectif produit

**CourtSide-CV** = pipeline d'analyse vidéo de tennis. Objectif : un joueur **amateur** filme au téléphone → upload → **data exploitable** (vitesse, profondeur, placement, coups, heatmaps, vidéo annotée).

- Point d'entrée canonique unique : **`run_pipeline_8s.py`**.
- Vidéo de test principale : **`felix.mp4`** (1920×1080 @60fps) — **angle oblique "court-level"** (PAS broadcast standard). C'est le cas d'usage cible mais le plus dur.
- Vérité-terrain rebonds : `tests/fixtures/bounces/felix.bounces.json` (16 rebonds).
- Évaluateur : `tools/bounce_eval/eval_bounces.py`. Contrainte : **pas de GPU** pour entraîner (Mac/MPS, inférence seule, modèles pré-entraînés).

---

## 2. Travail accompli cette session (chronologique)

### a) Audit + recherche SOTA initiale
- Audit multi-agents du système → 2 docs : `AUDIT_2026-06-19.md`, `SOTA_RESEARCH_2026-06-19.md`.
- Verrous identifiés : détection de rebonds cassée (F1 0.30), homographie absente, pas de data structurée en sortie.

### b) Détection de rebonds : F1 0.30 → 0.83 (mesuré end-to-end)
- **Diagnostic** : la balle est dans 100% des détections YOLO brutes ; c'est le **tracking** qui s'effondrait (filtre statique + Kalman).
- **Solution finale** : tracker balle **WASB** (heatmap, `models/wasb_ball_tracker.py`, MIT) + détecteur de rebonds robuste (`vision/bounce.py: detect_bounces_robust`).
- **Leçon clé** : WASB pré-entraîné **exige la normalisation ImageNet** (sinon ~0 détection — bug qui avait fait conclure "NO-GO" à tort).
- Activé via `--ball-tracker wasb` (défaut reste `kalman`). Poids : gdrive `14AeyIOCQ2UaQmbZLNQJa1H_eSwxUXk7z` dans `training/wasb/`.

### c) Homographie court→mètres (`models/court_detector.py`)
- TennisCourtDetector (14 keypoints → mètres ITF → `cv2.findHomography`). Activé via `--homography`.
- ✅ Marche sur **angle broadcast** (`tennis.mp4` : 14/14 keypoints, dims ITF à ~1%).
- ❌ **Échoue sur felix** (oblique) : 3/14 keypoints → fallback échelle scalaire, `speed_source=scalar`, `homography_confidence=none`. `detect_court_zones` retourne aussi **rien** sur felix.

### d) Pose top-down (`vision/pose.py`) — squelettes des 2 joueurs
- Recette : détecter personnes (YOLO) → **crop** → pose haute-résolution par joueur → remap.
- Récupère le **joueur lointain** (avant absent → 16-17/17 keypoints).
- **Filtre anti-parasite** (`is_valid_player`) : exige squelette valide (≥10 kps) + pieds sur le court → 0 parasite (spectateurs/arbitre/ramasseurs éliminés).

### e) Coups + qualité (`vision/shots.py`) — recherche SOTA dédiée
- Doc : `SOTA_RESEARCH_POSE_SHOTS_2026-06-20.md` (RTMPose/rtmlib pour pose, antoinekeller+THETIS pour FH/BH, ATP Shot Quality pour le score).
- `detect_hits` : contact = reversal vertical de la balle **près d'un joueur** (≠ rebond), attribué au joueur.
- `classify_forehand_backhand` : géométrique (côté balle vs corps/poignet).
- `shot_quality` : 0-100 multi-facteurs (vitesse + profondeur + placement, façon ATP).
- Affichage FH/BH + Q au moment du contact. **Export `_stats.json`** structuré (la data exploitable).

### f) Anti-mouvements parasites (2 bugs signalés par l'utilisateur)
- **Balle aberrante "à l'autre bout de la map"** : gate de trajectoire + despike dans `wasb_ball_tracker.py` → sauts aberrants **14 → 0**, densité 83%, F1 inchangé.
- **Personnes parasites** : filtre `is_valid_player` → **0 parasite** sur 44 joueurs.

### g) Minimap 2D (`vision/minimap.py`) — DERNIER ajout, INCOMPLET
- Court vu de dessus en bas à droite + trajectoire balle + points de rebond. Activé par défaut, `--no-minimap` pour désactiver.
- ⚠️ **La projection est INEXACTE sur felix** (voir §4) — c'est le problème en cours.

---

## 3. État actuel — ce qui marche

- **Tracking balle** (WASB) : propre, dense (83%), 0 saut aberrant.
- **Rebonds** : F1 0.83 end-to-end (R 0.94) sur felix. 2 tests de non-régression passent (`tests/test_bounce_regression.py` F1≥0.72, `tests/test_bounce_wasb_regression.py` F1≥0.80).
- **Squelettes** : 2 joueurs bien détectés (top-down), 0 parasite. **L'utilisateur les valide ("les squelettes sont bon").**
- **Coups FH/BH + qualité** : détectés, classés, affichés, exportés en JSON.
- **Vidéo finale actuelle** : `data/output/felix_FINAL_minimap.mp4` (+ `_stats.json`).

---

## 4. FEEDBACK UTILISATEUR EN COURS (à traiter à la reprise)

L'utilisateur a donné 2 retours sur la dernière vidéo, **à faire avant tout nouveau chantier** :

1. **Supprimer les bounding boxes des joueurs.** "la bounding box du player 2 est fixe, oublie les bounding box". → Retirer le dessin des boîtes P1/P2 (garder les squelettes). Probablement via `--no-player-boxes` par défaut, ou retirer l'appel `draw_player_bbox` en Pass 2 de `run_pipeline_8s.py`. NB : la box "fixe" du player 2 vient sûrement du `PlayerTracker` 2-slots qui garde la dernière box quand la détection manque.

2. **🔴 La trajectoire/rebonds sur la minimap ne sont PAS exacts.** "le rebond reste principalement sur la partie haute du terrain et ne traverse même pas tout le terrain". → Le **fallback géométrique** de `project_to_court` (dans `vision/minimap.py`) est trop grossier : il compresse tout dans la moitié haute et ne couvre pas la profondeur du court. 
   - Cause racine : sur felix, pas d'homographie → `project_to_court` utilise une approximation `y_image → profondeur` avec des bornes `top=0.30, bot=0.84` et une correction de perspective `persp = 0.55 + 0.9*depth` qui ne reflètent pas la vraie géométrie de felix.
   - **Piste de fix** : construire une vraie correspondance image→court pour felix. Options : (a) annoter manuellement 4 coins du court sur felix une fois → homographie fixe pour ce clip ; (b) améliorer le détecteur de court pour angle oblique (arXiv 2404.06977, nécessite GPU) ; (c) caler le mapping géométrique de fallback sur des points connus du court de felix (les positions des rebonds GT sont connues : `felix.bounces.json` a x,y image ET depth — on peut calibrer la projection pour que ces points tombent au bon endroit). **L'option (c) est la plus pragmatique sans GPU.**

---

## 5. Roadmap / prochaines étapes (au-delà du feedback)

- Généralisation **homographie aux angles obliques** = le verrou produit n°1 restant (nécessite détecteur de court angle-robuste, fine-tuning GPU).
- Segmentation rallye/point pour stats agrégées par joueur.
- Upgrade pose éventuel : RTMPose via `rtmlib` (MPS, +4 AP) si besoin.
- Palier 2 : moteur propre vidéo→JSON (déjà entamé avec `vision/`), puis web.

---

## 6. Fichiers clés

| Fichier | Rôle |
|---|---|
| `run_pipeline_8s.py` | Point d'entrée unique. Flags clés : `--ball-tracker wasb`, `--homography`, `--no-minimap`, `--no-player-boxes`, `--dump-bounces` |
| `vision/bounce.py` | Détection rebonds (curve-fit + robust) + smoothing |
| `vision/pose.py` | Pose top-down + filtre anti-parasite (`is_valid_player`) |
| `vision/shots.py` | Hits + FH/BH + quality |
| `vision/minimap.py` | **Minimap 2D — projection à corriger (§4.2)** |
| `models/wasb_ball_tracker.py` | Tracker balle WASB + gate anti-aberrant |
| `models/court_detector.py` | Homographie keypoints (broadcast only) |
| `tests/test_bounce_*.py` | Non-régression (lancer après chaque change) |

**Commandes utiles :**
```bash
# Pipeline complet sur felix (config actuelle)
venv/bin/python run_pipeline_8s.py felix.mp4 -s 0 -d 31 --device mps --ball-tracker wasb -o data/output/out.mp4
# Tests de non-régression (rapides, sans GPU)
venv/bin/python tests/test_bounce_regression.py
venv/bin/python tests/test_bounce_wasb_regression.py
# Éval rebonds vs GT
venv/bin/python tools/bounce_eval/eval_bounces.py --pred preds.json --truth tests/fixtures/bounces/felix.bounces.json
```

> ⚠️ Toujours utiliser `venv/bin/python` (torch/MPS y est ; `python` direct échoue). Env fragile : Python 3.14 beta (pyexpat/gdown cassés).
