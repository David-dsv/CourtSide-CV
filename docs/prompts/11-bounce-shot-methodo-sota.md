# PROMPT — Méthodo SOTA rebond vs frappe : classifier par direction + cohérence WASB, itérer sur benchmark (Backend)

> **Session d'IMPLÉMENTATION + R&D mesurée.** Effort recommandé : **ultracode**.
> **Objectif explicite de l'utilisateur : viser le SOTA — la meilleure app du marché.**
> Tu DOIS **itérer (mesurer → améliorer → re-mesurer) tant que le score sur le
> benchmark n'est pas EXCELLENT.** Ne t'arrête pas à « ça compile ».

## Contexte — lire d'abord
Repo CourtSide-CV. Lire `CLAUDE.md`, `PROJET.md`, `docs/pm.md`, et **en entier** :
- `docs/research/bounce-vs-shot-direction.md` (la physique mesurée : rebond préserve vx-signe + flip vy ; frappe inverse vx).
- Le code de détection : `vision/bounce.py` (`detect_bounces_robust` ~l.484, `vx_flip_veto` ajouté récemment), `vision/shots.py` (`detect_hits` ~l.200, swing poignet), `run_pipeline_8s.py` (appels rebond ~l.1100, frappe ~l.1270, `raw_ball_centers` pré-spline ~l.860/1022).

## Le problème (validé par l'utilisateur, en regardant la vraie vidéo)
La détection **confond encore frappe et rebond** ; le veto vx-flip actuel (one-directional, ne fait que *rejeter*) **améliore peu en pratique** et l'utilisateur a « l'impression qu'on dégrade le système ». Il faut une **méthodo plus forte**, pas un patch.

## La méthodo cible (décisions ARRÊTÉES par le PM)

### 1. Classifier CHAQUE événement, puis corriger par alternance (hybride)
Au lieu de deux détecteurs orthogonaux + un veto, traiter chaque **changement de direction de la balle** comme un *événement candidat* à **classifier** en `BOUNCE` ou `HIT` :
- **Features par événement** : vx-flip (oui/non), vy-flip (oui/non), proximité au joueur le plus proche (normalisée hauteur boîte), hauteur image / Ym métrique si homographie, énergie de restitution (déjà présente, `hit_ratio_max`).
- **Règle physique de départ** (à raffiner empiriquement) : `HIT` = vx s'inverse OU swing poignet coïncide OU balle très près d'un joueur ; `BOUNCE` = vx préservé + vy-flip + loin des deux joueurs au sol. **L'utilisateur suggère** : le 1er changement de direction d'un échange tend à être un rebond, puis ça alterne — **utiliser cette alternance comme POST-CORRECTION, pas comme ancrage dur**.
- **Post-correction par alternance** : le long d'un rallye, les événements devraient alterner HIT↔BOUNCE. Si on voit `BOUNCE, BOUNCE` consécutifs (sans HIT entre), l'un est probablement une frappe ratée → ré-examiner avec un seuil plus permissif. Idem `HIT, HIT`. **L'alternance est un signal de confiance + un correcteur, jamais une vérité imposée** (un événement raté ne doit pas décaler toute la cascade).

### 2. Cohérence avec WASB (le « 2e modèle de détection » de l'utilisateur)
WASB (trajectoire heatmap dense, déjà dans le repo, `--ball-tracker wasb`, F1 bounce 0.865) est la **2e source indépendante**. Dériver les changements de direction depuis la trajectoire WASB dense et **mesurer le % de cohérence** avec notre classification géométrique. **Un faible % de cohérence = signal de décalage** (event mal classé ou raté). Utiliser ce % comme (a) métrique de santé, (b) déclencheur de ré-examen des events incohérents. Zéro nouvelle dépendance, 100% local-first.

### 3. La session DÉCIDE sur preuve benchmark
Mesurer le système ACTUEL (veto inclus) sur le benchmark ci-dessous. Puis garder / raffiner / refondre **selon ce qui score le mieux**. **Contrainte dure : tu dois BATTRE le baseline mesuré, jamais le dégrader.** Si une idée baisse le score, jette-la.

## LE BENCHMARK (existe déjà — l'utilisateur l'a annoté à la main)
`data/output/tennis_demo3_annotations.json` = **ground-truth manuelle sur un clip de tennis.mp4** (source @73s, 650 frames @50fps, offset 3650, 1920×1080) :
- `bounces_true` : **9 rebonds** vérifiés frame-by-frame (`demo_frame`, `source_frame`, `xy`).
- `strikes` : **8 frappes** avec label `forehand`/`backhand` (`demo_frame`, `source_frame`, `xy`, `type`).
- Le clip `data/output/tennis_demo3.mp4` existe (sinon régénérer : `tennis.mp4 -s 73 -d 13`).

### CHANTIER A — Formaliser le benchmark (au standard du repo)
1. Convertir cette GT en fixtures évaluables, miroir de `tests/fixtures/bounces/felix.bounces.json` :
   - `tests/fixtures/bounces/tennis_demo3.bounces.json` (9 rebonds, format `{video,fps,frame_offset,bounces:[{frame,x,y}]}`).
   - **NOUVEAU** `tests/fixtures/shots/tennis_demo3.shots.json` (8 frappes, `{...,shots:[{frame,x,y,type}]}`).
   - Attention aux conventions de frame : `demo_frame` (0-based dans le clip) vs `source_frame` (absolu tennis.mp4). Documenter laquelle le pipeline produit et aligner.
2. Committer un **cache de détection** (centres balle pré-spline + masque is_real + poses) pour ce clip, pour une régression rapide SANS GPU/vidéo (comme `felix_wasb_centers.json`).

### CHANTIER B — Évaluateur de CONFUSION rebond↔frappe
Étendre `tools/bounce_eval/` (ou un nouveau `tools/event_eval/`) pour produire la **matrice de confusion** au matcher greedy 1-1 (tol `round(0.15*fps)=8` @50fps) :

|  | Prédit REBOND | Prédit FRAPPE | Manqué |
|---|---|---|---|
| **GT REBOND** | TP_b | confusion B→H | FN_b |
| **GT FRAPPE** | **confusion H→B** ★ | TP_h | FN_h |

★ `confusion_H→B` = frappes émises comme rebonds = **exactement le bug utilisateur** → métrique cible n°1. Règle de désambiguïsation déterministe (frame la plus proche ; égalité → label prédit ; greedy 1-1) — spécifiée dans `docs/research/bounce-vs-shot-direction.md` §5.2, l'appliquer.

### CHANTIER C — Méthodo (classify + alternance + cohérence WASB)
Implémenter la méthodo §1+§2. Itérer les seuils/règles **contre le benchmark**. Respecter l'invariant de sécurité existant (lire la direction sur les centres **pré-spline + masque**, jamais le spline ; s'abstenir sur trous/épars).

### CHANTIER D — Itération jusqu'à score EXCELLENT
Boucle obligatoire : mesurer (B) → identifier les events faux → raffiner (C) → re-mesurer. **Cible de réussite (SOTA-viseur)** :
- `confusion_H→B == 0` (aucune frappe affichée comme rebond) — **non négociable**.
- `confusion_B→H == 0` idéalement (aucun rebond affiché comme frappe).
- Rebond F1 ≥ baseline mesuré (ne PAS régresser ; viser > 0.90).
- Frappe (shot) F1 élevé (mesurer le baseline `detect_hits` sur les 8 strikes, puis battre).
- % de cohérence WASB↔géométrie rapporté et en hausse.
Si tu n'atteins pas `confusion_H→B == 0`, **continue d'itérer** ou documente précisément le cas physique irréductible qui bloque.

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-methodo feat/bounce-shot-methodo
```
Branche `feat/bounce-shot-methodo` (depuis `feat/accuracy-overhaul`). Commit/push après chaque chantier.
⚠️ [[courtside-edit-targets-main-tree]] : édite sous `/Users/vuong/Documents/CourtSide-CV-methodo/...`.

## Contraintes de design (CLAUDE.md)
- **Zéro hardcoding** : tout seuil = ratio fps/dimensions/géométrie. Pas de magie tirée de demo3 (sinon tu sur-apprends le benchmark — valider aussi que felix ne régresse pas).
- **Généralisation > précision** : la méthodo doit tenir sur n'importe quelle vidéo, pas juste demo3.
- **Ne régresse RIEN** : `tests/test_bounce_regression.py` (Kalman ≥0.72), `tests/test_bounce_wasb_regression.py` (0.865), `tests/test_vx_veto.py`, `tests/test_shots_regression.py` doivent tous rester verts (ou être améliorés en connaissance de cause).

## Validation + ré-itération (le cœur de la mission)
- `python -m py_compile` sur tous les fichiers touchés.
- Le nouvel évaluateur de confusion tourne sur le cache demo3 → matrice + F1.
- **Mesure AVANT/APRÈS** documentée à chaque itération (le journal de progression est exigé dans le CR).
- Anti-surapprentissage : re-mesurer felix (bounce F1) à chaque grosse itération — si felix régresse, la méthodo généralise mal.
- **Itère jusqu'à atteindre les cibles de CHANTIER D, ou jusqu'à preuve qu'une cible est physiquement inatteignable** (documenter le cas).

## CR à la fin (OBLIGATOIRE)
- **Le journal d'itérations** : baseline mesuré → chaque changement → score résultant (la preuve que tu as itéré, pas juste codé une fois).
- La matrice de confusion finale sur demo3 (`confusion_H→B`, `confusion_B→H`, F1 rebond, F1 frappe) + le % cohérence WASB.
- Confirmation que felix ne régresse pas (bounce F1) + tous les tests existants verts.
- La méthodo finale retenue (classify+alternance+WASB) et **pourquoi elle bat le baseline** (chiffres).
- Honnête : cas non résolus, sur-apprentissage potentiel (1 seul clip GT), ce qui demanderait plus de clips annotés.
- py_compile + tous les tests : résultats exacts.
Pousse tout sur `feat/bounce-shot-methodo`. Fichiers : `vision/bounce.py`, `vision/shots.py`, `run_pipeline_8s.py`, `tests/fixtures/{bounces,shots,cache}/`, `tools/`, `tests/test_*`.
