# PROMPT — Câbler la méthodo d'events en PROD + régénérer (Backend, chirurgie pipeline)

> **Session d'IMPLÉMENTATION (intégration prod).** Effort recommandé : **ultracode**.
> Modifie le pipeline canonique → un bug casse felix/match-mode/HUD. **Re-valide
> end-to-end, itère jusqu'à zéro régression.**

## Contexte — lire d'abord
Repo CourtSide-CV. Lire `CLAUDE.md`, `PROJET.md`, `docs/pm.md`, et :
- `docs/research/ball-tracking-density-CR.md` + `docs/research/bounce-vs-shot-methodo-CR.md` (ce que fait la méthodo).
- `vision/events.py` : `classify_events(...)` (signature complète ~l.278), `detect_sharp_turns`, `detect_turning_points`. **FIGÉ — ne pas le modifier** (juste l'appeler).
- `run_pipeline_8s.py` : l'ordre actuel des passes (voir ci-dessous).

## Ce qui est prouvé (sur le benchmark `tools/event_eval/run_demo3.py`)
`classify_events` (avec `detect_sharp_turns`) → **rebond F1 0.842, recall 8/9, confusion_H→B=0 / B→H=0** sur demo3. Mais **PAS câblé** : aujourd'hui `run_pipeline_8s.py` produit `bounce_events` (Pass 1) et `shot_by_frame` (Pass 1.5) séparément, **sans arbitrage** → la vidéo ne montre pas le gain. **But : brancher `classify_events` comme couche d'arbitrage unifiée, derrière un flag, sans rien régresser.**

## Le problème d'ORDRE (la raison du non-câblage)
`classify_events` a besoin EN MÊME TEMPS de :
- les **candidats rebond** (`detect_bounces_robust`, Pass 1 ~l.1110),
- les **candidats frappe** (`detect_hits`, dans Pass 1.5 ~l.1298),
- la **trajectoire pré-spline + masque** (`raw_ball_centers` ~l.1022, `ball_is_real` ~l.1031),
- les **poses** `players_per_frame` (produites en Pass 1.5 ~l.1236, APRÈS la détection rebond).

Ordre actuel : Pass 1 (ball+bounce) → Pass 1.5 (pose+hits) → Pass 2 (annotate). Donc l'arbitrage doit s'insérer **APRÈS Pass 1.5** (quand poses ET candidats existent), et **remplacer** les `bounce_events` / `shot_by_frame` consommés par Pass 2 par la sortie unifiée de `classify_events`.

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-wire feat/wire-event-methodo
```
Branche `feat/wire-event-methodo` (depuis `feat/accuracy-overhaul`). Commit/push par étape.
⚠️ [[courtside-edit-targets-main-tree]] : édite sous `/Users/vuong/Documents/CourtSide-CV-wire/...`.

## Le câblage (étapes)

### 1. Flag `--event-methodo` (défaut OFF au début)
Ajouter le flag. **OFF = comportement actuel strictement inchangé** (felix/prod intacts). ON = chemin d'arbitrage. Une fois prouvé non-régressif, on pourra discuter le défaut.

### 2. Insérer l'arbitrage après Pass 1.5
Après que `players_per_frame` ET les candidats existent :
- garder les candidats rebond bruts de Pass 1 (NE PAS les filtrer/veto-er en amont quand le flag est ON — `classify_events` arbitre) ;
- générer les `turning_frames` (`detect_turning_points`) + passer `raw_ball_centers`, `ball_is_real`, `all_ball_centers` (smoothed), `players_per_frame`, et la trajectoire WASB si dispo (`wasb_centers`/`wasb_is_real`) ;
- appeler `classify_events(...)` → liste d'events labellisés `{frame, x, y, label∈{BOUNCE,HIT}}` + rapport cohérence WASB.
- **Mapper la sortie** vers les structures que Pass 2 consomme : `bounce_by_frame` (events BOUNCE) et `shot_by_frame` (events HIT, avec le `fhb`/side dérivé de `detect_hits` quand l'event coïncide avec un hit candidat — sinon HIT générique). ⚠️ Préserver les clés exactes que Pass 2 + le HUD par-échange + la minimap + `_stats.json` attendent (côté, fhb, quality, speed).

### 3. Préserver les chemins spéciaux
- **`--demo-override`** : si actif, il gagne (events manuels) — ne pas le court-circuiter.
- **match-mode** : `human_id` sur shots/rallies doit rester correct (les events HIT alimentent toujours `shot_by_frame`).
- **rally segmentation / HUD ÉCHANGE N** : lit `shot_by_frame`/`bounce_by_frame` → vérifier que le mapping préserve la numérotation.
- **`direction_state`** dans `_stats.json` : alimenté par la décision de `classify_events` (metric/unconfirmed).

### 4. Le flag par défaut
Quand ON est prouvé non-régressif sur felix ET meilleur sur demo3, **proposer** de basculer le défaut à ON (le PM tranchera) — ou laisser OFF avec doc. Documenter le choix.

## Validation + ITÉRATION (le cœur)
- `py_compile`.
- **Tests existants TOUS verts** (flag OFF = bit-identique) : `test_bounce_regression`, `test_bounce_wasb_regression`, `test_vx_veto`, `test_event_confusion_regression`, `test_sharp_turns`, `test_shots_regression`, `test_match_tracker`, `test_annot_clip`, `test_shot_guard`.
- **Run end-to-end réel** flag ON sur le clip benchmark : `python run_pipeline_8s.py tennis.mp4 -s 73 -d 13 --device cpu --event-methodo -o /tmp/wire_demo3.mp4`. Comparer le `_stats.json` bounces/shots aux GT demo3 (`tools/event_eval/`). **Cible : la confusion H→B=0 et le rappel rebond se retrouvent dans le `_stats.json` de prod** (pas juste sur le cache).
- **Run felix flag ON** : `tennis`/`felix` ne doit pas régresser end-to-end (la méthodo a besoin de poses — vérifier le fallback propre quand poses pauvres : felix prod reste sur l'ancien chemin si le flag est OFF, et si ON, ne doit pas casser).
- Itère jusqu'à : flag OFF bit-identique + flag ON reproduit le gain demo3 dans le `_stats.json` réel + zéro régression.

## CR à la fin (OBLIGATOIRE)
- Où l'arbitrage est inséré (l.) + comment la sortie mappe vers `bounce_by_frame`/`shot_by_frame`.
- Preuve : `_stats.json` du run réel demo3 flag ON → confusion H→B / rebond recall vs GT (via event_eval).
- Confirmation flag OFF = bit-identique (tous les tests verts, valeurs exactes).
- Comment demo-override / match-mode / HUD / minimap sont préservés.
- Reco sur le défaut du flag (ON/OFF) avec justification.
- Honnête : cas où la méthodo dégrade vs l'ancien chemin (felix pauvre en poses ?), à garder OFF si besoin.
Pousse sur `feat/wire-event-methodo`. Fichiers : `run_pipeline_8s.py` (principal), éventuellement `vision/events.py` UNIQUEMENT si un adaptateur est nécessaire (sinon ne pas y toucher). **Le PM régénérera la vidéo après merge.**
