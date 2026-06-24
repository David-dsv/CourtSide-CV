# PROMPT — Suivi d'identité à travers les changements de côté (Backend)

## Contexte
Repo CourtSide-CV, pipeline Python. Point d'entrée canonique : `run_pipeline_8s.py`.
Lire `CLAUDE.md`, `PROJET.md`, `vision/player_track.py` **en entier**, et
`docs/pm.md` (workflow multi-sessions).

## Le problème
En mode **match**, les joueurs changent de côté entre jeux (odd games). Le
`TwoPlayerTracker` actuel (`vision/player_track.py`) verrouille P1=far / P2=near
par **côté du filet** (`_side()` = feet au-dessus/en-dessous du divider médian
dynamique). C'est drift-proof… tant qu'un joueur ne traverse pas le filet.
Dès qu'il change de côté, **P1/P2 s'inversent** → toutes les stats par côté
(vitesse near, fatigue `analyze_fatigue`, outcome par rally
`classify_rally_outcome`) deviennent FAUSSES au-delà du premier changement de
côté.

## Worktree (OBLIGATOIRE — voir docs/pm.md)
```
git worktree add ../CourtSide-CV-matchmode feat/match-mode
```
Branche : `feat/match-mode` (depuis `feat/accuracy-overhaul`, HEAD courant).
Un commit par étape, push après chaque.

## Objectif
Une **identité joueur stable à travers les changements de côté**, derrière un
flag `--match-mode` (default OFF = comportement actuel inchangé pour
l'entraînement). En match-mode, P1/P2 représentent des **humains stables**,
pas des côtés du filet.

## Approche (géo + apparence, SANS ré-entraîner de modèle)

1. Lire `vision/player_track.py` en entier : comprendre `_side()`, le divider
   médian dynamique, l'association centroid+IoU, le gap-fill, le « lend spare
   to empty slot ».

2. Créer `vision/match_tracker.py` qui wrap/étend `TwoPlayerTracker` :
   - **Détection d'événement side-swap** : un track qui passe d'un côté du
     divider à l'autre ET **y reste** (filtre les poaches éphémères au filet).
     Un side-swap réel = les deux joueurs ont traversé (changement de côté de
     jeu), pas un seul.
   - **Ré-identité par apparence** : signature de couleur de tenue (histogramme
     HSV de la bbox joueur, normalisé) + position. Au moment d'un swap, matcher
     chaque humain du nouveau côté à sa signature pré-swap — celui qui était P1
     garde l'identité P1 même s'il est maintenant côté near.
   - **Fallback** : si l'apparence est ambiguë (tenues similaires), garder
     l'identité par continuité temporelle (humain le plus proche du dernier P1
     avant le swap) et marquer `confidence` basse.

3. Ajouter `--match-mode` au CLI de `run_pipeline_8s.py`. ON → `match_tracker` ;
   OFF → `TwoPlayerTracker` inchangé (zéro régression entraînement).

4. Émettre dans `_stats.json` : `"match_mode": true/false`, et par rally un
   `human_id` (P1/P2 stable) en plus du `player_side` (near/far) existant.

## Contraintes
- **Zéro hardcoding** (CLAUDE.md design principles) : seuils en ratios de frame.
- **Ne PAS casser le mode entraînement** (`--match-mode` OFF = comportement actuel).
- Ne PAS toucher au frontend (session `feat/match-type-ui` gère l'UI).

## Validation + ré-itération
- Teste sur `tennis.mp4` complet (`frame_range [0,11760]`, 50 fps) avec
  `--match-mode`. Vérifie que les side-swaps sont détectés et que l'identité
  P1/P2 **reste stable** à travers.
- Ajoute `tests/test_match_tracker.py` (assert-based, compatible `python3 ...`) :
  - un cas avec swap simulé où l'identité doit survivre,
  - un cas sans swap (comportement = TwoPlayerTracker),
  - un cas de tenues ambiguës (confidence basse).
- **RÉ-ITÈRE** jusqu'à ce que l'identité soit stable sur le clip complet.

## CR à la fin (OBLIGATOIRE — renvoyé au chef de projet)
- L'approche de ré-identité (apparence HSV + continuité) + ses limites.
- Combien de side-swaps détectés sur `tennis.mp4` complet + stabilité P1/P2.
- Signatures exactes (`vision/match_tracker.py`, `--match-mode`, clés JSON
  `match_mode` + `human_id`) pour l'intégration frontend.
- Résultats des tests. Ce qui reste peu fiable (ex. tenues identiques).
- `py_compile` + tests : résultats exacts.
Pousse tout sur `feat/match-mode`. Ne touche pas au frontend.
