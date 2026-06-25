# PROMPT — Minimap radar : mode « un seul rebond » (Backend dessin)

> **Session S3.** Effort recommandé : **high** (changement de dessin ciblé et bien borné dans un seul fichier ; validation visuelle simple).

## Contexte
Repo CourtSide-CV. Lire `CLAUDE.md`, `PROJET.md`, `docs/pm.md`, puis :
- `vision/minimap.py` — tu le possèdes **en exclusivité**. Fonction principale `draw_minimap(frame, ball_trail_img, bounces_img, frame_w, frame_h, homography=..., players_img=..., pulse=..., ground_path=..., now_frame=...)` (~ligne 255).
- L'appelant (NE le modifie pas, c'est S1) : `run_pipeline_8s.py:1213`.
- Aujourd'hui : les rebonds sont dessinés en **heatmap cumulative** (`_draw_heat`, lignes 240–252 + 345–356) + un **glow marker par rebond** (lignes 428–431). Donc **tous** les rebonds restent affichés.

## Le problème (décision user, ARRÊTÉE)
Sur le **court radar**, on ne doit voir **qu'UN SEUL rebond** : le dernier survenu. Pas d'accumulation, pas de heatmap de tous les rebonds. Quand un nouveau rebond arrive, il **remplace** l'ancien.

## Worktree (OBLIGATOIRE — voir docs/pm.md)
```
git worktree add ../CourtSide-CV-minimap feat/minimap-single-bounce
```
Branche : `feat/minimap-single-bounce` (depuis `feat/accuracy-overhaul`). Commit + push après chaque étape.

## Objectif — un mode « single/latest bounce » propre
Ajouter un paramètre à `draw_minimap` qui, quand activé, dessine **au plus un** rebond (le dernier de `bounces_img`), avec un **marqueur propre et lisible** (pas la heatmap cumulative).

## Contrat d'API (à respecter EXACTEMENT — S1 code contre ça)
Nouvelle signature **rétro-compatible** :
```python
def draw_minimap(frame, ball_trail_img, bounces_img, frame_w, frame_h,
                 homography=None, players_img=None, pulse=0.0,
                 ground_path=None, now_frame=None,
                 single_bounce=False):
```
- `single_bounce=False` (défaut) → **comportement legacy inchangé** (heatmap + tous les markers). Ne casse aucune autre session.
- `single_bounce=True` → ignore la heatmap cumulative ; dessine **uniquement le DERNIER** élément de `bounces_img` (S1 passera déjà une liste à 0 ou 1 élément, mais sois robuste : si la liste en contient plusieurs, prends le dernier). Marqueur unique mis en valeur (glow + anneau de pulsation + couleur depth). Liste vide → ne dessine aucun rebond.

## Étapes
1. Ajoute le paramètre `single_bounce` (défaut False).
2. Branche : si `single_bounce`, saute `_draw_heat` pour les rebonds et la boucle multi-markers ; dessine le seul dernier rebond avec un marqueur premium (réutilise `fx.glow_marker` + un anneau, couleur depth via `BOUNCE_COLORS`).
3. Garde le **ground-path arc** (trajectoire lissée) et les **player pucks** intacts — ils ne sont pas concernés par #4. (Si tu juges que l'arc cumulatif gêne la lisibilité « un seul rebond », documente-le dans le CR mais NE le supprime pas sans accord — c'est une autre feature.)
4. Vérifie que le titre « COURT RADAR » / le flag `~approx` (homographie basse confiance) restent inchangés.

## Validation + ré-itération (jusqu'à parfait)
- `python -m py_compile vision/minimap.py`.
- **Test de rendu isolé** : écris un petit script jetable (`/tmp/test_minimap.py`, ne pas committer) qui appelle `draw_minimap` sur une frame noire avec `bounces_img=[]`, puis `[(x,y,"deep")]`, puis `[(x1,y1,"short"),(x2,y2,"deep")]` en `single_bounce=True`, et sauve les PNG. Vérifie à l'œil : 0 marker / 1 marker / 1 marker (le dernier) respectivement.
- Vérifie aussi `single_bounce=False` → rendu legacy identique (non-régression).
- Ré-itère jusqu'à ce que le marqueur unique soit net et bien mis en valeur.

## CR à la fin (OBLIGATOIRE)
- La signature finale de `draw_minimap` (+ confirmation rétro-compatibilité `single_bounce=False`).
- Comment le marqueur unique est rendu (couleur/forme/pulse) — joins les PNG de test ou décris.
- Ce que tu as gardé (ground-path, pucks) vs changé.
- Honnête : tout effet de bord (ex. l'arc cumulatif qui peut encore « raconter » d'anciens rebonds).
- Résultats py_compile + test de rendu.
Pousse tout sur `feat/minimap-single-bounce`. NE touche QUE `vision/minimap.py`.
