# PROMPT — Shot-guard « cut » : couper les frames hors-terrain de l'annotation (Backend)

> **Session S2.** Effort recommandé : **xhigh** (logique de détection déjà existante à porter + nouveau découpage propre ; pas de rendu visuel lourd, mais une décision d'architecture timeline à soigner).

## Contexte
Repo CourtSide-CV. Lire `CLAUDE.md`, `PROJET.md`, `docs/pm.md`.
- Un module **shot-guard existe déjà** sur la branche `feat/shot-guard` (worktree présent : `../CourtSide-CV-shotguard`) : `vision/shot_guard.py` + `tests/test_shot_guard.py`. Étudie-le (`git show feat/shot-guard:vision/shot_guard.py`). Il détecte « terrain visible ? » par HoughLinesP (≥40 lignes longues) + hystérésis de dwell (~0.3 s), et expose :
  - `frame_court_visible(frame) -> bool`
  - `compute_court_visibility_mask(frames, fps) -> list[bool]`
  - `non_court_spans(mask) -> list[[start, end]]`
- **MAIS** l'intégration `feat/shot-guard` GARDE toutes les frames (timeline 1:1) et dessine juste « clean » sur les frames hors-terrain. **Ce n'est PAS ce qu'on veut.**

## Décision (ARRÊTÉE par le chef de projet)
**On veut COUPER réellement les frames hors-terrain de la vidéo ANNOTÉE** (replays / close-ups / graphiques disparaissent) pour **ne pas gaspiller de CPU/temps à annoter du hors-jeu**, **tout en gardant la vidéo originale intacte de côté** pour pouvoir ré-aligner.

Conséquence : la vidéo annotée de sortie est **plus courte** que l'originale (mapping frame↔temps différent). Il faut donc émettre un **index de correspondance** annotated-frame → original-frame, pour que le front puisse ré-aligner (S5 le consommera).

Tu possèdes **`vision/shot_guard.py` + `tests/test_shot_guard.py` + un nouveau `vision/annot_clip.py`** en exclusivité. **Tu ne touches PAS `run_pipeline_8s.py`** (c'est S1). Tu fournis un module + des fonctions que S1 (et l'intégration PM) câbleront.

## Worktree (OBLIGATOIRE — voir docs/pm.md)
```
git worktree add ../CourtSide-CV-shotguard-cut feat/shot-guard-cut
```
Branche : `feat/shot-guard-cut` (depuis `feat/accuracy-overhaul`). Commit + push après chaque étape.
> ⚠️ N'utilise PAS le worktree `../CourtSide-CV-shotguard` (c'est l'ancienne branche). Crée le tien.

## Objectif — un module de « cut » réutilisable + sidecar original
Livrer un module qui, à partir du masque court-visibility, produit :
1. La **liste ordonnée des indices de frames À ANNOTER** (uniquement court-visible), pour que la boucle d'annotation saute les autres → **0 CPU dépensé** sur le hors-jeu.
2. Un **clip-index sidecar JSON** `annotated_frame_i -> original_frame_i` (+ fps, total original, total annoté, spans coupés) pour le ré-alignement front.
3. La garantie que **la vidéo originale n'est pas modifiée** (on lit, on ne réécrit pas l'input).

## Étapes

### A. Porter shot_guard.py sur ta branche
1. Récupère `vision/shot_guard.py` et `tests/test_shot_guard.py` depuis `feat/shot-guard` (`git checkout feat/shot-guard -- vision/shot_guard.py tests/test_shot_guard.py`). Garde l'algo (HoughLinesP + dwell) — il est testé et color-independent.

### B. Nouveau module `vision/annot_clip.py`
Expose des fonctions PURES (testables sans vidéo) :
1. `frames_to_annotate(mask) -> list[int]` : indices `i` où `mask[i]` est True (l'ordre temporel).
2. `build_clip_index(mask, fps, start_frame) -> dict` : retourne
   ```
   {
     "fps": float,
     "original_total": int,        # len(mask)
     "annotated_total": int,       # sum(mask)
     "kept_original_frames": [i,...],   # original idx gardés, ordre annoté
     "cut_spans": [[s,e],...],     # spans coupés (relatifs au start), via non_court_spans
     "start_frame": int,
   }
   ```
   `kept_original_frames[k]` = l'index original de la k-ème frame annotée → c'est le mapping de ré-alignement.
3. `original_to_annotated(clip_index, original_frame) -> int | None` et l'inverse `annotated_to_original(clip_index, annotated_frame) -> int` (helpers de conversion).

### C. Contrat pour S1 / l'intégration
Documente clairement (docstring + CR) comment S1 doit câbler :
- Calculer le masque (Pass 0.5) → `frames = frames_to_annotate(mask)`.
- La boucle d'annotation **n'écrit que ces frames** (et ne fait pose/ball/overlay que sur elles) → CPU économisé.
- Écrire `build_clip_index(...)` à côté du `_stats.json` (ex. `<out>_clipindex.json`).
- L'original n'est jamais réécrit.
Fournis une **fonction d'intégration optionnelle** que S1 pourra appeler, mais ne modifie pas `run_pipeline_8s.py` toi-même.

## Validation + ré-itération (jusqu'à parfait)
- `tests/test_shot_guard.py` passe (porté).
- **Nouveau** `tests/test_annot_clip.py` (assert-based, sans vidéo) :
  - masque tout-True → `kept == range(n)`, aucun cut, mapping identité.
  - masque avec un trou → frames coupées absentes, `annotated_total` correct, round-trip `original_to_annotated ∘ annotated_to_original` cohérent sur les frames gardées, et `None` sur une frame coupée.
  - bornes : masque vide, masque tout-False.
- `python -m py_compile vision/annot_clip.py vision/shot_guard.py`.
- Ré-itère jusqu'à ce que tous les tests passent et que le mapping soit exact.

## CR à la fin (OBLIGATOIRE)
- L'API exacte de `vision/annot_clip.py` (signatures + format du clip-index JSON).
- **Le mode d'emploi pour S1** : où appeler quoi dans Pass 0.5 / la boucle d'annotation / l'écriture sidecar, et la garantie « original non modifié ».
- **Le contrat pour S5 (front)** : format du `_clipindex.json` et comment ré-aligner annotated↔original/temps.
- Honnête : limites (détection HoughLinesP qui peut rater des angles amateurs ; sur felix = 100 % court donc 0 cut — c'est normal).
- Résultats des tests + py_compile.
Pousse tout sur `feat/shot-guard-cut`. NE touche PAS `run_pipeline_8s.py`.
