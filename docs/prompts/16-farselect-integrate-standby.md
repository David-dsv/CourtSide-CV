# PROMPT — Far-select INTÉGRATEUR (standby) : combiner le meilleur de A+B (Backend) — SESSION 3/5

> **Session AUTONOME ULTRACODE — STANDBY.** Lance en `/ultracode`, MAIS **attends**
> que les sessions 1 (`farselect-conf-motion`) et 2 (`farselect-courtband-ball`)
> aient **poussé leur CR** avant de coder.
> ⛔ **NE POSE JAMAIS DE QUESTION à l'utilisateur. Aucun outil de clarification.**
> Décide seule, documente, ne bloque jamais. Une question = run raté.

## Ton rôle
Les deux sessions far-select ont chacune corrigé le bug de sélection du joueur far dans `vision/pose.py` (chacune sur sa branche) et écrit un CR. **Ton job : lire les DEUX CR, prendre la MEILLEURE combinaison de leurs idées, et coder UN fix intégré propre** qui maximise le far-CORRECT % sur la GT humaine SANS régresser felix.

## Étape 0 — attendre les CR (tu es en standby)
Avant de coder, vérifie que ces deux fichiers existent (pull la branche d'intégration) :
- `docs/research/farselect-conf-motion-CR.md` (session A)
- `docs/research/farselect-courtband-ball-CR.md` (session B)
Si l'un manque, **attends** (le PM te dira, ou re-checke). Lis les deux **en entier** + leur diff de `vision/pose.py` (`git show feat/farselect-conf-motion:vision/pose.py`, idem B).

## Contexte — le diagnostic (mesuré)
Lire `CLAUDE.md`, `PROJET.md`, `docs/pm.md`, mémoire `courtside-far-player-selection-not-detection`. Le joueur far EST détectable (YOLO conf 0.15 8/8) mais le pipeline sélectionne un spectateur. C'est un bug de sélection+seuil dans `vision/pose.py`. Baseline far-CORRECT = 0.9%.

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-farINT feat/farselect-integrated
```
Branche `feat/farselect-integrated` (depuis `feat/accuracy-overhaul`). Commit/push par itération.
⚠️ `courtside-edit-targets-main-tree` : édite SOUS `/Users/vuong/Documents/CourtSide-CV-farINT/...`. **Touche `vision/pose.py`** + un **nouveau** `tests/test_far_coverage.py`.

## Le travail
1. **Combine** les `best_idea` complémentaires des deux CR (ex. conf+mouvement de A, court-band+balle de B) en UNE logique de sélection cohérente. Cherry-pick ou ré-implémente — au plus propre. **Zéro hardcoding** (ratios géométrie/mouvement).
2. **Itère** en mesurant `python tools/pose_gt/measure_far_coverage.py` jusqu'à **battre les DEUX approches individuelles** (ou plateau prouvé). Felix + tous les tests verts à chaque étape.
3. **Ajoute** `tests/test_far_coverage.py` qui asserte far-CORRECT % ≥ un plancher sûr (sous ton chiffre atteint), pour verrouiller le gain.
4. Si les deux approches se contredisent (ex. A baisse felix, B non), choisis la sûre et documente.

## Contraintes dures
- **NE régresse PAS felix** ni aucun test existant — verts à chaque itération.
- Anti-surapprentissage : la logique reste géométrie/mouvement, pas des pixels demo3.

## Livrable — CR (OBLIGATOIRE)
`docs/research/farselect-integrated-CR.md` : quelles idées combinées, far-CORRECT % avant/après vs CHAQUE approche individuelle, preuve felix intact, le nouveau test. Commit + push sur `feat/farselect-integrated`. VRAIS chiffres mesurés.
