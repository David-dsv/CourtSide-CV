# PROMPT — Confusion rebond/frappe S2 : pousser la méthodo au-delà de 0.842 (Backend) — SESSION 2/3

> **Session AUTONOME ULTRACODE.** Lance en `/ultracode`.
> ⛔ **NE POSE JAMAIS DE QUESTION à l'utilisateur. Aucun outil de clarification.**
> Décide seule, documente, ne bloque jamais. Une question = run raté.

## Contexte
Repo CourtSide-CV. Lire `CLAUDE.md`, `PROJET.md`, `docs/pm.md`, et les mémoires `courtside-bounce-shot-methodo`, `courtside-ball-density-apex-bounce`, `courtside-event-methodo-prod-gap`.

**Le classifieur `classify_events` (`vision/events.py`)** donne déjà confusion **0/0** sur le cache demo3 mais reste imparfait :
- **bounce F1 0.842** (TP=8/9, **1 raté = f174**, 2 FP)
- **hit F1 0.632** (TP=6/8, 2 ratés = f333/f525, 2 FP)
Ton job : **pousser ces F1 plus haut SANS jamais casser le 0/0 de confusion** (le firewall logique `hit_like ⇏ bounce` est sacré).

## CE QUI A CHANGÉ (un nouveau signal disponible)
La pose du joueur far vient de passer **0.9% → 84.9%** (far-select mergé) ET la densité de squelette far **18% → 84%**. Donc les **poignets du joueur far** sont maintenant disponibles → de nouveaux signaux pour récupérer les frappes far ratées (f333/f525 sont peut-être des contacts far auparavant invisibles). Exploite ça.

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-methodoimp feat/event-methodo-improve
```
Branche `feat/event-methodo-improve` (depuis `feat/accuracy-overhaul`). Commit/push par itération.
⚠️ `courtside-edit-targets-main-tree` : édite SOUS `/Users/vuong/Documents/CourtSide-CV-methodoimp/...`. **Touche UNIQUEMENT `vision/events.py`** (le classifieur figé ailleurs, à toi de l'améliorer ici).

## Le benchmark + l'évaluateur
- GT : `tests/fixtures/bounces/tennis_demo3.bounces.json` (9 rebonds) + `tests/fixtures/shots/tennis_demo3.shots.json` (8 frappes).
- Évaluateur : `python tools/event_eval/run_demo3.py` → matrice de confusion + F1 (sur le cache committé `demo3_event.json`).
- ⚠️ Le cache existant a une pose far à 78% (ancienne capture). Si tu veux exploiter la NOUVELLE pose far dense, régénère un cache via `scripts/build_demo3_event_cache.py` (qui utilisera le far-select mergé) — documente que tu as fait ça.

## La boucle (itère SANS plafond)
mesure (confusion + F1) → identifie un event faux (f174 raté ? un FP ?) → change UNE chose dans `classify_events` → re-mesure → garde si mieux. **Le firewall 0/0 ne doit JAMAIS se rompre** (si une idée crée de la confusion, jette-la).

## Pistes (à valider par la mesure, pas à coder aveuglément)
- **f174 raté** (rebond) : le CR `ball-density-apex` dit que 120/369 étaient des apex far sans extremum y → sharp-turns les a récupérés. f174 est-il dans le pool de candidats ? Si non = trou de génération (pas classification). Si oui = mal classé.
- **f333/f525 ratés** (frappes far) : maintenant que la pose far est dense, le poignet far est-il disponible à ces frames ? → nouvel ancre `hit_like`.
- **FP résiduels** : sur-génération de sharp-turns ? Affiner le seuil d'angle (plateau 60-75° connu) ou la fusion candidats.

## Contraintes dures
- **confusion_H→B = 0 ET confusion_B→H = 0** à chaque itération (non négociable).
- **bounce F1 ≥ 0.842 et hit F1 ≥ 0.632** comme planchers (tu améliores, jamais régresses).
- **NE régresse PAS** felix ni les tests (`test_bounce_regression`, `test_bounce_wasb_regression`, `test_vx_veto`, `test_event_confusion_regression`, `test_sharp_turns`, `test_far_coverage`).
- Zéro hardcoding ; anti-surapprentissage (1 clip GT — vérifie le plateau).

## Livrable — CR (OBLIGATOIRE)
`docs/research/event-methodo-improve-CR.md` : journal d'itérations (F1 bounce/hit + confusion à chaque étape), ce qui a récupéré f174/f333/f525, preuve confusion 0/0 maintenue + felix intact, l'idée la plus réutilisable. Commit + push. VRAIS chiffres mesurés.
