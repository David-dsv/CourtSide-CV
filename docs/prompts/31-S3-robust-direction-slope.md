# PROMPT — S3 : pente de direction ROBUSTE (tue les vy-flips fantômes dus au jitter) — Backend

> **Session AUTONOME ULTRACODE.** Lance en `/ultracode`.
> ⛔ **NE POSE JAMAIS DE QUESTION. Aucun outil de clarification.** Décide, documente, livre.

## Contexte
Repo CourtSide-CV. Lire `CLAUDE.md`, `docs/pm.md`, le CR
`docs/research/event-rootcause-2026-06-28-CR.md`, et les mémoires
`courtside-hits-overfire-device`, `courtside-bounce-vs-shot-direction`,
`courtside-s3-event-recall-negative`.

**Le bug.** Un rebond fantôme (ex. BOUNCE@540 sur l'extrait) est un artefact de pente
NON-ROBUSTE : `_veto_slope` (`vision/bounce.py:841`) est un moindres-carrés simple SANS
rejet d'outlier. Un seul pic de jitter de +57px sur un arc descendant propre fabrique un
faux `vy_flip` (vy_pre=+54, vy_post=-3.8) → ancre BOUNCE mandatory fantôme. Despiker f540
(médiane) → vy_post=+2.7 → `vy_flip=False` → le fantôme disparaît.

**Le fix.** Rendre la pente de `_direction_features` / `_veto_slope` ROBUSTE (médiane /
Theil-Sen / despike par fenêtre médiane avant fit), scale-free. **Contrefactuel vérifié** :
despike → fantôme parti, conf B→H 1→0, bounce F1 0.625→0.750 (sur l'extrait).

## ⚠️ INPUT CANONIQUE
Benchmarks events sur **`tennis.mp4 -s 73 -d 13`** (PAS `data/output/tennis_demo3.mp4`).
Voir le CR. Mais NOTE : sur l'input canonique le fantôme dominant est différent — mesure
TOI-MÊME quels vy-flips fantômes existent en live avant de coder.

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-s3slope feat/s3-robust-slope
```
Branche depuis `feat/accuracy-overhaul`. ⚠️ édite SOUS `/Users/vuong/Documents/CourtSide-CV-s3slope/...`.
**Touche UNIQUEMENT `vision/bounce.py`.** (Fichier DISJOINT de S1/S2 → parallèle-safe.)

## Le thermomètre
- Replay : `venv/bin/python tools/event_eval/live_pool.py tests/fixtures/methodo/demo3_methodo_inputs_fullvideo.json`
- LIVE : `venv/bin/python run_pipeline_8s.py tennis.mp4 -s 73 -d 13 --device mps --match-mode`
  puis `score_stats.py data/output/tennis_annotated_stats.json`.

## ⚠️ Le piège LOAD-BEARING (lis avant de coder)
`_veto_slope` / `_direction_features` sont utilisés par **felix** (le garde-fou universel).
Un changement de pente robuste DOIT garder :
- `tests/test_bounce_regression.py` (felix Kalman F1 ≥ 0.72),
- `tests/test_bounce_wasb_regression.py` (felix WASB ≥ 0.80),
- `tests/test_vx_veto.py`, `tests/test_event_confusion_regression.py`.
Si une pente robuste fait tomber felix < 0.72 → c'est une RÉGRESSION → abandonne cette
variante. Le contrefactuel PM est un oracle (despike forcé), PAS une implémentation
mesurée : à toi de vérifier qu'une pente robuste GÉNÉRIQUE reproduit le gain SANS supprimer
de vrais vy-flips de rebond ailleurs.

## Contraintes dures
- confusion_H→B = 0 maintenu. Vise à réduire les rebonds fantômes / confusion B→H.
- bounce F1 ≥ 0.625 (canonique) comme plancher.
- Tous les tests de régression verts (surtout felix).
- Zéro hardcoding (fenêtre en fps, seuils en fractions de diag).

## Livrable — CR (OBLIGATOIRE)
`docs/research/s3-robust-slope-CR.md` : chiffres LIVE full-video AVANT/APRÈS, preuve felix
intact (les 2 tests bounce), quels fantômes tués, la variante de pente retenue + pourquoi,
ou un résultat NÉGATIF honnête si aucune pente robuste ne tient felix. Commit + push.
