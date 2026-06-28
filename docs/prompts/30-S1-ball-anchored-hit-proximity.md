# PROMPT — S1 : proximité de frappe ANCRÉE SUR LA BALLE (le gain dominant vérifié) — Backend

> **Session AUTONOME ULTRACODE.** Lance en `/ultracode`.
> ⛔ **NE POSE JAMAIS DE QUESTION à l'utilisateur. Aucun outil de clarification.**
> Décide seule, documente inline, livre. Une question = run raté.

## Contexte
Repo CourtSide-CV. Lire `CLAUDE.md`, `docs/pm.md`, le CR
`docs/research/event-rootcause-2026-06-28-CR.md` (TON brief complet), et les mémoires
`courtside-hits-overfire-device`, `courtside-s3-event-recall-negative`,
`courtside-events-plateau-ball-tracking-wall`.

**Le bug racine vérifié (frappes fantômes).** Dans `vision/events.py` PASS-0
(`events.py:356`), chaque membre `hit` est inséré avec `xy=(h["x"],h["y"])` = le point de
poignet AUTO-REPORTÉ par `detect_hits`. Ensuite `dmin` (`events.py:375`) et l'ancre
`hit@wrist` (`events.py:419-421`, `dmin<near`) sont calculés À CE POINT DE POIGNET → donc
n'importe quel pic de vitesse de poignet est trivialement « près d'un poignet » par
construction et déclenche une ancre HIT que le firewall ne peut pas supprimer. **C'est
circulaire.**

**Le fix vérifié.** Calculer `dmin` (et le prédicat `hit_like` / l'ancre `hit@wrist`) à la
position RÉELLE DE LA BALLE suivie (`_xy_at(fr)`, `events.py:335`), pas au point de poignet.
Séparation propre mesurée : les vraies frappes GT ont ball-dmin ≤0.51 ; les ancres fantômes
≥0.90.

## ⚠️ INPUT CANONIQUE (lis ça, c'est non négociable)
Le clip de test canonique est **`tennis.mp4 -s 73 -d 13`** (l'invocation réelle de
l'utilisateur), PAS le ré-encodé `data/output/tennis_demo3.mp4` (qui dégrade la pose et
fausse les chiffres de frappe ×4). **Tous tes benchmarks events tournent sur la vidéo
pleine.** (CPU ≡ MPS sur le même input — pas un problème de device.)

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-s1prox feat/s1-ball-anchored-hit-prox
```
Branche depuis `feat/accuracy-overhaul`. Commit/push par itération. ⚠️
`courtside-edit-targets-main-tree` : édite SOUS `/Users/vuong/Documents/CourtSide-CV-s1prox/...`.
**Touche UNIQUEMENT `vision/events.py`.**

## Le thermomètre (LIVE + replay rapide)
- Replay instantané (byte-identique au `_stats.json` live) :
  `venv/bin/python tools/event_eval/live_pool.py tests/fixtures/methodo/demo3_methodo_inputs_fullvideo.json`
- Confirmation LIVE (obligatoire avant de merger) — relance la vraie vidéo pleine :
  ```
  venv/bin/python run_pipeline_8s.py tennis.mp4 -s 73 -d 13 --device mps --match-mode
  venv/bin/python tools/event_eval/score_stats.py data/output/tennis_annotated_stats.json
  ```
- GT : `tests/fixtures/bounces/tennis_demo3.bounces.json` (9) + `shots/tennis_demo3.shots.json` (8).

## Baseline canonique (à battre)
confusion **1/1**, bounce F1 **0.625**, hit F1 **0.500**, pool 14/17 ; spurious
(108,HIT),(231,BOUNCE),(292,HIT),(487,HIT). **Gain attendu mesuré du fix : bounce F1
0.625→0.706** (récupère B570), hit F1 0.500 tenu, confusion tenue. (Sur l'extrait dégradé,
le même fix remonte hit F1 0.133→0.500 — c'est le mécanisme.)

## La boucle (itère SANS plafond)
mesure (live_pool + LIVE) → applique le fix proprement (découple le `xy` de proximité du `xy`
de rep-frame ; pour un membre `hit`, proximité = `_xy_at(fr)`) → re-mesure → garde si mieux.

## Implémentation propre (pas mon proxy)
Mon test PM réécrivait `h["x"],h["y"]` AVANT l'appel, ce qui déplace AUSSI le rep-frame
(effet de bord). TOI, fais le fix CORRECT : ne change pas le `xy` de rep-frame ; ne change que
la valeur passée à `_nearest_player_dist_norm` pour les membres `hit` (utilise la balle).
Garde le rep-frame `xy` tel quel pour l'affichage.

## Contraintes dures
- **confusion_H→B = 0** maintenu (le firewall logique est sacré). Le **+1 B→H résiduel @308**
  est CONNU et HORS-SCOPE (territoire S2) — ne le « répare » pas ici par un hack.
- **Floors** : bounce F1 ≥ 0.625 ET hit F1 ≥ 0.500 sur l'input canonique (tu améliores).
- **Tests verts** : `tests/test_event_confusion_regression.py` (cache 0/0, bounce F1 0.889,
  hit F1 ≥ 0.60 — le fix doit le faire MONTER à ~0.737), `tests/test_bounce_regression.py`
  (felix F1 ≥ 0.72), `tests/test_sharp_turns.py`, `tests/test_vx_veto.py`.
- **Zéro hardcoding** : pas de nouvelle constante (réutilise `near` box-height-normalisé,
  déjà scale-free). Vérifie le plateau `near=0.4-1.0`.
- Ship UNIQUEMENT si LIVE (full-video) ET cache sont d'accord.

## Livrable — CR (OBLIGATOIRE)
`docs/research/s1-ball-anchored-hit-prox-CR.md` : chiffres LIVE full-video AVANT/APRÈS
(confusion + bounce/hit F1), preuve cache montant, felix intact, le diff exact, plateau
`near`. Commit + push. VRAIS chiffres mesurés sur la vidéo pleine.
