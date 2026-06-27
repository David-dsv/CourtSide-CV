# PROMPT — Réconcilier S1+S2 dans le firewall events.py (Backend) — SESSION RÉCONCILIATION

> **Session AUTONOME ULTRACODE.** Lance en `/ultracode`.
> ⛔ **NE POSE JAMAIS DE QUESTION à l'utilisateur. Aucun outil de clarification.**
> Décide seule, documente, ne bloque jamais. Une question = run raté.

## Contexte — le problème précis
Repo CourtSide-CV. Deux sessions ont amélioré le classifieur d'events `vision/events.py` (méthodo `--event-methodo`), mais leurs changements **collisionnent sur la même ligne `hit_like`** et, mal fusionnés, **cassent le firewall**.

Lire `CLAUDE.md`, `PROJET.md`, `docs/pm.md`, et les CR :
- `docs/research/event-methodo-prod-CR.md` (S1)
- `docs/research/event-methodo-improve-CR.md` (S2)
+ mémoires `courtside-event-methodo-prod-resolved` (si présente), `courtside-event-methodo-prod-gap`, `courtside-bounce-shot-methodo`.

## Les deux contributions (mesurées séparément)
- **S1** (`feat/event-methodo-prod`, DÉJÀ MERGÉ dans la base) : **vx-gate de l'ancre vy**. `vy_bounce = vy and not vx_flip` → une frappe far dont la balle s'inverse verticalement (vy_flip) ne peut plus ancrer comme rebond. **Résultat : confusion 0/0 LIVE en prod**, bounce F1 0.667, hit F1 0.750. `--event-methodo` passe ON par défaut.
- **S2** (`feat/event-methodo-improve`) : **`detect_wrist_hits`** (générateur de candidats frappe par poignet far, normalisé par la hauteur de boîte) + **relaxation de proximité** `bounce_traj = turn and (y_max or vx_preserved) and not hit` (récupère un rebond qui tombe au pied d'un joueur, f174). **Résultat sur cache : bounce F1 0.842→0.900 (9/9), hit F1 0.632→0.700, confusion 0/0.**

## LE BUG À RÉSOUDRE (mesuré par le PM)
Une fusion naïve (combiner `vy_bounce` de S1 ET `bounce_traj` de S2 dans `hit_like`) donne **bounce F1 0.900 (bien !) MAIS confusion_H→B=1** (au lieu de 0) sur le cache. **Les deux protections interagissent mal sur 1 frame** : une frappe fuit en rebond. C'est l'interaction `vy_bounce` × `bounce_traj` × le générateur far-wrist qu'il faut comprendre et corriger.

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-reconcile feat/event-methodo-reconciled
```
Branche `feat/event-methodo-reconciled` **depuis `feat/event-methodo-prod`** (= la base qui contient DÉJÀ S1 mergé). Ainsi tu pars de S1 (0/0 live) et tu greffes S2.
⚠️ `courtside-edit-targets-main-tree` : édite SOUS `/Users/vuong/Documents/CourtSide-CV-reconcile/...`. **Touche UNIQUEMENT `vision/events.py`** (+ le cache de test si tu dois le régénérer, + tests).

## Étapes
1. **Pars de S1** (ta base a déjà `vy_bounce`, le vx-gate, merge_frac 0.18). Confirme : `python tools/event_eval/run_demo3.py` → méthodo confusion 0/0.
2. **Greffe S2 proprement** : récupère ses 3 morceaux SÉPARÉMENT (ils ne sont pas tous en conflit) —
   - `detect_wrist_hits` (+ alias `detect_far_wrist_hits`) : nouvelle fonction, zone propre.
   - les helpers `_is_y_maximum` / `_is_edge_artifact` + le bloc de merge far-wrist dans `classify_events` : zone propre.
   - la **relaxation `bounce_traj`** dans `hit_like` : C'EST LE POINT DE COLLISION avec `vy_bounce`.
   `git show origin/feat/event-methodo-improve:vision/events.py` pour le code S2 exact.
3. **Diagnostique l'interaction** : avec les deux greffes, quelle frame fuit en H→B=1 ? Est-ce le far-wrist qui crée un faux candidat ? la relaxation qui ouvre une frappe ? le vx-gate qui n'attrape pas ce cas ? **Mesure, ne devine pas** (dump la frame fautive, ses features vx_flip/vy/dmin/bounce_traj).
4. **Corrige** pour atteindre **confusion 0/0 ET bounce F1 ≥ 0.900 ET hit F1 ≥ 0.700**. La logique correcte combine les deux protections sans ouvrir de fuite.

## Validation + ITÉRATION
- `python tools/event_eval/run_demo3.py` → **confusion_H→B = 0, confusion_B→H = 0, bounce F1 ≥ 0.900, hit F1 ≥ 0.700** (cache).
- **VALIDE EN LIVE** (la leçon cache-trompeur, deux fois apprise) : `python run_pipeline_8s.py tennis.mp4 -s 73 -d 13 --device cpu --event-methodo --dump-bounces /tmp/r.json` puis `python tools/event_eval/score_stats.py <le _stats.json>` → **confusion 0/0 EN LIVE**, bounce F1 amélioré vs S1 seul (0.667). Le gain cache DOIT reproduire en live.
- **NE régresse PAS** : `test_event_confusion_regression`, `test_wrist_hits_and_relaxation`, `test_bounce_regression` (felix 0.800), `test_bounce_wasb_regression`, `test_vx_veto`, `test_far_coverage` — TOUS verts.
- **Revue adverse** (recommandé pour cette logique firewall délicate) : fais-toi challenger le 0/0 par un sceptique (cas durs : frappe far avec vy_flip, rebond au pied d'un joueur, far-wrist fantôme).
- Itère jusqu'à 0/0 live + F1 0.900, ou documente précisément pourquoi une frame est irréductible.

## CR à la fin (OBLIGATOIRE)
`docs/research/event-methodo-reconciled-CR.md` : la frame qui fuyait + pourquoi, comment tu as combiné vy_bounce + bounce_traj + far-wrist sans fuite, confusion + F1 sur cache ET LIVE, preuve felix intact. Commit + push sur `feat/event-methodo-reconciled`. VRAIS chiffres mesurés, cache ET live.
