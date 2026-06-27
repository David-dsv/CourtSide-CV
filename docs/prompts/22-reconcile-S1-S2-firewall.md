# PROMPT — Recall + fantômes + fusion S1+S2 dans events.py (Backend) — SESSION RÉCONCILIATION

> **PRIORITÉ : récupérer les rebonds/frappes ratés (surtout au début) et tuer les
> 3 fantômes — c'est ce que l'utilisateur voit. La fusion S1+S2 et le 0/0 robuste
> viennent ensuite.**

> **Session AUTONOME ULTRACODE.** Lance en `/ultracode`.
> ⛔ **NE POSE JAMAIS DE QUESTION à l'utilisateur. Aucun outil de clarification.**
> Décide seule, documente, ne bloque jamais. Une question = run raté.

## ⭐ CE QUE L'UTILISATEUR VOIT (le vrai objectif, observé sur la vidéo régénérée)
La confusion rebond↔frappe est RÉSOLUE (0/0). MAIS l'utilisateur constate, à l'œil sur `data/output/tennis_demo3_methodo_annotated.mp4` : **« des rebonds et tirs pas détectés au début, et un rebond + un tir fantôme »**. Mesuré sur ce run live (méthodo ON) vs la GT :
- **Rebonds RATÉS (4/9)** : f62 (1.2s), f120 (2.4s), **f174** (3.5s), f308 (6.2s) — **3/4 au DÉBUT du clip**.
- **Frappes RATÉES (2/8)** : f81 (1.6s, début), f468 (9.4s).
- **FANTÔMES (3)** : 1 rebond f229 (4.6s), 2 frappes f444/f487 (~9s).
→ **Le problème n'est plus le firewall (0/0 tient), c'est le RECALL (surtout au début) + les FAUX POSITIFS.** C'est PRIORITAIRE pour le ressenti utilisateur.

**Hypothèse à vérifier (début du clip)** : le tracking balle (Kalman/WASB) n'a pas d'historique sur les premières frames → les premiers events n'ont pas de candidats. Mesure-le (combien des ratés du début sont absents du POOL de candidats vs mal classés ?). Si c'est la génération, c'est `detect_turning_points`/`detect_sharp_turns`/`detect_wrist_hits` qu'il faut renforcer en début de séquence, pas le firewall.

**Les fantômes** : f229 (rebond) + f444/f487 (frappes) n'ont pas de GT proche → sur-génération (sharp-turns trop sensible ? far-wrist fantôme ?). À supprimer SANS tuer les vrais events.

## Contexte — le problème de fusion (secondaire mais à régler aussi)
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

## Cibles (dans l'ordre de PRIORITÉ utilisateur)
1. **RECALL ↑ (le plus visible)** : récupérer les rebonds ratés f62/f120/f174/f308 (surtout le début) et les frappes f81/f468. Bounce recall 5/9 → viser ≥ 8/9, hit recall 6/8 → viser ≥ 7/8.
2. **FANTÔMES → 0** : supprimer f229 (rebond) + f444/f487 (frappes) sans tuer de vrais events.
3. **Confusion 0/0 ROBUSTE** : le 0/0 actuel est run-dépendant (un run de vérif PM donnait 1/1). Le rendre structurel (vx-gate + alternation), vérifié sur ≥2 runs live.
4. **F1** : bounce F1 ≥ 0.842 (cache), idéalement 0.900 via la relaxation S2 (f174).

## Validation + ITÉRATION
- `python tools/event_eval/run_demo3.py` → **confusion 0/0, bounce F1 ≥ 0.842** (cache).
- **VALIDE EN LIVE ≥2 FOIS** (la détection balle est NON-DÉTERMINISTE — un 0/0 sur 1 run ne suffit pas ; le CR S1 l'a appris à ses dépens) : `python run_pipeline_8s.py tennis.mp4 -s 73 -d 13 --device cpu --dump-bounces /tmp/r.json` (méthodo ON par défaut) puis `python tools/event_eval/score_stats.py <_stats.json>`. Rapporte le recall + confusion sur CHAQUE run. **Le recall doit monter ET la confusion rester 0/0 sur les deux runs.**
- Diagnostique d'abord les ratés du DÉBUT : sont-ils absents du pool de candidats (génération) ou mal classés (firewall) ? Corrige la vraie cause.
- **NE régresse PAS** : `test_event_confusion_regression`, `test_wrist_hits_and_relaxation`, `test_bounce_regression` (felix 0.800), `test_bounce_wasb_regression`, `test_vx_veto`, `test_far_coverage` — TOUS verts.
- **Revue adverse** (recommandé pour cette logique firewall délicate) : fais-toi challenger le 0/0 par un sceptique (cas durs : frappe far avec vy_flip, rebond au pied d'un joueur, far-wrist fantôme).
- Itère jusqu'à 0/0 live + F1 0.900, ou documente précisément pourquoi une frame est irréductible.

## CR à la fin (OBLIGATOIRE)
`docs/research/event-methodo-reconciled-CR.md` : la frame qui fuyait + pourquoi, comment tu as combiné vy_bounce + bounce_traj + far-wrist sans fuite, confusion + F1 sur cache ET LIVE, preuve felix intact. Commit + push sur `feat/event-methodo-reconciled`. VRAIS chiffres mesurés, cache ET live.
