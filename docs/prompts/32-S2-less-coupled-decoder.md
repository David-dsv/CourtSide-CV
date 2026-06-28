# PROMPT — S2 : décodeur moins couplé + le B→H résiduel @308 (OPT-IN, après S1) — Backend

> **Session AUTONOME ULTRACODE.** Lance en `/ultracode`.
> ⛔ **NE POSE JAMAIS DE QUESTION. Aucun outil de clarification.** Décide, documente, livre.

## ⚠️ ORDONNANCEMENT — démarre APRÈS le merge de S1
Cette session touche **le MÊME fichier que S1** (`vision/events.py`). Elle NE PEUT PAS
tourner en parallèle de S1. Attends que `feat/s1-ball-anchored-hit-prox` soit mergé sur
`feat/accuracy-overhaul`, puis branche depuis le HEAD à jour. (S3 lui est disjoint et peut
tourner en parallèle de S1.)

## Contexte
Repo CourtSide-CV. Lire `CLAUDE.md`, `docs/pm.md`, le CR
`docs/research/event-rootcause-2026-06-28-CR.md`, les mémoires
`courtside-events-plateau-ball-tracking-wall`, `courtside-s3-event-recall-negative`,
`courtside-hits-overfire-device`.

**Le bug.** Le DP de parité global (`_global_alternation_decode`, `vision/events.py:496`)
est fragile : il « parity-fill » le rebond GT 308 en HIT@307 (aucune ancre) et propage les
ancres fantômes. Un décodeur alternatif segment-local **existe déjà**
(`_anchor_parity_decode`, `events.py:574`) mais `classify_events` code en dur
`decoder="global"` (`events.py:443`).

## ⚠️ Le BLOCAGE mesuré (lis avant de coder)
Flipper le défaut sur `decoder="anchor"` RÉGRESSE le cache committé : bounce F1
0.889→0.714 et hit F1 0.632→0.308 → **fait échouer `test_event_confusion_regression`**. Le
désaccord cross-track (live MPS +F1, cache Kalman −F1) est lui-même un signal de
non-généralisation. **Donc : NE FLIPPE PAS le défaut.**

**Scope de cette session :**
1. Tourne APRÈS S1 (S1 enlève les ancres fantômes ; sans ça le swap de décodeur seul ne
   donne que hit F1 0.250 car les fantômes verrouillent encore le placement).
2. Soit expose `decoder="anchor"` en **opt-in** (flag, défaut = global), soit fais un fix
   CIBLÉ du placement @308 dans le DP global (sans casser le reste).
3. Le flip-par-défaut reste INTERDIT tant qu'un dump LIVE felix + un cache re-baseliné ne
   sont pas d'accord.

## ⚠️ INPUT CANONIQUE
Benchmarks sur **`tennis.mp4 -s 73 -d 13`** (PAS l'extrait). Le résiduel à viser est
**confusion_B→H @308** (rebond GT labellisé HIT) que S1 laisse en place.

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-s2decoder feat/s2-less-coupled-decoder
```
Branche depuis `feat/accuracy-overhaul` **À JOUR (post-S1)**. ⚠️ édite SOUS
`/Users/vuong/Documents/CourtSide-CV-s2decoder/...`. **Touche UNIQUEMENT `vision/events.py`.**

## Le thermomètre
- Replay (compare decoder global vs anchor) :
  `venv/bin/python tools/event_eval/live_pool.py tests/fixtures/methodo/demo3_methodo_inputs_fullvideo.json`
  (⚠️ ce dump a été produit AVANT S1 ; après S1 mergé, RE-GÉNÈRE le dump canonique :
  `COURTSIDE_DUMP_METHODO_INPUTS=/tmp/s2/inputs.json venv/bin/python run_pipeline_8s.py tennis.mp4 -s 73 -d 13 --device mps --match-mode`)
- LIVE : `run_pipeline_8s.py tennis.mp4 -s 73 -d 13 --device mps --match-mode` →
  `score_stats.py`.

## Contraintes dures (le gate MANDATORY)
- `tests/test_event_confusion_regression.py` DOIT rester VERT (cache 0/0, bounce F1 ≥ 0.80,
  hit F1 ≥ 0.60). C'est le bloqueur du flip-défaut.
- confusion_H→B = 0 maintenu.
- felix intact (`test_bounce_regression`, `test_bounce_wasb_regression`).
- Zéro hardcoding ; pas de seuil tuné sur 1 clip.

## Livrable — CR (OBLIGATOIRE)
`docs/research/s2-less-coupled-decoder-CR.md` : chiffres LIVE full-video global vs anchor,
preuve que le cache reste vert (ou résultat NÉGATIF honnête si le décodeur ne généralise
pas), ce qui récupère @308, recommandation finale (opt-in vs fix ciblé vs ne-pas-ship).
Commit + push.
