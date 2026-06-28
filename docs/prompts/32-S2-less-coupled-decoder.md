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

## ⚠️ INPUT CANONIQUE + baseline POST-S1 (S1 est mergé)
Benchmarks sur **`tennis.mp4 -s 73 -d 13`** (PAS l'extrait `data/output/tennis_demo3.mp4`,
qui dégrade la pose). **Baseline POST-S1 mesurée et vérifiée (ce que tu dois battre) :**
bounce F1 **0.706**, hit F1 ~0.40-0.50, **confusion 1/1**, pool 14/17.
- Le résiduel **confusion_H→B** = une vraie frappe far (vers @271) qui a un vy-flip et
  s'ancre en BOUNCE. Le résiduel **confusion_B→H** = un rebond GT (cold-start, vers @65)
  labellisé HIT par la parité. (PAS @308 — c'était l'extrait dégradé.) Mesure TOI-MÊME les
  frames exactes sur l'input canonique avant de coder.
- Ratés de rebond restants : **174, 308** (trous de génération de candidats, hors-scope DP).
- Spurious restants : HIT@108, BOUNCE@231, HIT@292, HIT@487.

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-s2decoder feat/s2-less-coupled-decoder
```
Branche depuis `feat/accuracy-overhaul` **À JOUR (post-S1+S3 = HEAD 59459b4 ou plus récent)**.
⚠️ édite SOUS `/Users/vuong/Documents/CourtSide-CV-s2decoder/...`. **Touche UNIQUEMENT `vision/events.py`.**

## ⚠️ Les PIÈGES de mesure worktree (3 — ils ont coûté 2 faux verdicts au PM)
1. **Édition** : Edit/Write sur le chemin canonique frappe le MAIN tree. Édite sous ton worktree.
2. **Import** : `tools/event_eval/live_pool.py` résout `ROOT=parents[2]` = MAIN tree → il
   importe `vision/events.py` du MAIN tree, PAS ton worktree. Pour valider TON code :
   soit (a) lance `run_pipeline_8s.py` DEPUIS ton worktree (résout correctement), soit
   (b) force-load par chemin : `importlib.util.spec_from_file_location('vision.events',
   '<worktree>/vision/events.py')` + register dans `sys.modules` AVANT tout import. Vérifie
   `print(E.__file__)` + assert un symbole du fix présent.
3. **Output** : un `run_pipeline_8s.py` lancé depuis le worktree écrit son `_stats.json`
   SOUS LE WORKTREE (`<worktree>/data/output/...`), pas le main tree. Score CE fichier-là
   et VÉRIFIE son mtime == le timestamp "Stats written" du run.

## Le thermomètre
- Replay (compare decoder global vs anchor) sur le dump canonique POST-S1 committé :
  `tests/fixtures/methodo/demo3_methodo_inputs_fullvideo_postS1.json`
  (⚠️ via live_pool depuis ton worktree → applique le PIÈGE 2 ci-dessus pour mesurer TON code).
- LIVE (l'arbitre final) : depuis ton worktree
  `venv/bin/python run_pipeline_8s.py tennis.mp4 -s 73 -d 13 --device mps --match-mode` →
  `score_stats.py <worktree>/data/output/tennis_annotated_stats.json` (PIÈGE 3).
- ⚠️ La piste balle live VARIE d'un run à l'autre (±0.08 F1, non-déterminisme MPS/YOLO
  pré-existant). Compare global vs anchor sur LE MÊME dump (apples-to-apples), pas sur 2
  runs live séparés.

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
