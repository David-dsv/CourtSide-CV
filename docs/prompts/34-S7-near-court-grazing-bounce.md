# PROMPT — S7 : récupérer les rebonds NEAR-COURT rasants (le trou de génération mesuré) — Backend

> **Session AUTONOME ULTRACODE.** Lance en `/ultracode`.
> ⛔ **NE POSE JAMAIS DE QUESTION. Aucun outil de clarification.** Décide, documente, livre.

## Contexte
Repo CourtSide-CV. Lire `CLAUDE.md`, `docs/pm.md`, le CR
`docs/research/event-rootcause-2026-06-28-CR.md`, et les mémoires
`courtside-ball-density-apex-bounce` (le jumeau far : « apex sans extremum-y → detect_sharp_turns
les récupère ; plateau 0/0 connu à 60-75° »), `courtside-hits-overfire-device` (l'état post-S6 +
le mur DP restant), `courtside-s3-event-recall-negative`, `courtside-edit-targets-main-tree`
(les 3 pièges worktree).

**Retour utilisateur (2026-06-30), VÉRIFIÉ par la mesure :** sur le hero render, les rebonds non
comptés sont **TOUS near-court (bas de l'écran)** : **b62 (966,561), b174 (1085,610), b308
(1061,598)** — tous ratés ; les rebonds far (haut) sont quasi tous attrapés. C'est un **trou de
GÉNÉRATION DE CANDIDATS**, pas le tracking ni la classification.

## Le diagnostic PRÉCIS (mesuré — ne le re-suppose pas, exploite-le)
Aux 3 rebonds ratés, **la balle est parfaitement suivie** (dist au GT = 4-6px à l'impact). Mais
aucun générateur ne fire :
- **Pas de vy-flip** : la balle descend vite, rebondit, puis **repart vers le bas LENTEMENT** (vy
  passe de ~+30 à ~+4 mais NE S'INVERSE PAS). C'est un *ralentissement* (un « genou »/knee de la
  série vy), pas une remontée → `detect_turning_points` (extremum-y) ne voit rien.
- **Angle de virage trop doux** : ce sont des rebonds RASANTS. Angles mesurés : b62=**38°**,
  b174=**44°**, b308=**55°** — tous SOUS le seuil `angle_deg=70` de `detect_sharp_turns`
  (`vision/events.py:82`).
- **⚠️ Baisser le seuil d'angle SEUL est un cul-de-sac MESURÉ** : à 45° on ne couvre toujours
  que **1/3** des cibles ET le flood commence (non-GT 3→4). Raison : `detect_sharp_turns` retient
  les PICS LOCAUX proéminents (`find_peaks`, `vision/events.py:131`) ; près du near court la série
  d'angle est étalée, donc le coude rasant n'est pas un pic proéminent même quand l'angle monte.

## Le levier (à TROUVER par la mesure, pas à coder en aveugle)
Un **générateur de candidat near-court dédié** qui capte le « genou » de trajectoire :
- piste A : détecter le **knee de la série vy** (vy descend puis chute brutalement vers ~0 sans
  s'inverser) — c'est la signature physique d'un rebond rasant. Conditionné au near-court (y > net,
  où la balle est grosse/nette) pour ne pas flooder le far.
- piste B : un seuil d'angle **plus bas MAIS local au near-court** + une fenêtre `win_frac`
  adaptée (la fenêtre k=round(fps*0.06)≈3f peut lisser le coude — teste k plus petit côté near).
- piste C : la **courbure** (|v×a|/|v|³) sur la trajectoire near, qui pique au coude même quand
  l'angle absolu reste < 70°.
Mesure laquelle couvre **b62+b174+b308** SANS générer de faux candidats ailleurs.

## ⚠️ INPUT CANONIQUE + les 3 pièges worktree
Benchmarks sur **`tennis.mp4 -s 73 -d 13`** (PAS l'extrait dégradé). Si `tennis.mp4` manque dans
ton worktree : `ln -sf /Users/vuong/Documents/CourtSide-CV/tennis.mp4 .`
Les 3 pièges (mémoire `courtside-edit-targets-main-tree`) : (1) édite SOUS ton worktree ; (2) un
outil/tool du main tree IMPORTE le code du main tree → pour mesurer TON code, soit lance
`run_pipeline_8s.py` depuis ton worktree, soit force-load le module par chemin
(`importlib.util.spec_from_file_location('vision.events','<worktree>/vision/events.py')`) ; (3) la
pipeline lancée du worktree écrit son `_stats.json` SOUS le worktree — score CE fichier (vérifie
le mtime).

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-s7nearbounce feat/s7-near-court-bounce
```
Branche depuis `feat/accuracy-overhaul` (HEAD à jour, post-S6). ⚠️ édite SOUS
`/Users/vuong/Documents/CourtSide-CV-s7nearbounce/...`. **Touche UNIQUEMENT `vision/events.py`**
(les générateurs de candidats). NE touche PAS `vision/bounce.py` ni le tracking.

## Les thermomètres
- Replay rapide (pool recall + F1, byte-identique au _stats.json) sur le dump canonique post-S1 :
  `venv/bin/python tools/event_eval/live_pool.py tests/fixtures/methodo/demo3_methodo_inputs_fullvideo_postS1.json`
  (applique le PIÈGE 2 pour mesurer TON code). Regarde la ligne POOL RECALL : cible **b62, b174,
  b308 passent de NO CANDIDATE à OK**, et la F1 bounce monte.
- LIVE (arbitre final) : depuis ton worktree, `run_pipeline_8s.py tennis.mp4 -s 73 -d 13 --device
  mps --match-mode` → `score_stats.py <worktree>/data/output/tennis_annotated_stats.json`. Lance
  ≥2 runs (la piste varie un peu).

## Contraintes dures
- **NE génère PAS de flood** : le candidat near doit couvrir b62/174/308 sans ajouter de faux
  rebonds. Surveille le **fantôme @231** (il ne doit PAS empirer) et le total de spurious.
- **confusion_H→B = 0 ET firewall intact** (le levier near ne doit pas faire passer une frappe
  near pour un rebond — le firewall `hit_like ⇏ bounce` reste sacré).
- **Planchers** (input canonique) : bounce F1 ≥ 0.750, hit F1 ≥ 0.625 — tu améliores le bounce.
- **felix + tous les tests verts** : `test_event_confusion_regression` (cache 0/0, bF1 0.889),
  `test_bounce_regression`, `test_bounce_wasb_regression`, `test_vx_veto`, `test_sharp_turns`,
  `test_static_fp_reject`, `test_ball_anchored_prox`.
- **Zéro hardcoding** : seuils en fractions de diag/fps/net_y, jamais des pixels d'un clip ; la
  condition « near » = y > net_y (net déjà dérivé à frame_height*0.47), pas une bande tunée.
- **Anti-surapprentissage** : 1 clip GT (3 rebonds cibles). Vérifie qu'un seuil ne fait pas juste
  passer ces 3 frames par coïncidence ; valide la généralité (le mécanisme physique du knee/courbure
  doit tenir, pas un magic number).

## Livrable — CR (OBLIGATOIRE)
`docs/research/s7-near-court-bounce-CR.md` : la couverture pool b62/174/308 AVANT/APRÈS, la F1
bounce LIVE AVANT/APRÈS sur ≥2 runs, preuve 0 flood (fantôme @231 et spurious non aggravés),
confusion 0 + firewall intact, felix + tests verts, le générateur retenu (knee/angle-local/courbure)
+ pourquoi les 2 autres pistes perdent. Commit + push. VRAIS chiffres mesurés LIVE.
