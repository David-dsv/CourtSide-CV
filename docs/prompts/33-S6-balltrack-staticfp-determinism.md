# PROMPT — S6 : déterminisme du tracking balle — rejet du static-FP « téléporté-puis-figé » — Backend

> **Session AUTONOME ULTRACODE.** Lance en `/ultracode`.
> ⛔ **NE POSE JAMAIS DE QUESTION. Aucun outil de clarification.** Décide, documente, livre.

## Contexte
Repo CourtSide-CV. Lire `CLAUDE.md`, `docs/pm.md`, le CR
`docs/research/event-rootcause-2026-06-28-CR.md`, et les mémoires
`courtside-hits-overfire-device` (le « now-dominant wall »), `courtside-s3-event-recall-negative`
(root-cause-A, le blob qui devient une ancre mandatory), `courtside-ball-track-conf-not-detection`,
`courtside-ball-density-adaptive-imgsz`, `courtside-edit-targets-main-tree` (les 3 pièges worktree).

**Le mur (MESURÉ, pas supposé).** Le classifieur d'événements est maintenant solide (S1+S2+S3
mergés : bounce F1 0.75 / hit F1 0.625, confusion H→B tuée sur un track propre). MAIS la **piste
balle live VARIE d'un run à l'autre** sur le MÊME clip+code : couverture 90%↔84%, et 1↔2 **blobs
static-FP « téléportés-puis-figés »** (un faux positif statique, ex. logo/coin/marque, dans lequel
la balle « saute » à >0.15·diag puis reste figée ≥8 frames). Ça suffit à faire osciller le F1
événementiel de **0.75 à 0.40** entre deux renders → c'est ça les « rebonds fantômes qui vont et
viennent » que voit l'utilisateur. Un blob figé lu par `_direction_features` fabrique un `vy_flip`
→ une **ancre BOUNCE MANDATORY** (events.py ~519-525, non-skippable) qui inonde le DP d'alternance.

## Le bug a DEUX moitiés (corrige les deux)
Dans `run_pipeline_8s.py` Pass 1 (le filtre static-FP + le tracking Kalman) :
1. **Détection trop laxiste** : `static_threshold = max(10, int(max_frames*0.08))` (~52 frames sur
   650) → un blob figé 16 frames PASSE SOUS le seuil et n'est jamais marqué static
   (`run_pipeline_8s.py:1140`).
2. **Acceptation trop laxiste** : à la ré-init (coast expiré), `reinit_gate = frame_diag * 0.50`
   (~1100px) ACCEPTE une détection à ~1097px de la dernière position connue
   (`run_pipeline_8s.py:1246-1253`) → la balle « téléporte » dans le blob et s'y verrouille.

## Le fix (le discriminateur propre, déjà identifié par S3)
Rejeter À LA SOURCE (avant le Kalman, donc DÉTERMINISTE et corrige TOUS les consommateurs —
events, bounce, speed, minimap) un cluster qui est **(a) figé** (≥N frames dans un rayon ~25px)
ET **(b) entré par un saut de >~0.15·diag depuis la dernière position balle valide** (la signature
« teleport-then-frozen »). Le saut d'entrée du blob (0.5·diag) vs un vrai rebond figé (0.02·diag)
sépare proprement (mesuré S3 : 26× d'écart).

## ⚠️ LE PIÈGE À NE PAS REFAIRE (mémoire + S3)
Un VRAI rebond / apex far est AUSSI quasi-figé (la balle ralentit au sommet de l'arc) — un rejet
« frozen-plateau » SEUL DÉTRUIT le vrai tracking (GT confirme la balle figée à ±5px aux apex).
**Le rejet DOIT exiger la CONJONCTION téléport-entrée + figé + (idéalement) hors-aire-de-jeu** ;
jamais « figé » seul. Valide contre la GT balle que tu ne supprimes AUCUN vrai point.

## ⚠️ INPUT CANONIQUE + les 3 pièges worktree
Benchmarks sur **`tennis.mp4 -s 73 -d 13`** (PAS l'extrait dégradé `data/output/tennis_demo3.mp4`).
Si `tennis.mp4` manque dans ton worktree : `ln -sf /Users/vuong/Documents/CourtSide-CV/tennis.mp4 .`
Les 3 pièges (voir mémoire `courtside-edit-targets-main-tree`) : (1) édite SOUS ton worktree ;
(2) un outil du main tree importe le code du main tree — pour mesurer TON code lance
`run_pipeline_8s.py` depuis ton worktree ; (3) il écrit son `_stats.json` SOUS ton worktree —
score CE fichier (vérifie le mtime == le « Stats written » du run).

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-s6staticfp feat/s6-balltrack-staticfp
```
Branche depuis `feat/accuracy-overhaul` (HEAD 2fabcd4 ou +). ⚠️ édite SOUS
`/Users/vuong/Documents/CourtSide-CV-s6staticfp/...`. **Touche `run_pipeline_8s.py` Pass 1 et/ou
`models/yolo_detector.py`** (le filtre static-FP). NE touche PAS `vision/events.py` ni
`vision/bounce.py` (figés — c'est l'amont qu'on corrige).

## Les thermomètres (DÉTERMINISME = la métrique clé)
- **Couverture/CORRECT balle** (le garde-fou anti-sur-rejet) :
  `venv/bin/python tools/ball_gt/measure_ball_coverage.py` (GT 350 pts ;
  baseline ~95.5% cov / 92.6% CORRECT — tu NE dois PAS les baisser).
- **Déterminisme** : lance la pipeline LIVE **≥3 fois** (`run_pipeline_8s.py tennis.mp4 -s 73 -d 13
  --device mps --match-mode`), dump `--dump-ball-track` à chaque fois, et mesure (1) la variance
  de couverture et (2) le nombre de blobs « teleport-then-frozen » par run. Cible : **0 blob**,
  variance réduite. Compte les blobs avec la signature du §"DEUX moitiés" (jump>0.15·diag → cluster
  ≥8f dans 25px).
- **F1 événementiel** (le bénéfice aval) : `score_stats.py <worktree>/data/output/..._stats.json`
  sur ≥3 runs → la distribution doit se RESSERRER vers le haut (moins de runs à 0.40).

## Contraintes dures
- **Couverture balle ≥ ~95% et CORRECT ≥ ~92%** (NE régresse pas le tracking en sur-rejetant).
- **felix intact** : `test_bounce_regression` (≥0.72), `test_bounce_wasb_regression` (≥0.80),
  `test_vx_veto`, `test_event_confusion_regression` (0/0, bF1 0.889), `test_ball_conf_regression`,
  `test_ball_roi_regression`, `test_roi_redetect`, `test_far_coverage` — TOUS verts.
- **Zéro hardcoding** : seuils en fractions de diag/fps/max_frames, jamais des pixels d'un clip.
- **Anti-triche** : pas de seuil tuné pour faire passer 1 run ; valide la distribution sur ≥3 runs.

## Livrable — CR (OBLIGATOIRE)
`docs/research/s6-balltrack-staticfp-CR.md` : la distribution couverture+F1 sur ≥3 runs AVANT/APRÈS,
preuve que 0 vrai point balle GT n'est supprimé, le discriminateur retenu (téléport+figé+aire),
felix + tests verts, et un test de régression déterministe si possible. Commit + push à chaque
itération. VRAIS chiffres mesurés LIVE (jamais un cache seul).
