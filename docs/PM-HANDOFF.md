# PM HANDOFF — CourtSide-CV accuracy overhaul (état au 2026-06-28)

> Document de reprise pour la prochaine session PM (la conversation est rafraîchie).
> Branche de travail : **`feat/accuracy-overhaul`** (HEAD `423e4a2`). Lis aussi
> `CLAUDE.md`, `PROJET.md`, `docs/pm.md`, et les mémoires listées en bas.

---

## 0. Le rôle PM (comment travailler)

Tu es **chef de projet**. Tu n'écris pas (ou peu) de code toi-même : tu **découpes**
en chantiers disjoints, tu **écris des prompts `.md`** dans `docs/prompts/`, l'humain
lance les sessions Claude Code (`/ultracode`), te renvoie les **CR**, et tu **vérifies
INDÉPENDAMMENT chaque chiffre EN LIVE** avant de merger. Règles non-négociables :
- **L'humain veut ZÉRO question** (mémoire `courtside-no-questions-autonomy`) — décide,
  énonce tes décisions inline, livre. (Les `AskUserQuestion` ont été utilisés ici pour
  de vraies bifurcations produit ; reste parcimonieux.)
- **Vérifie tout en LIVE, jamais sur un cache** (le cache a menti 4× — voir §4).
- **Sessions = worktree isolé + fichiers disjoints** (mémoire `courtside-parallel-sessions-worktree`).
- **Pousser après chaque merge** (`git push`). Re-checke `git branch --show-current`
  avant chaque commit (arbre partagé — mémoire `courtside-shared-branch-collision`).
- **felix est le garde-fou universel** : toute modif doit garder les tests de régression verts.

---

## 1. L'objectif produit

Pipeline d'analyse vidéo tennis pour **amateur sur trépied** (voir PROJET.md). Le
chantier en cours = **accuracy** : suivi balle/joueurs, détection rebonds/frappes,
homographie, le tout mesuré honnêtement. Point d'entrée unique = `run_pipeline_8s.py`.

---

## 2. CE QUI MARCHE (mergé sur `feat/accuracy-overhaul`, chaque gain vérifié en live)

| Domaine | État | Chiffre vérifié |
|---|---|---|
| **Squelettes (pose)** | ✅ EXCELLENT (l'utilisateur : "parfaits") | — |
| **Sélection joueur far** | ✅ résolu | far-CORRECT **0.9%→100%** sur clip vierge (vs spectateur avant) |
| **Tracking balle (density-adaptive)** | ✅ gros gain | couverture **86%→95.5%**, cold-start **74.5%→90.8%** |
| **ROI balle (idée utilisateur)** | ✅ gardé | **95.5%→98.4%**, CORRECT 93.2% (gate 0.25 anti-poison) |
| **Confusion rebond/frappe** | 🟡 réduite, PAS parfaite | `--event-methodo` ON : legacy 2b/13s → ~7-10/8-10, confusion **danse 0/0↔2/2** (non-déterminisme, §4) |
| **Cut hors-jeu + audio synchro** | ✅ | vidéo plus courte, son re-mux aux spans gardés |
| **Match-mode (P1/P2 stable)** | ✅ | `--match-mode` |
| **Homographie auto (S4)** | ✅ honnête | `--auto-court` (défaut OFF) : tennis→high, felix/grazing→ABSTAIN |
| **Forehand/Backhand (S5)** | ✅ | 2/8→7/7 appelés, confusion FH↔BH **0/0** (abstient plutôt que deviner) |

**Flags par défaut ON** : `--ball-auto`, `--ball-roi`, `--event-methodo`.
**À activer** : `--match-mode`, `--auto-court`.

---

## 3. CE QUI NE MARCHE PAS ENCORE (le retour utilisateur du 2026-06-28)

> Verbatim utilisateur sur la dernière vidéo full-stack : **"overall toujours très
> brouillon : la balle part parfois en cacahuète, il y a des rebonds fantômes, des
> non-détectés, des frappes fantômes et non-détectées. Les squelettes sont parfaits."**

Donc les **2 murs restants** :
1. **TRACKING BALLE encore instable** ("part en cacahuète") — malgré 95.5% couverture
   GT, il y a des **sauts/glitches** visibles. Cause-racine mesurée (S3) : un **blob
   static-FP** (faux positif statique, ex. coin à (90,772)) où la balle "téléporte"
   et reste figée 16 frames → corrompt la trajectoire → faux ancrage d'événement.
2. **ÉVÉNEMENTS bruités** (rebonds/frappes fantômes ET non-détectés) — conséquence
   directe du tracking balle instable + du non-déterminisme (§4).

**Le diagnostic est convergent et SOLIDE (4 sessions) :** ce N'EST PAS la classification
(firewall 0/0 OK), NI la génération de candidats (pool live **17/17**, mesuré par S3).
**C'est la QUALITÉ + le DÉTERMINISME du tracking balle.** Voir §5 pour le plan.

---

## 4. LES PIÈGES APPRIS (ne pas les refaire — ils ont coûté cher)

1. **LE CACHE MENT (4×).** Un cache de détection figé ≠ la piste balle live (qui varie
   selon YOLO/CPU/BLAS). Conséquence : un "0/0 confusion" sur cache donne 1/1 ou 2/2 en
   live, de façon REPRODUCTIBLE par machine. **TOUJOURS valider sur ≥2-3 runs LIVE frais**
   (`run_pipeline_8s.py ... --dump-bounces` + `tools/event_eval/score_stats.py`), jamais
   `run_demo3.py` (cache) seul. Le cache peut mentir dans LES DEUX SENS (S2 : il
   sous-estimait far-pose de 15 pts car bâti sur l'ancien clip annoté).
2. **LE CLIP demo3 ÉTAIT POLLUÉ.** `data/output/tennis_demo3.mp4` était la vidéo
   ANNOTÉE (minimap COURT RADAR → YOLO détectait les points colorés comme des balles).
   **CORRIGÉ** : ré-extrait RAW frame-exact de `tennis.mp4 -s73 -d13` (650 frames, vierge).
   **Avant tout benchmark : confirmer que le clip est la SOURCE BRUTE, pas une sortie pipeline.**
3. **NON-DÉTERMINISME cross-env.** La même branche donne 0/0 dans un env, 2/2 dans un
   autre. Ne jamais committer un `_stats.json` "chanceux" comme preuve. Mesurer la
   distribution sur plusieurs runs.
4. **Anti-triche / anti-hardcode.** Pas de seuil tuné sur 1 clip (surapprentissage sur
   8 frappes / 1 vidéo). Les bons seuils sont ancrés sur une **feature mesurée** (ex.
   S1 : densité de candidats 2.0/2.5/6.9 → seuil 3.5 robuste). Pas de lecture de la GT
   par le code de prod.
5. **Worktree, pas le tree principal.** Éditer sous `/Users/vuong/Documents/CourtSide-CV-<slug>/`
   (mémoire `courtside-edit-targets-main-tree`). J'ai mergé S1 dans le tree principal par
   erreur une fois — annulé proprement.

---

## 5. LE PLAN (prochaines sessions — l'utilisateur a "autant de sessions à dispo")

**Le mur n°1 = tracking balle (qualité + déterminisme).** Cible : tuer les sauts
("cacahuète") et les fantômes. Pistes mesurées/handed-off :

- **(PRIORITÉ 1) Reject static-FP téléporté-puis-figé** (handed off par S3 → zone S1) :
  un blob détecté au MÊME pixel sur N frames consécutives où la balle "saute" dedans =
  faux positif statique → le supprimer À LA SOURCE (avant le Kalman) le corrige pour
  tous les consommateurs (events inclus), et c'est DÉTERMINISTE donc validable proprement.
  Fichiers : `models/yolo_detector.py` (static-FP filter) / `run_pipeline_8s.py` Pass 1.
- **(PRIORITÉ 2) Décodeur d'événements moins couplé** : le DP d'alternance global fait
  cascader une erreur (S3). Un décodeur local/par-rallye limiterait les fantômes en chaîne.
  Fichier : `vision/events.py`. ⚠️ mesurer 0/0 sur ≥2 runs LIVE.
- **(PRIORITÉ 3) Pilotes tracker SOTA licence-clean** (web, évalués par S1) : RacketVision
  MS-TrackNetV3 (MIT, poids HF) + TOTNet (occlusion). À tester MPS + scorer vs 1920-Kalman.
- **(autre) Homographie** : S4 court-level "never positively fits" → DeepLSD/M-LSD
  (licence-clean) comme next lever pour densifier les lignes.

**Découpage suggéré (disjoint) pour N sessions :**
| Session | Chantier | Fichiers | Métrique |
|---|---|---|---|
| A | static-FP reject (déterminisme balle) | `models/yolo_detector.py` + Pass 1 | `measure_ball_coverage.py` + visuel "cacahuète" |
| B | décodeur events moins couplé | `vision/events.py` | `score_stats.py` LIVE ≥2 runs, confusion 0/0 |
| C | pilote MS-TrackNetV3 / TOTNet | nouveau `models/<tracker>.py` | `measure_ball_coverage.py` vs 1920-Kalman |
| D | homographie M-LSD/DeepLSD | `models/court_*.py` | `tools/court_eval/eval_homography.py` |
| E | R&D libre (web SOTA) selon le mur restant | — | — |

---

## 6. LES THERMOMÈTRES (GT humaines + scorers — utiliser, JAMAIS de cache seul)

- **GT balle** : `tests/fixtures/ball_gt/tennis_demo3.ball_gt.json` (311 pts) → scorer
  `tools/ball_gt/measure_ball_coverage.py` (coverage + CORRECT, overall + cold-start).
  Replay rapide : `tools/ball_gt/replay_strat.py` / `roi_lab.py` (caches committés).
- **GT pose far** : `tests/fixtures/pose_gt/tennis_demo3.pose_gt.json` (225 pts) →
  `tools/pose_gt/measure_far_coverage.py` (far-CORRECT, baseline 100% sur clip vierge).
- **GT events** : `tests/fixtures/bounces/tennis_demo3.bounces.json` (9) +
  `shots/tennis_demo3.shots.json` (8) → `tools/event_eval/score_stats.py` (LIVE) /
  `run_demo3.py` (cache, MENT) / `live_pool.py` (pool recall live).
- **GT FH/BH** : labels `type` dans le shots.json → `tools/event_eval/score_fhb.py`.
- **Annotateurs** (pour étendre les GT) : `tools/ball_gt/annotate_server.py` (port 8012),
  `tools/pose_gt/annotate_server.py` (port 8011) — clic-point, prev/next ghost.

**Tests de régression (garde-fous, doivent rester verts) :** `test_far_coverage`,
`test_roi_redetect`, `test_ball_roi_regression`, `test_ball_conf_regression`,
`test_court_regression`, `test_fhb_regression`, `test_event_confusion_regression`,
`test_bounce_regression` (felix Kalman ≥0.72), `test_bounce_wasb_regression` (≥0.80),
`test_vx_veto`, `test_sharp_turns`. (11 clés + les anciens.)

---

## 7. ÉTAT GIT / MÉNAGE

- HEAD `feat/accuracy-overhaul` @ `423e4a2`, poussé.
- **16 worktrees secondaires** traînent (`git worktree list`) — des sessions passées,
  la plupart mergées. À nettoyer (`git worktree remove --force ../CourtSide-CV-<slug>`)
  une fois confirmé que les fenêtres Claude Code correspondantes sont fermées. NE PAS
  supprimer un worktree d'une session encore ouverte.
- Branches distantes des sessions mergées (`feat/s1-*`, `feat/s4-*`, etc.) : suppression
  = action destructive → demander l'autorisation de l'humain.

---

## 8. MÉMOIRES À LIRE (le contexte profond)

`courtside-no-questions-autonomy`, `courtside-parallel-sessions-worktree`,
`courtside-edit-targets-main-tree`, `courtside-events-plateau-ball-tracking-wall`,
`courtside-s3-event-recall-negative`, `courtside-ball-track-conf-not-detection`,
`courtside-far-player-selection-not-detection`, `courtside-event-methodo-prod-gap`,
`courtside-bounce-broadcast-multifactor`, `courtside-shot-guard`, `courtside-match-mode`.

---

## 9. TL;DR pour la reprise

Les **squelettes sont parfaits**, le **far-player est résolu** (100%), la **balle est
bien plus dense** (98%) mais **part encore "en cacahuète"** par moments → ça crée des
**rebonds/frappes fantômes + non-détectés**. Le mur est le **tracking balle (qualité +
déterminisme)**, PAS la classification ni la génération (mesuré). Le fix net = **rejeter
le faux-positif statique téléporté à la source** (déterministe, corrige tous les
consommateurs). Lance des sessions sur §5, mesure TOUT en LIVE sur les GT du §6, garde
felix vert, zéro question / zéro hardcode / zéro triche.
