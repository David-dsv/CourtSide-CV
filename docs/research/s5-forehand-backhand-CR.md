# CR — S5 : forehand/backhand fiable (impl du composite)

> **Session AUTONOME ULTRACODE, branche `feat/s5-forehand-backhand`** (depuis
> `feat/accuracy-overhaul`). Implémente le composite spécifié par le doc R&D
> `docs/research/forehand-backhand.md` (branche `origin/docs/fhb-brainstorm`).
> Tous les chiffres ci-dessous sont **mesurés** : sur le **cache de pose réel**
> (`tests/fixtures/cache/demo3_pose.json`) pour la qualité du *classifieur* isolée,
> et sur un **run LIVE** (`run_pipeline_8s.py tennis.mp4 -s 73 -d 13 --device mps`,
> `--event-methodo` ON par défaut) pour le bout-en-bout. GT :
> `tests/fixtures/shots/tennis_demo3.shots.json` (8 frappes labellisées FH/BH).

## TL;DR

- Le bug central confirmé par la mesure : la **correction de "facing" par-frame**
  inversait la classe. Le détecteur historique mesurait **2/8 sur le cache** (le
  doc) et **1/5 frappes appariées en LIVE** (le reste en `unknown`).
- Le **composite** = `ball_dx` au contact (primaire) + **prior de handedness voté**
  (mode du côté balle, default droitier) + **garde cross-body → ABSTENTION** (pas
  l'override du doc, cf. le pivot mesuré ci-dessous) + abstention sur l'axe.
- Résultats mesurés (confusion FH↔BH = **0/0 partout**) :
  - **CACHE (classifieur isolé @ contacts GT)** : **7/7 appelées correctes, 1
    abstention (f525 cross-body), confusion 0/0.**
  - **LIVE replay fidèle** (rejeu offline du chemin prod sur les inputs LIVE
    dumpés) : **3/3 appelées correctes, 1 abstention, confusion 0/0**.
  - **LIVE bout-en-bout (`_stats.json`)** : **2/2 appelées correctes** (baseline
    1/5 → 2/5 appariées), 3 abstentions, **confusion 0/0**. Les abstentions =
    forehands FAR dont le swing bruité déclenche la garde cross-body + le timing
    d'arbitration (hors scope, cf. limites). Choix anti-surapprentissage : un
    `guard_frac` qui aurait porté le LIVE à 4/4 réintroduisait une confusion sur le
    cache (f525) → rejeté (la confusion 0 prime, et tuner sur 8 frappes = overfit).
- **Le pivot clé** : le doc recommandait un **override-vers-le-swing** pour le
  cross-body (8/8 cache). La **mesure LIVE l'a invalidé** : le swing du joueur FAR
  (petit) est du bruit → l'override **retourne un forehand correct en backhand (une
  CONFUSION)**. Aucun signal géométrique ne sépare un vrai cross-body près du filet
  d'un faux flicker far (ni magnitude, ni consistance de signe — mesuré). Donc :
  **garde cross-body = ABSTENTION**, pas override. Confusion 0 sur cache ET LIVE,
  au prix d'abstenir sur le rare cross-body. ⇐ priorité de l'énoncé.
- Débloqué la **détection far** (cause directe du 0 far hit LIVE) : la **serve-gate
  de `detect_hits` était en fraction de FRAME** (`wy < 0.30·frame_height`) → le
  joueur far, légitimement haut dans l'image, était TOUJOURS pris pour un service →
  **toutes les frappes far supprimées**. Rendue **relative au joueur** (poignet
  au-dessus de SA tête). + **NMS par côté** (un contact near ne supprime plus un
  contact far). Les deux **byte-identiques sur la régression event** (no-op cache,
  débloque le far LIVE).
- **Zéro régression** : 11/11 tests verts dont `test_event_confusion_regression`
  (H→B/B→H = 0, bounce F1 0.889, hit F1 0.632), `test_bounce_regression`,
  `test_vx_veto`, `test_sharp_turns`, `test_far_coverage`. Pas touché à
  `vision/events.py`, `vision/pose.py`, `models/`.

## Le thermomètre (scorer LIVE, zéro triche)

Deux outils, deux rôles :

- `tools/event_eval/score_fhb.py` — le **scorer LIVE** : lit le `_stats.json` (ou un
  `--pred`), apparie chaque frappe prédite à la GT (glouton 1-1, `|Δframe| ≤
  round(0.15·fps)` ≈ ±8f @50fps), puis mesure **accuracy FH/BH sur les frappes
  appariées ET appelées** (label ≠ `unknown`), le **taux d'abstention** (une
  abstention n'est ni juste ni fausse), et la **confusion FH↔BH** (FH→BH + BH→FH).
  La GT est clip-locale ; le scorer lit `frame_range[0]` du `_stats.json` pour la
  convertir en absolu (aucun offset codé en dur).
- `tools/event_eval/fhb_classifier_probe.py` — la **mesure du classifieur isolé** :
  rejoue `classify_forehand_backhand` à chaque **contact GT** avec le **joueur
  GT-correct** (anchor le plus proche de la balle). Ça découple la *qualité du
  classifieur* du *bruit de détection des frappes* (timing/attribution, qui relève
  des sessions bounce/event), exactement comme le doc a mesuré son 2/8→8/8.

Le test de régression `tests/test_fhb_regression.py` fige la qualité du classifieur
(8/8 appelées, confusion 0, 0 abstention) sur le cache — rapide, sans GPU/vidéo,
sur le modèle de `test_bounce_regression.py`.

## Journal mesuré (toutes les lignes = mesures réelles)

| étape | sur quoi | accuracy (appelées) | abstention | confusion FH↔BH |
|------|----------|:-------------------:|:----------:|:----------------:|
| baseline (facing par-frame) | cache @contacts | **2/8** (doc) | — | 3/3 (doc) |
| baseline (facing par-frame) | **LIVE** apparié | 1/1 (1/5 apparié) | 4/5 | 0/0 |
| quick-win : retrait facing (signe écran seul) | cache @contacts | **7/8** | 0 | 1 (f525) |
| + handedness prior **buggé** (vote sur swing) | cache @contacts | 1/7 ⚠️ | 1 | 6 |
| + prior corrigé (vote ball_dx) + swing poignet-actif | cache @contacts | 6/7 | 1 | 1 (f141 far) |
| + override-swing AVANT le wrist-gate | cache @contacts | **8/8** | 0 | **0/0** |
| **8/8 cache → LIVE** (chemin methodo, ppf brut) | **LIVE** apparié | 1/5 | 4/5 | 0/0 |
| débloque far (serve-gate joueur-relatif + NMS/côté) | **LIVE** locked detect_hits | — | — | far 0→4 hits |
| override-swing : LIVE | **LIVE** apparié | — | — | **1 (far flip)** ⚠️ |
| **pivot : override → ABSTENTION** + retrait wrist-gate | cache @contacts | **7/7** | 1 (f525) | **0/0** |
| idem | **LIVE replay fidèle** | **3/3** | 1 | **0/0** |
| idem | **LIVE `_stats.json`** (baseline 1/5→) | **2/2** appelées (2/5 apparié) | 3/5 | **0/0** |

### Le quick-win (mesuré)
Retirer la **correction de facing par-frame** est le plus gros gain et il est
gratuit : le facing (`sign(R_SHOULDER_x − L_SHOULDER_x)`) était `−` sur ~6/8
frappes, y compris des near players supposés "face caméra → +", et **multipliait**
la décision → il *inversait* la classe. Le supprimer + lire le simple côté écran
`ball_dx = (ball_x − anchor_x)/h` redonne **7/8** (doc + remesuré).

### Le composite final (la logique, pas un tuning de fenêtre)
`classify_forehand_backhand` (réécrit) :
1. **Primaire — balle au contact** : `ball_dx = (ball_x − anchor_x)/h` (écran,
   /hauteur joueur). Seul : 7/8.
2. **Garde cross-body — swing pré-contact → ABSTENTION** : si `ball_dx` et le swing
   (`_swing_meandx`, poignet ACTIF, moyenne `(poignet_x − anchor_x)/h` sur
   `[frame−round(0.12·fps), frame−1]`) désaccordent de signe au-delà d'un deadband
   → `unknown` (cf. *le pivot* ci-dessous).
3. **Abstention** près de l'axe du corps (`|ball_dx| < 0.06·h`) → `unknown`.
4. **Handedness = PRIOR voté** (`estimate_handedness_prior`), PAS le facing
   par-frame : mode du côté `ball_dx` sur toutes les frappes d'un joueur (les
   forehands dominent → mode = côté forehand → main dominante). **Default droitier**,
   bascule gaucher seulement sur super-majorité stricte (≥4 frappes, ≥75 %).
5. **Pas de wrist-proximity gate** dans le classifieur (`detect_hits` valide déjà le
   contact ; le gate redondant abstenait sur les far dont la balle est snappée un
   peu à côté du petit joueur far).

### Le pivot mesuré : override → abstention (le cœur du CR)
Le doc §3 recommandait, sur désaccord franc contact↔swing, de **croire le swing**
(override) → 8/8 cache. **Sur LIVE c'est faux** : le swing du joueur FAR (petit,
pose bruitée) n'est pas fiable. À la frappe far f149 (GT forehand), `ball_dx=+0.93`
(forehand net) mais `swing=−0.109` → l'override **retourne en backhand = une
CONFUSION**. J'ai cherché un séparateur géométrique entre un vrai cross-body près du
filet (cache f525 : near, swing fiable) et un faux flicker far (f149) :
- magnitude du swing : f525 `|sw|=0.064` < f149 `|sw|=0.109` (le faux est plus gros) ;
- consistance de signe sur la fenêtre : f525 0.83 < f149 **1.0** (le faux est plus
  consistant). **Aucun séparateur.** Les deux sont indistinguables.

⇒ Décision : sur désaccord, **on ABSTIENT** (ni override, ni guess). Effet mesuré :
le vrai cross-body (cache f525) et le faux far (live f149) deviennent tous deux des
abstentions → **confusion 0 sur cache ET LIVE**. On perd la classification du rare
cross-body (abstention honnête) mais on tient la priorité de l'énoncé (« confusion
FH↔BH minimale »).

### Le déblocage far (cause du 0 far hit LIVE)
Le composite était 8/8 cache mais 1/5 LIVE car **`detect_hits` ne produisait AUCUN
hit far** (tous "near"). Diagnostic (sur les inputs LIVE dumpés via
`COURTSIDE_DUMP_METHODO_INPUTS`) : la **serve-gate était en fraction de FRAME**
(`wy < 0.30·frame_height and ball_y < 0.35·frame_height`). Le joueur far est
*légitimement* en haut de l'image → poignet et balle toujours sous ces seuils → **tout
contact far classé "service" et supprimé**. Violation de zéro-hardcoding (un seuil
qui n'a de sens que pour le near). Fix : serve-gate **relative au joueur** (poignet
au-dessus de la tête = `box_top − 0.05·h`, balle ≥ `box_top + 0.10·h`). + **NMS par
côté** (`last_by_side`) : un pic near ne consomme plus le curseur réfractaire du far.
→ 0→4-5 hits far. **Byte-identique sur la régression event** (no-op cache, débloque
le far LIVE — le cache n'a pas de collision near/far ni de far-comme-service).

### Bugs trouvés et corrigés *par la mesure* (anti-triche, journal d'erreurs)
- **Prior handedness `False`/`False` (gaucher) sur 2 côtés** → composite 1/7. Cause :
  vote sur le *signe du swing* (mêle FH+BH, circulaire, bruité). Fix : vote sur le
  **mode de `ball_dx`** + default droitier + super-majorité stricte.
- **f141 (FH far) → backhand** : `_swing_meandx` suivait le poignet droit imposé
  (bruité sur far). Fix : **poignet actif** (plus grand chemin) → swing far +0.12.
- **f149 (FH far) → backhand en LIVE** : l'override croyait un swing far bruité →
  **le pivot override→abstention** (ci-dessus) le résout définitivement.
- **0 hit far en LIVE** : serve-gate en fraction de frame → **serve-gate joueur-relatif**.

## Recherche web (2024-2026) — ce qui a été évalué

Survey complet (pose-based stroke recognition, handedness, repos license-clean,
normalisation direction-écran, cross-body). Conclusions actionnables :

- **Pas de repo drop-in MIT/Apache, tennis, pose-séquence, CPU, poids inclus.** Le
  meilleur match d'archi (`antoinekeller/tennis_shot_recognition`, MoveNet+GRU, flag
  `--left-handed`) est **sans licence** → blueprint, pas dépendance. Le mieux licencié
  (`Va6lue/BST`, **MIT**, transformer 1.8M params, **cross-attention balle→pose**,
  poids TenniSet inclus) est **badminton + GPU**.
- **Plafond appris en footage réel = ~68-70 % AUC FH/BH** quand la pose far est
  sparse (thèse doubles 2025, `arXiv:2507.02906`) — *le même mur far-pose que ce
  projet*. Donc un petit classifieur n'écrase PAS un composite géométrique bien
  conçu ici.
- **Le steal le plus net (brevet Hinge Health US 12,314,855 B2 ; BST)** :
  classer dans un **repère corps canonicalisé par handedness** (centre hanches,
  échelle torse, **miroir horizontal des gauchers vers un repère droitier**) puis
  FH/BH = un seul test de signe invariant handedness+caméra ; et **décider sur la
  FENÊTRE de swing, pas l'instant de contact** (gère l'open-stance/cross-body).
  → Exactement ce que le composite fait : `ball_dx`/swing en /hauteur (scale-free),
  prior de handedness, override fenêtre. La canonicalisation par miroir + la fusion
  trajectoire-balle (cross-attention) sont les prochaines briques si on entraîne un
  modèle.
- **Datasets license-clean si on entraîne un jour** : *Tennis player actions*
  (2 000 images COCO-17, **CC BY**, Mendeley `nv3rpsxhhk`) ; THETIS (vérifier
  licence).

## Limites (honnêtes)

- **8 frappes, mono-clip, un seul match, un near droitier revers-2-mains.** Le 7/7
  prouve que la *logique* est saine (facing, cross-body, axe ambigu gérés), pas la
  généralité. Seuils (`0.06`, `0.05`, fenêtre `0.12·fps`) exprimés en ratios de
  hauteur/fps mais **calés sur ce clip → revalider dès une 2e GT** (gaucher, revers 1
  main, slice).
- **Handedness non validée** (pas de gaucher dans la GT) : le prior est raisonné +
  conservateur (default droitier, super-majorité pour basculer), pas prouvé. Note far
  importante : le **mapping écran→absolu far est auto-correcteur** (le vote par côté
  encode la convention écran far, donc même un far droitier vu "miroir" est classé
  correctement — l'étiquette `handedness` far peut être inversée mais le FH/BH sort
  juste). Sur demo3 le far a < 4 votes → default droitier (ce qui suffit ici).
- **Plafond de DÉTECTION far, pas de classification.** Même avec la serve-gate
  corrigée, `detect_hits` snappe parfois la balle far un peu à côté du contact réel
  (ce qui déclenche la garde cross-body → abstention), et n'apparie que ~3-5/8
  contacts en LIVE. Améliorer le *recall/snapping* far de `detect_hits` est le
  prochain levier (domaine bounce/event) ; le classifieur, lui, rend chaque frappe
  *bien détectée* correcte (cache 7/7, LIVE replay 3/3, confusion 0).
- **Cross-body abstenu, pas classé.** Le vrai cross-body (style avancé, rare en
  amateur) ressort `unknown` — choix assumé (confusion 0 > +1 classé fragile). Le
  débloquer proprement = un classifieur temporel (cf. recherche web) une fois ≥200
  frappes GT multi-vidéos disponibles.

## Idée réutilisable

> **Valider l'accuracy sur un run LIVE, jamais sur un cache figé** (la leçon
> event-methodo, re-confirmée ici : 8/8 cache → 1/5 LIVE). Deux corollaires mesurés :
> 1. **Sur signal ambigu, ABSTENIR plutôt qu'overrider.** Le doc voulait croire le
>    swing sur désaccord (8/8 cache) ; LIVE le swing far est du bruit et l'override
>    crée une CONFUSION. Quand aucun signal géométrique ne sépare le vrai cas du faux
>    (vérifié : ni magnitude ni consistance), l'abstention tient la confusion à 0 des
>    deux côtés. Une abstention honnête ≫ une mauvaise classification confiante.
> 2. **Un seuil "haut dans l'image" doit être relatif au SUJET, pas à la frame.** La
>    serve-gate en fraction de frame supprimait 100 % des frappes far (le far est
>    légitimement haut). Rendue relative au joueur → far débloqué. Tout seuil
>    vertical sur un sujet à perspective variable doit être en unités de sa propre
>    taille (corollaire direct du zéro-hardcoding).
> Et : un **prior de population voté** (default majoritaire + super-majorité pour
> basculer) bat un signal physique fin (variance de poignet, facing par-frame)
> illusoire sur peu de données.

## Fichiers touchés (scope respecté : `vision/shots.py` + scorer + test)

- `vision/shots.py` :
  - réécriture de `classify_forehand_backhand` (composite : contact + prior +
    garde cross-body→abstention, **pas** d'override, **pas** de wrist-gate) ;
  - nouveaux `_swing_meandx` (poignet ACTIF, handedness-agnostique),
    `estimate_handedness_prior` (vote mode ball-side + default droitier),
    `classify_shots` (orchestrateur batch), `_ball_dx_at` ;
  - `detect_hits` : **serve-gate joueur-relative** (était frame-fraction) +
    **NMS par côté** (`last_by_side`). C'est le "detect_hits si besoin" de l'énoncé,
    strictement nécessaire pour que le far soit détecté/attribué. Byte-identique sur
    la régression event.
  - **Inchangés** : `shot_quality`, `detect_handedness` (legacy), les détecteurs.
- `run_pipeline_8s.py` — câblage : chemin **legacy** via `classify_shots` (prior voté
  une fois) ; chemin **methodo** source les contacts FH/BH d'un `detect_hits` sur les
  **tracks locked denses** (bien attribués near+far) apparié aux events HIT — sans
  toucher l'arbitration (les votes HIT de classify_events restent sur la pose brute).
- `tools/event_eval/score_fhb.py` — **scorer LIVE** FH/BH (apparie au `_stats.json`).
- `tools/event_eval/fhb_classifier_probe.py` — classifieur isolé @ contacts GT
  (cache, instantané) ; `fhb_live_replay.py` — rejeu fidèle du chemin prod sur les
  inputs LIVE dumpés ; `fhb_cache_probe.py` — bout-en-bout sur cache.
- `tests/test_fhb_regression.py` — garde-fou (7/7 appelées, ≤1 abstention, **confusion
  0** sur le cache).

## NON touché (scope)
`vision/events.py`, `vision/pose.py`, `models/` — intacts (contrainte de l'énoncé).
