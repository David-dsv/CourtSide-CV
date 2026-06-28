# CR — S3 : pente de direction ROBUSTE (Theil-Sen) — tue les vy-flips fantômes dus au jitter

**Branche :** `feat/s3-robust-slope` (depuis `feat/accuracy-overhaul`), worktree
`../CourtSide-CV-s3slope`. **Fichier touché : `vision/bounce.py` UNIQUEMENT** (+ tests/outils
force-add). ULTRACODE, measure-driven, vérifié adversarialement.

## TL;DR

Le bug : `_veto_slope` (`vision/bounce.py:841`) était un **moindres-carrés simple** (breakdown
point 0 %). Un **seul** pic de jitter du tracker (1 frame) sur un arc balistique propre **domine
la pente** et **inverse son signe** → un faux `vy_flip` → une **ancre BOUNCE mandatory fantôme**
dans `classify_events` (et un veto fantôme dans `vx_flip_veto`). Ce sont les MÊMES `_veto_slope` /
helpers que `vision/events.py` importe (`from vision.bounce import _veto_slope`), donc le fix au
**point source** propage aux DEUX chemins (legacy `vx_flip_veto` ET `--event-methodo`
`_direction_features`).

Le fix : `_veto_slope` devient un **estimateur de Theil-Sen** (médiane des pentes par paires
espacées en frames). Breakdown point ~29 % : un outlier parmi ≥4 points ne peut plus déplacer la
médiane. **Scale-free, sans paramètre** (la médiane EST le robustifieur — aucun seuil à régler,
aucune constante hardcodée). Cas dégénéré 2 points = **byte-identique** au lstsq (une seule pente
par paire), d'où felix intact.

**Résultats — un signal nuancé mais HONNÊTE :**
- **Replay caches (dump canonique figé + extrait dégradé)** : VRAI GAIN. Bounce F1
  **0.625 → 0.667** (+0.042) sur le canonique → **le rebond fantôme BOUNCE@231 disparaît** ; et
  sur l'extrait dégradé il tue le **BOUNCE@540** nommé dans le prompt (conf B→H 1→0). hit F1
  inchangé, recall inchangé.
- **Track LIVE frais (le `_stats.json` réel)** : **NO-OP strict** (bounce F1 0.526→0.526, conf
  H→B **0** tenue) — le ball-track live a un **autre** jeu de candidats que le dump figé, et ses
  fantômes (152/300/320/406/583) ne sont **PAS** des faux vy-flips de jitter mais des artefacts
  du **DP global + blobs static-FP** (le mur amont de `courtside-s3-event-recall-negative`). La
  pente robuste n'a rien à tuer ici → 0 gain, 0 régression. Mesuré sur le MÊME track (dump live
  rejoué lstsq vs Theil-Sen), instrument byte-for-byte vs `_stats.json`.
- **Partout** : **0 régression** — les 6 tests de régression BYTE-IDENTIQUES au baseline,
  conf H→B live reste 0.

Bilan : changement **strictement plus correct** (tue une CLASSE de bug — le faux vy-flip de jitter
— de façon générique et permanente, byte-identique ailleurs, felix prouvé), à shipper comme
robustesse. Il **ne résout pas** le mur de placement live (hors scope S3 : décodeur + reject
static-FP amont).

## Le mécanisme (vérifié, pas théorisé)

`tools/event_eval/diag_vyflip.py` mesure, pour chaque candidat, le `vy_flip` sous lstsq vs
despike vs Theil-Sen + le max |dy| par frame de chaque côté :

```
  frame  near-GT  | lstsq vyflip      | theilsen vyflip   | max|dy|pre post | PHANTOM?
    122  B120@2   | True  +0.26/-12.09 | False +0.00/-9.00 |   3.0  33.0     | <<< PHANTOM
    253  B255@2   | True  +4.50/-59.51 | True  +4.50/-66.67 |   5.0 118.0     |
    369  B369@0   | True  +0.57/ -8.86 | True  +0.50/ -9.00 |   2.0   9.0     |
    456  B456@0   | True +26.74/ -5.60 | True +26.20/ -5.50 |  45.0   7.0     |
    510  B511@1   | True  +0.77/ -5.46 | True  +0.80/ -5.40 |   2.0   7.0     |
PHANTOM vy-flips (lstsq=True, robust=False): [122]
```

- **Un seul fantôme : frame 122.** Côté pré ≈ PLAT (vy_pre +0.26), un **seul** saut +33 px côté
  post traîne le lstsq vers vy_post −12.09 → faux flip. Theil-Sen lit vy_pre **+0.00** → pas de
  flip. C'est exactement la pathologie « un pic de jitter fabrique un flip ».
- **Les 5 vrais rebonds (255, 369, 456, 511) gardent TOUS leur vy-flip** sous Theil-Sen
  (réflexion verticale réelle, pas un pic isolé). **Aucun vrai rebond n'est supprimé.**
- 122 est à 2 frames du **vrai** rebond GT 120 ; le rebond 120 reste détecté (pred BOUNCE
  contient toujours 122, match dans la tolérance). Seul l'effet **downstream** disparaît :
  l'ancre fantôme @122 corrompait la cadence/parité du DP global → un BOUNCE spurious @231.
  En enlevant l'ancre fantôme, le DP ne place plus le 231. Gain de **précision pur** (231 ∉ GT,
  aucun vrai rebond perdu, recall R tenu à 0.56).

## Chiffres AVANT / APRÈS

### Replay canonique (`live_pool.py demo3_methodo_inputs_fullvideo.json`)

| métrique | AVANT (lstsq) | APRÈS (Theil-Sen) |
|---|---|---|
| conf H→B / B→H | 1 / 1 | 1 / 1 (maintenu) |
| **bounce F1** | **0.625** (R 0.56) | **0.667** (R 0.56) |
| hit F1 | 0.500 (R 0.50) | 0.500 (R 0.50) |
| pred BOUNCE | [122,**231**,253,271,369,456,510] | [122,253,271,369,456,510] |
| spurious | (108,HIT),**(231,BOUNCE)**,(292,HIT),(487,HIT) | (108,HIT),(292,HIT),(487,HIT) |
| pool recall | 14/17 | 14/17 |

→ **le rebond fantôme (231,BOUNCE) tué**, bounce F1 +0.042, zéro perte ailleurs.

### Preuve indépendante — extrait dégradé (le fantôme NOMMÉ dans le prompt)

| `..._extracted_DEGRADED.json` | AVANT (lstsq) | APRÈS (Theil-Sen) |
|---|---|---|
| conf H→B / B→H | 0 / 1 | 0 / 0 |
| bounce F1 | 0.625 | 0.667 |
| spurious BOUNCE | …,**(540,BOUNCE)** | (231,BOUNCE) ; **540 disparu**, **570 récupéré** (≈GT569) |

→ le **même** fix robuste tue **aussi** le `BOUNCE@540` fantôme du prompt (pente non-robuste sur
un arc descendant propre + 1 pic), conf B→H 1→0. Deux entrées indépendantes, même mécanisme, même
sens d'amélioration ⇒ **robustesse générique, pas un hack mono-frame.**

## Preuve felix INTACT (les 2 tests bounce + les garde-fous events)

Tous mesurés sur le **code édité réel** (pas un monkeypatch) — **byte-identiques** au baseline :

| garde-fou | plancher | AVANT | APRÈS |
|---|---|---|---|
| `test_bounce_regression` (felix Kalman no-arg) | F1 ≥ 0.72 | 0.800 (TP12/FP2/FN4) | **0.800 (idem)** |
| `test_bounce_regression` (felix Kalman PROD turn-pass) | F1 ≥ 0.72 | 0.774 (TP12/FP3/FN4) | **0.774 (idem)** |
| `test_bounce_wasb_regression` (felix WASB) | F1 ≥ 0.80 | 0.865 (TP16/FP5/FN0) | **0.865 (idem)** |
| `test_vx_veto` (synthétiques + gain felix WASB) | F1 ≥ 0.90 | 0.9143, rejette [1609] | **0.9143, rejette [1609] (idem)** |
| `test_event_confusion_regression` (cache demo3) | conf 0/0, B≥0.80 H≥0.60 | 0/0, B 0.889, H 0.632 | **0/0, B 0.889, H 0.632 (idem)** |

Pourquoi byte-identique sur felix : les fenêtres felix (WASB/Kalman denses) lisent la pente sur
des côtés où la médiane des pentes par paires == lstsq quand il n'y a pas d'outlier (cas propre),
et un côté à 2 points est mathématiquement identique. La robustesse ne s'active QUE quand un pic
existe — et felix n'en a pas dans ses fenêtres de direction.

## La variante retenue + pourquoi (sweep mesuré)

5+ variantes testées end-to-end (canonique + les 4 garde-fous) via `tools/event_eval/probe_slope.py` :

| variante | canon bounce F1 | canon hit F1 | conf H→B/B→H | felix K | felix WASB | demo3 conf | retenu ? |
|---|---|---|---|---|---|---|---|
| **lstsq** (baseline) | 0.625 | 0.500 | 1/1 | ✅ | ✅ | ✅ 0/0 | — |
| **theilsen** ✅ | **0.667** | 0.500 | 1/1 | ✅ | ✅ | ✅ 0/0 | **OUI** |
| theilsen_g3 (≥3 pts→TS) | 0.667 | 0.500 | 1/1 | ✅ | ✅ | ✅ 0/0 | équivalent (TS pur suffit) |
| despike (±1) | 0.625 | 0.500 | 1/1 | ✅ | ✅ | ✅ 0/0 | no-op (rate le fantôme) |
| **despike (±2)** | 0.667 | 0.625 | **0/1** | ✅ | ✅ | ❌ **hit F1 0.526 < 0.60** | NON (casse le plancher) |
| conddespike (MAD) | 0.625 | 0.500 | 1/1 | ✅ | ✅ | ✅ 0/0 | no-op (seuil rate le pic) |
| huber/IRLS (MAD) | 0.625 | 0.500 | 1/1 | ✅ | ✅ | ✅ 0/0 | no-op (le pic survit l'IRLS) |

**Theil-Sen gagne** : seule variante qui (a) tue le fantôme, (b) tient **TOUS** les planchers
byte-identiques, (c) **sans paramètre** (pas de fenêtre `k` ni de multiplicateur MAD à régler —
un avantage « zéro hardcoding » décisif vs despike). `despike2` donnait un canonique plus joli
(hit F1 0.625, conf H→B 0) mais **casse le plancher hit F1 du cache demo3 (0.526 < 0.60)** : il
sur-lisse le vrai signal → REFUSÉ (la contrainte dure prime).

## Contraintes dures — état

- ✅ **confusion_H→B = 0 maintenu** : cache demo3 0/0 ; **`_stats.json` LIVE = 0** ; replay du
  dump canonique figé reste 1 (PRÉ-EXISTANT, territoire S2), non aggravé.
- ✅ **bounce F1 ≥ 0.625 (canonique)** : replay dump 0.625 → **0.667** ; live 0.526 (no-op, pas de
  régression — le plancher canonique du dump est tenu, le live est un autre track).
- ✅ **tous les tests de régression verts** (les 5 garde-fous + le nouveau `test_robust_slope.py`).
- ✅ **zéro hardcoding** : pente paramètre-libre (médiane), aucune constante px/fps ajoutée
  (littéraux AST du corps = {0,1,2}, tous structurels).

## Caveats honnêtes (à lire avant d'intégrer)

1. **La conf canonique H→B=1 / B→H=1 reste** — ce sont des artefacts PRÉ-EXISTANTS du décodeur
   global (parity-fill de B308→HIT@307 etc.), **territoire S2** (`docs/research/
   event-rootcause-2026-06-28-CR.md` §S3/§S2). S3 ne les aggrave pas et n'avait pas pour but de
   les résoudre (la pente n'en est pas la cause).
2. **Protection par taille de côté (corrigé après revue adverse).** Theil-Sen out-vote 1 outlier
   quand il y a assez de pentes propres : **n≥5 = confortablement protégé** ; **n=4 = marginal**
   (la médiane d'un nombre PAIR de pentes moyenne les deux rangs centraux) ; **n=3 = NON protégé**
   (1 outlier corrompt 2 des 3 pentes) → ces côtés courts retombent vers l'ancien comportement,
   acceptable car un signe lu sur si peu de points est déjà peu fiable (les gates avals s'abstiennent
   là). À `half=round(50*0.10)=5`, un côté plein couvre jusqu'à ~6 points → régime confortable, et
   le fantôme @122 y est.
3. **felix : la pente DIVERGE numériquement mais SANS changer de signe (corrigé).** felix n'est PAS
   sûr parce que « Theil-Sen diverge rarement » — il diverge sur **52/54** appels `_veto_slope`
   denses du chemin vx_veto. Il est sûr parce que **0 de ces divergences ne traverse zéro**
   (signe préservé) → chaque décision vx/vy_flip est byte-identique, et `detect_bounces_robust`
   (chemin détecteur WASB) **n'appelle jamais** `_veto_slope` (intact par construction).
4. **Mono-clip de validation accuracy** : le gain replay est mesuré sur demo3/tennis (canonique +
   extrait), précision-only (recall live R 0.556 inchangé). Le DP de `classify_events` est
   globalement couplé → sur un autre track (autre clip / jitter YOLO cross-env) le même
   step-shift ancre→cadence→DP pourrait aider OU nuire un slot de parité différent. **Re-confirmer
   sur ≥1 `_stats.json` live de plus avant de généraliser.** felix = garde-fou (intact), pas un 2e
   clip à gain.
5. **Le replay n'est pas le live** (leçon CR #1/#2, *vérifiée ici*) : le gain replay (231) ne
   reproduit PAS le track live (no-op) — j'ai validé sur un `_stats.json` LIVE frais ET sur un
   dump live rejoué, pas seulement le cache figé. C'est la bonne méthodo.

## Validation LIVE (`_stats.json` frais) — le résultat HONNÊTE : NO-OP sur CE track live

Commande : `run_pipeline_8s.py tennis.mp4 -s 73 -d 13 --device mps --match-mode` (depuis le
worktree, `vision/bounce.py` édité). Scoring : `tools/event_eval/score_stats.py` sur le vrai
`_stats.json`.

**Le track LIVE ≠ le dump figé** (leçon CR #1/#2 : « the cache lies in both directions »). Le
ball-track frais (YOLO/MPS) produit un **jeu de candidats DIFFÉRENT** du
`demo3_methodo_inputs_fullvideo.json` committé : pool **17/17** en live (vs 14/17 dans le dump),
spurious BOUNCEs live = **152, 300, 320, 406, 583** (≠ le 231 du dump). Pour isoler l'effet du
fix de la non-déterminisme du track, j'ai **dumpé CE run live** (`COURTSIDE_DUMP_METHODO_INPUTS`)
et rejoué les DEUX pentes sur le **MÊME track** :

| MÊME track live (`/tmp/s3_live_dump.json`, replay = `_stats.json` byte-for-byte) | AVANT (lstsq) | APRÈS (Theil-Sen) |
|---|---|---|
| conf H→B / B→H | 0 / 1 | **0 / 1 (identique)** |
| bounce F1 | 0.526 (R 0.556) | **0.526 (identique)** |
| hit F1 | 0.421 | **0.421 (identique)** |
| pool recall | 17/17 | 17/17 |

→ **Sur CE track live, Theil-Sen est un NO-OP. Aucune régression, aucun gain.** Et le vrai
`_stats.json` du pipeline confirme : conf H→B **0** (la contrainte dure tenue), B→H 1, bounce F1
0.526, hit F1 0.421 — exactement le replay.

**Pourquoi NO-OP (mesuré, `diag_vyflip.py /tmp/s3_live_dump.json`)** : les rebonds fantômes du
track live (152/300/320/406/583) ne sont **PAS** des faux vy-flips dus au jitter. Le diag ne
trouve que 2 « phantoms » de pente (177, 406) et ce sont des **cas dégénérés quasi-immobiles**
(vy_pre +0.00/+0.03, vy_post −0.00/−0.37, max|dy| 0–1 px) — des **blobs static-FP gelés** (177
près du frozen-FP B174 ; 406 = le blob de coin static-FP documenté dans
`courtside-s3-event-recall-negative`). Theil-Sen **dé-promeut bien** ces ancres vy-flip
dégénérées, **MAIS le BOUNCE spurious@406 SURVIT quand même** (`(406,'BOUNCE')` toujours dans le
spurious après) : le **DP global le re-dérive par cadence/parité**, pas via l'ancre vy-flip. C'est
EXACTEMENT le mur de `courtside-s3-event-recall-negative` / `courtside-events-plateau-ball-
tracking-wall` : **le mur live = le décodeur globalement couplé + les blobs static-FP amont, PAS
la pente.** La pente robuste ne peut pas, à elle seule, faire bouger le F1 live tant que ces deux
causes amont tiennent.

### Verdict honnête
- **Sur les caches replay (dump figé + extrait dégradé)** : VRAI GAIN net (+0.042 bounce F1,
  2 fantômes de jitter tués : 231 et 540), 0 régression.
- **Sur le track live frais** : **NO-OP strict** (0.526→0.526, conf H→B 0 tenue) — il n'y a pas de
  faux vy-flip de jitter à tuer sur ce track ; les fantômes live ont une cause amont (DP+static-FP).
- **Partout** : **0 régression** (les 6 tests byte-identiques, conf H→B live reste 0).

C'est un changement **strictement plus correct** (il tue les vrais artefacts de jitter là où ils
existent, et est byte-identique ailleurs — felix prouvé), à **shipper comme robustesse**, mais qui
**ne résout pas le mur de recall/placement live** (hors scope S3 — c'est le décodeur + le reject
static-FP amont, cf. `courtside-s3-event-recall-negative`). Le fix élimine une **classe** de bug
(faux vy-flip de jitter) de façon générique et permanente ; sur le clip hero précis, cette classe
n'est pas le facteur limitant aujourd'hui.

## Fichiers / livrables

- `vision/bounce.py` — `_veto_slope` → Theil-Sen (seul changement de prod).
- `tests/test_robust_slope.py` (force-add) — unit : arc propre == analytique ; 1 outlier ne flip
  PAS le signe (et lstsq SI → contraste prouvé) ; côté 2-pts == lstsq ; sparse → None.
- `tools/event_eval/diag_vyflip.py` (force-add) — thermomètre fantôme : liste les vy-flips
  lstsq vs robuste par candidat, flag PHANTOM. La colonne « lstsq » utilise un **lstsq LOCAL**
  (pas `vision.bounce._veto_slope`, désormais Theil-Sen) pour montrer le vrai contraste ancien/nouveau.
- `tools/event_eval/probe_slope.py` (force-add) — sweep : mesure une variante de pente end-to-end
  sur n'importe quel dump + les 4 garde-fous d'un coup (monkeypatch import-safe sur bounce + events).
  Sert aussi au **BEFORE/AFTER matched sur un track live** (`probe_slope.py {lstsq,theilsen} <dump>`).

## Note méthodo (race worktree partagé)
Un `.pyc` périmé + l'édition concurrente d'autres sessions ont fait apparaître transitoirement le
corps comme lstsq à 2 reviewers adverses (faux « REFUTED »). Résolu : valider TOUJOURS avec
`python -B` après une édition, et **committer tôt** (fait : la branche est poussée). cf.
`courtside-shared-branch-collision`, `courtside-edit-targets-main-tree`.
