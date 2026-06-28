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

**Résultat (replay canonique `tennis.mp4 -s 73 -d 13`)** : bounce F1 **0.625 → 0.667** (+0.042),
**le rebond fantôme BOUNCE@231 disparaît**, hit F1 inchangé 0.500, conf H→B/B→H inchangées (1/1,
maintenu — pas aggravé), recall pool 14/17 inchangé. **Tous les garde-fous felix/demo3 verts et
BYTE-IDENTIQUES** au baseline (zéro régression). Fix confirmé **générique** : il tue aussi le
fantôme BOUNCE@540 du prompt sur l'extrait dégradé (preuve indépendante).

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

- ✅ **confusion_H→B = 0 maintenu** : cache demo3 reste 0/0 ; canonique reste 1 (PRÉ-EXISTANT,
  territoire S2 — voir ci-dessous), non aggravé.
- ✅ **bounce F1 ≥ 0.625 (canonique)** : 0.625 → **0.667**.
- ✅ **tous les tests de régression verts** (les 5 ci-dessus + le nouveau `test_robust_slope.py`).
- ✅ **zéro hardcoding** : pente paramètre-libre (médiane), aucune constante px/fps ajoutée.

## Caveats honnêtes (à lire avant d'intégrer)

1. **La conf canonique H→B=1 / B→H=1 reste** — ce sont des artefacts PRÉ-EXISTANTS du décodeur
   global (parity-fill de B308→HIT@307 etc.), **territoire S2** (`docs/research/
   event-rootcause-2026-06-28-CR.md` §S3/§S2). S3 ne les aggrave pas et n'avait pas pour but de
   les résoudre (la pente n'en est pas la cause).
2. **Protection n=3.** Theil-Sen est garanti contre 1 outlier dès **n≥4** (3 des 6 pentes par
   paires touchent l'outlier → la médiane tient). À **n=3** (3 pentes, l'outlier en corrompt 2),
   un outlier peut encore biaiser — mais à `half=round(50*0.10)=5`, un côté plein couvre jusqu'à 6
   points : le régime utile est protégé. Le fantôme @122 (≥4 pts/côté) est dans ce régime.
3. **Mono-clip de validation accuracy** : le gain accuracy est mesuré sur demo3/tennis (canonique
   + extrait). felix sert de garde-fou (intact), pas de second clip à gain. Cohérent avec l'état
   du repo (un seul GT events). Re-mesurer cross-clip quand un 2e GT events existera.
4. **Le replay n'est pas le live.** Leçon CR #1 : valider sur un `_stats.json` LIVE frais, pas
   seulement le cache. → run live ci-dessous.

## Validation LIVE (`_stats.json` frais)

Commande : `run_pipeline_8s.py tennis.mp4 -s 73 -d 13 --device mps --match-mode` (depuis le
worktree, `vision/bounce.py` édité). Scoring : `tools/event_eval/score_stats.py`.

> _(rempli après la fin du run LIVE — voir section finale du commit.)_

## Fichiers / livrables

- `vision/bounce.py` — `_veto_slope` → Theil-Sen (seul changement de prod).
- `tests/test_robust_slope.py` (force-add) — unit : arc propre == analytique ; 1 outlier ne flip
  PAS le signe (et lstsq SI → contraste prouvé) ; côté 2-pts == lstsq ; sparse → None.
- `tools/event_eval/diag_vyflip.py` (force-add) — thermomètre fantôme : liste les vy-flips
  lstsq vs robuste par candidat, flag PHANTOM.
- `tools/event_eval/probe_slope.py` (force-add) — sweep : mesure une variante de pente end-to-end
  sur le canonique + les 4 garde-fous d'un coup (monkeypatch import-safe sur bounce + events).
