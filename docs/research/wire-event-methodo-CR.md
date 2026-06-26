# CR — Câblage de la méthodo d'events en PROD (`--event-methodo`)

Branche : `feat/wire-event-methodo` (depuis `feat/accuracy-overhaul`).
Fichiers : `run_pipeline_8s.py` (principal), `vision/events.py` (1 adaptateur shape).
Statut : **wiring complet, non-régressif (OFF byte-identique), gated derrière un flag, défaut OFF.**

---

## 1. Ce qui a été câblé + où

`classify_events` (vision.events, FIGÉ) est branché comme **couche d'arbitrage unifiée
bounce-vs-hit**, derrière `--event-methodo` (défaut OFF). OFF = comportement legacy
**strictement inchangé** (détecteurs indépendants + `vx_flip_veto` + `detect_hits`).

### Le problème d'ordre, résolu
`classify_events` a besoin EN MÊME TEMPS des candidats rebond, des candidats frappe,
de la trajectoire pré-spline + masque, ET des poses (produites en Pass 1.5, APRÈS la
détection rebond de Pass 1). L'arbitrage est donc inséré **après Pass 1.5**, dans le
shot pre-pass, quand poses + candidats coexistent.

### Points d'insertion (numéros de ligne finaux)
| l. | quoi |
|----|------|
| 602 | flag `--event-methodo` (argparse) |
| 833 | `event_methodo = bool(args.event_methodo)` |
| 1124-1147 | **Pass 1 : détection rebond SKIPPÉE quand ON** (le classifieur possède son propre firewall ; garde `all_ball_centers` = spline propre pour `smoothed_centers`) |
| 1163 | `--dump-bounces` Pass-1 skippé quand ON (dump post-arbitrage à la place) |
| 1223-1250 | `_build_bounce_by_frame()` : helper factorisé (depth+quality) **partagé** legacy/methodo → forme d'enregistrement bounce identique |
| 1332 | `if event_methodo and not args.demo_override:` → bloc d'arbitrage |
| 1350-1354 | candidats : `detect_bounces_robust` + `detect_turning_points` ∪ `detect_sharp_turns` (recette demo3 exacte) |
| 1355-1372 | `methodo_ppf` = `player_tracks` (lockés gap-fillés) reshapés `[near, far]` par frame |
| 1373-1377 | **`classify_events(...)`** appelé |
| 1380-1383 | events BOUNCE → `bounce_events` → `_build_bounce_by_frame` (rebuild) |
| ~1395-1410 | `direction_state_by_frame` dérivé de `features.vx_flip` (preserved=metric) |
| ~1413-1450 | events HIT → `shot_by_frame` (side/fhb/quality/speed/x/y/human_id) |

### Mapping de la sortie → contrats Pass 2
- **BOUNCE events** → `bounce_events` (liste `(frame,x,y)`) → `_build_bounce_by_frame`
  → `bounce_by_frame[fidx] = (x, y, depth, avg_speed, quality)` — **forme 5-tuple
  identique** au legacy (mêmes `classify_bounce_depth` + fenêtre vitesse + formule
  qualité). Consommé tel quel par Pass 2 / minimap / `_stats.json`.
- **HIT events** → `shot_by_frame[frame] = {side, fhb, quality, speed, x, y, human_id}`
  — **clés exactes** attendues par le HUD/minimap/stats/match-mode. Chaque HIT est
  ré-apparié à son enregistrement `detect_hits` d'origine (même frame) pour récupérer
  `player_side` + `classify_forehand_backhand` + `human_id` ; un HIT de remplissage
  parité (sans candidat) retombe sur un side géométrique (y vs net) + `fhb="unknown"`.
- **`direction_state`** : un BOUNCE dont `vx_flip is False` (sens préservé = signature
  bounce confirmée) → `"metric"`, sinon `"unconfirmed"` — même sémantique que le veto.

---

## 2. Preuve — `_stats.json` réel, flag ON, vs GT demo3

Run : `run_pipeline_8s.py tennis.mp4 -s 73 -d 13 --device cpu --event-methodo`
(Kalman par défaut, comme le cache demo3 où WASB est aveugle à 10 %.)
Scoring : `_stats.json` (frames absolues, offset -3650) vs `tools/event_eval` GT
(9 bounces, 8 shots, tol ±8f), même matcher de confusion croisée que le cache.

| métrique | OFF (legacy, réel) | ON (methodo, réel) | cache (preuve, frozen) |
|---|---|---|---|
| confusion **H→B** | 0 | **4** | **0** |
| confusion **B→H** | 3 | **4** | **0** |
| bounce recall | **1/9** (F1 0.182) | **5/9** (F1 0.370) | **8/9** (F1 0.842) |
| hit recall | 4/8 (F1 0.381) | 3/8 (F1 0.261) | 6/8 (F1 0.632) |
| # bounces prédits | 2 | 18 | 8 |

**Le gain demo3 NE se reproduit PAS pleinement en prod.** La méthodo améliore le
RAPPEL bounce (1/9 → 5/9) mais **dégrade la précision et casse la garantie 0/0 de
confusion** (4/4). Diagnostic complet ci-dessous (§4) : ce n'est **pas** un bug de
câblage — l'échec se reproduit hors-ligne sur les inputs prod dumpés, et le câblage
appelle fidèlement le classifieur figé. La cause est **la qualité des inputs prod**.

---

## 3. Confirmation flag OFF = BYTE-IDENTIQUE

Méthode rigoureuse : run du code **pré-câblage** (`HEAD~`, worktree à `17c1ce9`) et
du code câblé **flag OFF**, mêmes args (`tennis.mp4 -s 73 -d 13`), puis `diff` des
deux `_stats.json` :

```
same top-level keys: True
byte-identical stats JSON: True
```

→ **`_stats.json` octet-pour-octet identique.** Le flag OFF ne touche à rien. La seule
modif du chemin OFF est l'extraction `_build_bounce_by_frame` (refactor pur, même
math, même ordre de logs) et le garde `dump_bounces and not event_methodo` (équivaut
à `dump_bounces` quand `event_methodo=False`).

Tests (tous verts, OFF + ON) : `test_bounce_regression`, `test_bounce_wasb_regression`,
`test_vx_veto`, `test_event_confusion_regression`, `test_sharp_turns`,
`test_match_tracker`, `test_annot_clip`, `test_shot_guard`, `test_fatigue`,
`test_rally_outcome`. (`test_shots_regression` cité au prompt n'existe pas dans le
repo.) `py_compile` OK sur les deux fichiers.

---

## 4. Diagnostic honnête — pourquoi la prod diverge du cache

Dump des inputs réels d'arbitrage (Kalman & RTMW) comparés au cache figé :

| input | prod (live) | cache (frozen) |
|---|---|---|
| couverture balle | 549/650 (84 %) | 441/650 (68 %) |
| **poses FAR (P1)** | **152 (23.4 %)** | **511 (78 %)** |
| turning points | 39 | 25 |
| candidats rebond | 2 | 3 |

**Cause racine = la couverture pose du joueur FAR (lointain).** Le firewall de
`classify_events` garantit confusion_H→B = 0 par un gate logique « hit_like (=
balle-au-poignet OU swing) ⇒ jamais un bounce ». Mais ce gate **ne peut pas
s'activer si la pose du joueur far n'existe pas** : sur ce broadcast, le
`TwoPlayerTracker` ne verrouille le petit joueur far que **23.4 %** du temps (le cache
avait 78 %). Résultat : les frappes far-court ne sont pas marquées hit_like → leurs
turning points fuient en BOUNCE → confusion_H→B = 4.

**Ce N'EST PAS la faute du backend pose.** Testé `--pose-backend rtmw` (rtmlib, 133
kpts, censé gagner sur le far) : `methodo_ppf far` reste **152** (identique à yolo),
car le plafond est l'**association far-slot du TwoPlayerTracker** (23.4 %), pas le
modèle de pose. RTMW abaisse un peu la confusion (4/4 → 2/2 sur les inputs dumpés)
via de meilleurs poignets near, mais reste loin du 0/0.

Le cache (78 % far) provenait d'une capture spécifique (`build_demo3_event_cache.py` :
`court_zones=None` + hint balle WASB + un run où le tracker verrouillait mieux le far).
La prod recalcule tout en live → trajectoire + poses différentes → le classifieur,
figé et correct, reçoit simplement des inputs plus pauvres.

Bug réel corrigé en passant (caché par le cache) : `_nearest_player_dist_norm` faisait
`if not box:` → `ValueError` sur un `box` numpy (chemin pose live ; le cache JSON
désérialise `box` en liste donc jamais déclenché). Corrigé en `box is None or
len(box) < 4` (robustesse de forme pure, 0 changement de logique ; tests events + driver
demo3 reproduisent toujours 0/0 & F1 0.842).

---

## 5. Préservation des chemins spéciaux

- **`--demo-override`** : GAGNE. Quand `args.demo_override` est set, le bloc
  d'arbitrage est **court-circuité** (`if event_methodo and not args.demo_override`)
  → le chemin legacy honore l'override (bounces manuels + strikes), méthodo ignorée.
  C'est le chemin hero curé ; le mélanger à l'arbitrage serait contradictoire.
- **match-mode** : `human_id` reste correct — chaque event HIT alimente `shot_by_frame`
  avec `human_id_for_side(match_result, side, frame)`, exactement comme le legacy.
- **rally / HUD ÉCHANGE N** : `segment_rallies_relative` lit `shot_by_frame.keys()` +
  `["fhb"]` + `bounce_by_frame.keys()` ; le mapping préserve ces clés → numérotation
  intacte.
- **minimap COURT RADAR** : lit `bounce_by_frame[f] = (x,y,depth,...)` + `shot["x"/"y"]`
  → 5-tuple/clés préservés → radar OK.
- **`_stats.json`** : `direction_state`, `shots[].player_side/stroke/quality/speed_kmh/
  human_id`, `bounces[].depth/speed_kmh/quality` → tous alimentés à l'identique.
- **felix (pauvre en poses)** : ON ne crashe PAS. felix `-s 0 -d 6` → ON **0 bounce /
  0 hit** (abstention propre du firewall+alternation), vidéo écrite, audio ré-attaché ;
  **parité exacte avec OFF** (0/0 aussi). Dégradation gracieuse confirmée.

---

## 6. Reco sur le défaut du flag → **OFF** (le PM tranche)

**Garder `--event-methodo` OFF par défaut.** Justification :
1. Sur le broadcast par défaut (Kalman + yolo-pose), ON **casse la garantie 0/0**
   (confusion 4/4) et sur-produit des rebonds (18 vs 8) — un recul de précision/
   confiance visible dans la vidéo et `_stats.json`.
2. Le gain réel de la méthodo (0/0, recall 8/9) **dépend d'une pose far dense** (≈78 %)
   que la prod ne fournit pas (23.4 %, plafonnée par le TwoPlayerTracker, pas le backend).
3. OFF est prouvé byte-identique → zéro risque de régression en l'activant pas.

**ON deviendra activable** quand la couverture pose far montera (le vrai levier =
fiabiliser l'association far-slot du `TwoPlayerTracker`, p.ex. seeding par la balle /
calibration / un détecteur far dédié — PAS le backend pose, déjà réfuté). À ce
moment-là, ré-évaluer ON sur un clip à far-pose ≥ ~70 % et basculer le défaut si 0/0
tient end-to-end.

**Cas où la méthodo dégrade vs l'ancien chemin (à garder OFF)** : tout clip où le
joueur far est petit/peu détecté (broadcast oblique, far-court). C'est précisément le
régime de ce benchmark → d'où le défaut OFF.

---

## 7. TL;DR
Câblage **complet et propre**, derrière `--event-methodo` (défaut OFF). OFF = legacy
**byte-identique** (prouvé par diff vs HEAD pré-câblage + 10 tests verts). ON tourne
end-to-end (tennis + felix, sans crash) et **mappe correctement** la sortie unifiée sur
les contrats `bounce_by_frame`/`shot_by_frame`/`_stats.json`/HUD/minimap/match-mode,
demo-override préservé. **Mais** le gain demo3 ne se reproduit pas en prod : la pose far
(23.4 % vs 78 % du cache) affame le firewall → confusion 4/4 au lieu de 0/0. Diagnostic
prouvé côté inputs (pas câblage). **Défaut OFF ; activer quand la pose far sera dense.**
