# CR — Réconcilier S1+S2 dans le firewall `events.py` : la fuite `H→B=1` éliminée, f174 récupéré sur sa VRAIE physique

Branche : `feat/event-methodo-reconciled` (depuis `feat/event-methodo-prod` = S1 déjà mergé).
Fichier produit : `vision/events.py` (le classifieur) + cache de test (rebuild S2 100 % far-pose) + tests.
Statut : **fusion S1×S2 réussie SANS fuite. CACHE : confusion 0/0, bounce F1 0.947 (9/9), hit F1 0.778. LIVE : confusion 0/0 (déterministe, firewall-garanti), bounce/hit F1 = S1 (0.667/0.750, ZÉRO régression). `far_wrist_hits` passe OFF par défaut (validé apples-to-apples, cf. §4.0). Tous les tests verts (6 existants + 7 wrist-hits S2 + 4 réconciliation). felix intact (bounce.py intouché, OFF byte-identique).**

> Suite de `event-methodo-prod-CR.md` (S1) + `event-methodo-improve-CR.md` (S2) +
> mémoires `courtside-event-methodo-prod-resolved`, `courtside-event-methodo-improve`,
> `courtside-bounce-vs-shot-direction`, `courtside-ball-density-apex-bounce`.

---

## 0. TL;DR

| (CACHE demo3) | conf H→B | conf B→H | bounce F1 | hit F1 | f174 | f146 leak |
|---|---|---|---|---|---|---|
| S1 seul (vx-gate) sur le cache 100 %-far | 0 | **1** | 0.889 | 0.737 | ✗ | — (B→H leak à la place) |
| S2 seul (cache rebuild) | 0 | 0 | 0.900 | 0.700 | ✓ | — |
| **Fusion NAÏVE** (vy_bounce + bounce_traj + far-wrist) | **1** | 0 | 0.900 | 0.600 | ✓ | **OUI (f146→GT HIT 141)** |
| **RÉCONCILIÉ — far-wrist ON** | **0** | **0** | 0.947 | 0.737 | ✓ | non |
| **RÉCONCILIÉ — far-wrist OFF (DÉFAUT LIVRÉ)** | **0** | **0** | **0.947** | **0.778** | ✓ | non |

| (LIVE _stats.json, déterministe) | conf H→B | conf B→H | bounce F1 | hit F1 |
|---|---|---|---|---|
| S1 seul | 0 | 0 | 0.667 | 0.750 |
| **RÉCONCILIÉ (défaut far-wrist OFF)** | **0** | **0** | **0.667** (= S1, plafond recall upstream) | **0.750** (= S1, zéro régression) |

La fusion naïve donnait bounce F1 0.900 sur le cache MAIS rouvrait la confusion (`H→B=1`). Ce CR la ferme **et** monte le bounce cache à **0.947** — en récupérant f174 sur **sa propre physique de rebond sol** (l'angle de virage de la vitesse), pas sur le pont parasite qui était la fuite. En LIVE, le firewall **0/0 tient** et les F1 **égalent S1 sans régression** ; le gain bounce cache ne reproduit pas live (plafond de détection balle, §4c).

---

## 1. La frame qui fuyait : f146 (mesuré, pas deviné)

Le rallye GT de demo3 dans cette zone : **HIT@141 → BOUNCE@174 → HIT@196** (H-B-H, UN seul rebond).

La fusion naïve décode `H@134 — B@146 — H@162 — B@179 — H@196` (H-B-H-B-H, DEUX rebonds). Attribution par le matcher :
```
pred BOUNCE@146 -> GT HIT@141  d=5   <<< confusion_H->B = 1   (LA FUITE)
pred BOUNCE@179 -> GT BOUNCE@174 d=5  (le vrai TP = f174)
```

**Pourquoi f146 fuit (3 mesures) :**
1. Le générateur far-wrist (S2) récupère correctement le HIT far à **f134** (= GT 141).
2. L'**apex sortant** du coup (turn@146, un **y-MINIMUM** = point le plus haut à l'écran) + un `bounce_cand@153` parasite (sur la **branche descendante**, pas un apex) forment un cluster SÉPARÉ (134→146 = 12f > merge_gap 9).
3. Ce cluster orphelin a `hit=False`, `dmin=0.767 > near` → **pas hit_like** → le DP d'alternance est libre de l'étiqueter BOUNCE.

**Pourquoi supprimer f146 tuait f174 (l'enchevêtrement, la clé du bug) :** la récupération de f174 par S2 était **PARASITE** du pont de parité spurious `f146(B)/f162(H)`. Le DP exige une parité : `H@134 → B@179` est gap=45, step≈27, k=round(45/27)=**2 (PAIR)** ; or H→B est un *flip* → **parité violée → illégal**. Le DP a donc besoin d'un événement intermédiaire au demi-temps (~f161) pour basculer la parité. Le spurious f146/f162 le fournissait. **Retirer la fuite (requis pour 0/0) retire la récupération parasite de f174.** Les deux sont la même structure DP. (Diag : `tools/event_eval/eval_variant.py` + dump du cluster f146/f179.)

---

## 2. La logique correcte — combiner vy_bounce + bounce_traj + far-wrist SANS fuite

Trois greffes S2 reprises proprement (zones distinctes, non en conflit) :
- `detect_wrist_hits` (+ alias `detect_far_wrist_hits`) — générateur far-wrist, zone propre.
- helpers `_is_y_maximum` / `_is_edge_artifact` + bloc de merge far-wrist — zone propre.
- relaxation `bounce_traj = turn and (y_max or vx_preserved) and not hit` dans `hit_like`.

Et **S1 préservé** : `vy_bounce = bool(vy) and not vx_flip` (le vx-gate) reste dans `hit_like`, `bscore`, l'ancre et le dict d'event.

Puis **deux ajouts de réconciliation** dans `classify_events` / `_global_alternation_decode` :

### Fix A — CONTACT SHADOW (ferme la fuite f146)
Un cluster orphelin SANS coup (`hit=False`), **pas de forme rebond sol** (`not vy_bounce and not y_max`), qui tombe dans la fenêtre de l'arc sortant APRÈS un far-wrist hit committé → est l'arc sortant de CE coup, pas un nouveau rebond → forcé `hit_like` (interdit de BOUNCE).
```python
in_far_wrist_shadow = (not hit and not vy_bounce and not y_max
                       and any(0 < rep_fr - fwf <= fw_shadow for fwf in far_wrist_frames))
hit_like = (hit or (dmin < near and not bounce_traj) or in_far_wrist_shadow) and not vy_bounce
```
Gardé 3 fois pour ne JAMAIS supprimer un vrai rebond : (1) cluster sans coup ; (2) pas de forme rebond (un vrai rebond sol juste après un far hit — GT bounce@255, 7f après far hit@248 — a vy-flip/y-max, sort intact) ; (3) seulement APRÈS le hit. **Ne retire AUCUN candidat** → la cadence du rallye (PASS 2) est inchangée (contrairement à un re-clustering, qui décalait le DP et perdait f174).

### Fix B — GENTLE-APEX BOUNCE ANCHOR (récupère f174 sur sa VRAIE physique)
f179 (= GT 174) est un **y-MAXIMUM** au pied du joueur receveur (dmin 0.18, relaxé par `bounce_traj`), SANS vy_flip, SANS bounce_cand → il ne peut qu'être DP-rempli, et le DP le dropait par parité. On le ré-épingle sur **son propre signal de rebond sol** : l'**angle de virage du vecteur vitesse**.
```python
gentle_apex_bounce = (turn and y_max and not hit and not vy_bounce
                      and not in_far_wrist_shadow and not <far_wrist member>
                      and _velocity_turn_angle(rep_fr) <= 35.0)
# -> anchor = BOUNCE, +1.0 bscore, et MANDATORY dans le décodeur (bypass parité via le restart)
```
**Le discriminant (la même physique que `vx_flip_veto` / `detect_sharp_turns`)** : une frappe raquette INVERSE brutalement le vecteur vitesse (mesuré : f83/H81 **179°**, f335/H333 **160°**, f196/H196 **170°**) ; un rebond sol lointain ne fait que fléchir doucement la composante verticale tandis que l'horizontale passe tout droit → l'angle reste ~0° (**f179 0.0°**, f258/B255 0.9°). Seuil à **35°** (= moitié du plancher 70° de `detect_sharp_turns`) → une frappe ne peut JAMAIS qualifier. **Marge ~125°** (f179 à 0° vs traps à 160-179°). C'est le dual de l'ancre vy : vy prouve un rebond par inversion verticale ; ici un y-max à virage doux le prouve par la FORME sol + l'ABSENCE d'inversion raquette.

`soft_ymax` / re-clustering / cadence globale ont été ÉCARTÉS (cf. §4).

---

## 3. Résultat mesuré — CACHE

`python tools/event_eval/run_demo3.py` (méthodo, DÉFAUT far-wrist OFF) :
```
confusion_H->B = 0   confusion_B->H = 0
BOUNCE  R=1.000  F1=0.947  (9/9, f174 récupéré par l'ancre gentle-apex, SANS far-wrist)
HIT     F1=0.778  (seul miss = f333 ; far-wrist OFF => moins de FP qu'avec ON: 0.778 vs 0.737)
structure 130..200 : H@134 - B@179 - H@196  (= GT H141-B174-H196, le pont parasite f146/f162 supprimé)
# far-wrist ON (opt-in) : bounce 0.947, hit 0.737, + récupère f525 (test_wrist_hits_and_relaxation)
```

---

## 4. Résultat mesuré — LIVE (la leçon cache-trompeur, apprise trois fois)

### 4.0. La décision far-wrist : OFF par défaut (validée apples-to-apples sur la piste LIVE)

Première version de la réconciliation : far-wrist **ON** (fidèle à la greffe S2). Mesure LIVE (`run_pipeline_8s.py ... --event-methodo` + `score_stats.py`, **déterministe sur 3 runs**) → confusion **0/0** ✓ mais **hit F1 0.556** (vs 0.750 pour S1) : le générateur far-wrist sur-tire sur la pose far LIVE bruitée (+4 frappes far parasites). Diagnostic propre **apples-to-apples** (replay des MÊMES inputs live dumpés via `COURTSIDE_DUMP_METHODO_INPUTS`, à travers S1-base ET réconcilié — `tools/event_eval/eval_variant.py` + replay) :

| (replay sur inputs LIVE identiques) | conf | bounce F1 | hit F1 |
|---|---|---|---|
| S1-base | 0/0 | 0.667 | 0.750 |
| Réconcilié, far-wrist **ON** | 0/0 | 0.667 | **0.556** (−1 hit recall, +3 FP) |
| Réconcilié, far-wrist **OFF** | 0/0 | 0.667 | **0.750** (= S1) |

Et sur le CACHE, far-wrist OFF est **aussi meilleur** : bounce F1 0.947 (l'ancre gentle-apex récupère f174 INDÉPENDAMMENT du far-wrist) + hit F1 **0.778** (vs 0.737 ON — les FP far-wrist coûtent même sur le cache lisse). Le générateur far-wrist n'apporte uniquement, en propre, que f525 sur le cache — au prix d'une précision-frappe nette négative partout.

→ **Décision (entérinée) : `far_wrist_hits=False` par défaut**, opt-in via le flag pour le cas pose-far dense/cache (les tests `test_wrist_hits_and_relaxation` l'activent pour verrouiller le chemin f525). C'est la **meilleure config CACHE ET LIVE**, et elle ramène la live au comportement propre de S1 — **zéro régression**.

### 4.1. Résultat LIVE final (far-wrist OFF par défaut)

`python run_pipeline_8s.py tennis.mp4 -s 73 -d 13 --device cpu --event-methodo`
puis `python tools/event_eval/score_stats.py data/output/tennis_annotated_stats.json`.

Mesure faisant autorité = **replay déterministe des inputs live réels** (dump `COURTSIDE_DUMP_METHODO_INPUTS`) à travers le `classify_events` réconcilié (défaut far-wrist OFF) — c'est exactement ce que le pipeline exécute, sans le bruit de la détection balle :
```
RECONCILED (far-wrist OFF) replay sur inputs LIVE :
  confusion_H->B = 0   confusion_B->H = 0
  BOUNCE F1 = 0.667  (5 TP / 1 FP / 4 FN ; misses 62,120,174,308 = sans candidat, §4c)
  HIT    F1 = 0.750  (6 TP / 2 FP / 2 FN)
  -> BYTE-IDENTIQUE à S1-base sur les mêmes inputs (zéro régression)
```
**Run END-TO-END live confirmant (pas un replay — `_stats.json` réel, défaut far-wrist OFF) :**
```
predicted: 6 bounces / 8 shots   vs GT 9 bounces / 8 shots
  confusion_H->B = 0   confusion_B->H = 0
  BOUNCE  P=0.833 R=0.556 F1=0.667  (TP=5 FP=1 FN=4 ; misses 62,120,174,308)
  HIT     P=0.750 R=0.750 F1=0.750  (TP=6 FP=2 FN=2)
```
→ identique au replay. (La première version far-wrist ON donnait `0/0, bounce 0.667, hit 0.556` sur ces mêmes inputs — d'où le passage OFF, §4.0. La détection balle est déterministe sur ce clip, vérifiée sur plusieurs runs.)

**Lecture honnête :**
- ✅ **L'objectif DUR de la session — confusion 0/0 EN LIVE — est ATTEINT** (le `_stats.json` réel scoré contre la GT : `H→B=0, B→H=0`), déterministe. Le firewall (`_reward` renvoie `None` sur `hit_like`) le **garantit par construction**. C'était le but : *« la confusion rebond/frappe DISPARAÎT du `_stats.json` de PROD »*.
- ⚖️ **bounce F1 live = 0.667 = S1**. Le gain cache f174 (0.842→0.947) ne reproduit PAS en live car, sur la piste balle LIVE de ce clip, **les 4 rebonds manqués (62, 120, 174, 308) n'ont AUCUN candidat dans le pool** (`b_cands` live = [81, 253, 456], aucun turn proche — mesuré §4c). C'est un **plafond de GÉNÉRATION DE CANDIDATS (détection balle, UPSTREAM)** : aucune logique de `events.py` ne peut étiqueter un candidat inexistant. f174 est **live-irréductible ici** (l'ancre gentle-apex ne peut s'allumer sur un candidat absent). **Leçon cache-trompeur confirmée une 3ᵉ fois.**
- ✅ **hit F1 live = 0.750 = S1** (plus de régression depuis le passage far-wrist OFF).

### 4c. Preuve que les rebonds manqués sont UPSTREAM (génération de candidats)

Replay des inputs live (`/tmp/replay_live_inputs.py` sur le dump `COURTSIDE_DUMP_METHODO_INPUTS`) :
```
GT bounce 62 : NO candidate (ni b_cand ni turn dans ±8f) -> candidate-generation (UPSTREAM)
GT bounce 120: NO candidate -> UPSTREAM
GT bounce 174: NO candidate -> UPSTREAM
GT bounce 308: NO candidate -> UPSTREAM
```
Les 4 misses live sont **identiques entre S1-base et réconcilié** et **tous sans candidat** → c'est la détection/piste balle live (cf. `courtside-ball-density-apex-bounce` : les apex far-court live ont un creux de ~1-6px que la piste live perd), PAS la classification. **Prompt : « documente précisément pourquoi une frame est irréductible » → fait, par mesure.**

---

## 5. Pourquoi CE fix et pas les alternatives (revue adverse 4 familles + sceptique)

Exploré en parallèle (workflow, 4 familles + vérif adverse), mesuré sur `eval_variant.py` :

| famille (cache, far-wrist ON au moment de l'explo) | conf | bounce | hit | f174 | verdict |
|---|---|---|---|---|---|
| **gentle-apex angle (ADOPTÉ)** | 0/0 | **0.947** | 0.737 | ✓ | plus robuste : 0/0 sur le plus large balayage step0 |
| soft y-max restart | 0/0 | 0.947 | 0.737 | ✓ | fuite **B→H=1 dès step0_frac ≤ 0.40** (sceptique : « narrow sweet spot ») |
| local cadence (dual-k) | 0/0 | 0.947 | 0.700 | ✓ | global, plus risqué hors demo3 |
| suppress phantom f162 | 0/0 | 0.889 | 0.842 | ✗ | f162 déjà droppé → ne récupère pas f174 |

**Le gentle-apex gagne par ROBUSTESSE** : son discriminant (l'angle de virage du vecteur vitesse) est indépendant de la cadence, là où l'alternative soft-restart fuit aux extrêmes du seed (marge 1.5 frame). Le sceptique : « COULD NOT REFUTE … f174 recovery is LEGITIMATE (genuine y-maximum, vx preserved, real b_cand on the same arc) … firewall intact by construction (restart only when rB is not None, None on hit_like). » Plateau 0/0 final (config livrée, far-wrist OFF) = **step0_frac 0.50–0.75** (défaut prod 0.55, au centre) — testé par `test_event_reconciled::test_firewall_plateau_over_step0`.

**Anti-overfit honnête** : le `strong_bounce` brut (tout y-max → mandatory) fuyait `H→B=2` (f83/GT81, f335/GT333) — c'est l'angle de virage qui sépare proprement le vrai rebond (0°) des apex de frappe (160-179°). Sans ce discriminant, le fix sur-tire.

---

## 6. Non-régression (tous verts)

```
test_event_confusion_regression  PASS  (DÉFAUT far-wrist OFF : H->B=0 B->H=0, bounce F1 0.947 hit F1 0.778 ; planchers relevés 0.90/0.70)
test_wrist_hits_and_relaxation   PASS  (7 cas S2, far-wrist OPT-IN ON : f525 + f174 récupérés, plateau near_wrist 0.40-0.60)
test_event_reconciled            PASS  (NOUVEAU, 4 cas : 0/0+F1, f146-no-leak/f174-bounce, plateau step0 0.50-0.75, toggle far-wrist)
test_bounce_regression           PASS  (felix Kalman F1 0.800 >= 0.72)
test_bounce_wasb_regression      PASS  (felix WASB  F1 0.865 >= 0.80)
test_vx_veto                     PASS  (felix F1 0.9)
test_sharp_turns                 PASS
test_far_coverage                PASS
```
Le fix ne touche QUE le chemin méthodo (`classify_events` + `_global_alternation_decode`). `vision/bounce.py` et `_anchor_parity_decode` (legacy/OFF) byte-intouchés ; `run_pipeline_8s.py` byte-identique à S1 → **chemin OFF/legacy inchangé**. La fuite poseless (`players_per_frame=None` AVEC candidats) est **pré-existante S1** (vérifié sur `feat/event-methodo-prod`), pas une régression de ce CR ; le cas réaliste (slots None, chemin felix) passe.

---

## 7. Leçon (pour la mémoire)

**Quand un gain (f174) est PARASITE d'un bug (la fuite f146), corriger le bug détruit le gain — il faut alors re-fonder le gain sur une évidence INDÉPENDANTE et légitime.** Ici f174 vivait sur le pont de parité spurious qui ÉTAIT la confusion ; le ré-épingler a demandé un nouveau signal physique (l'angle de virage de la vitesse, famille `vx_flip_veto`) qui distingue un rebond sol doux (~0°) d'une frappe raquette (160-179°). Et : **valider sur un run LIVE end-to-end**, jamais le cache seul (leçon S1, re-confirmée ici).

**Un générateur de candidats validé seulement sur cache peut sur-tirer en live.** Le far-wrist de S2 était propre sur le cache (pose far lissée/gap-fillée) mais sur-tire sur la pose far LIVE bruitée → il a fallu une mesure **apples-to-apples sur les MÊMES inputs live** (replay du dump `COURTSIDE_DUMP_METHODO_INPUTS` à travers S1 ET réconcilié) pour le prouver sans bruit de piste, puis le passer **OFF par défaut** (opt-in conservé + testé). Le replay-sur-inputs-live est l'outil clé : il isole le classifieur du non-déterminisme de la détection balle.

## 8. Outillage ajouté
- `tools/event_eval/eval_variant.py` — score un `vision/events.py` candidat SANS toucher le fichier live (import dynamique). Sert à itérer/comparer des variantes en parallèle (utilisé par l'exploration adverse 4-familles).
- Replay des inputs live dumpés (`COURTSIDE_DUMP_METHODO_INPUTS` → `classify_events` de S1 vs réconcilié sur les MÊMES inputs) — la mesure apples-to-apples déterministe qui a tranché la décision far-wrist OFF.
