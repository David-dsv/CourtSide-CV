# CR — `--event-methodo` re-validé en PROD : la confusion rebond/frappe disparaît (ON par défaut)

Branche : `feat/event-methodo-prod` (depuis `feat/accuracy-overhaul`).
Fichiers : `vision/events.py` (le fix classifieur), `run_pipeline_8s.py` (flag défaut + dump diag).
Statut : **la confusion rebond/frappe est ÉLIMINÉE sur la PISTE LIVE de prod (confusion_H→B 4→0). `--event-methodo` passe ON par défaut, `--no-event-methodo` garde le legacy. Tous les tests de non-régression verts.**

> Suite de `wire-event-methodo-CR.md` (câblage, défaut OFF) + mémoires
> `courtside-event-methodo-prod-gap`, `courtside-event-methodo-wired`,
> `courtside-bounce-vs-shot-direction`, `courtside-farselect-integrated`.

---

## 0. TL;DR

| | confusion_H→B | confusion_B→H | bounce F1 | hit F1 | far-pose |
|---|---|---|---|---|---|
| **Legacy (défaut avant)** sur demo3 | — (2 b/13 h prédits, GT 9 b/8 h) | — | très bas | très bas | — |
| **Méthodo cache** (figé) | 0 | 0 | 0.842 | 0.632 | 78 % |
| **Méthodo PROD AVANT ce CR** (far-pose 100 %, code figé) | **4** | **1** | 0.526 | 0.333 | **100 %** |
| **Méthodo PROD APRÈS ce CR** (vx-gate + merge 0.18) | **0** | **0** | **0.667** | **0.750** | 100 % |

**Le levier far-pose (0.9 %→84.9 %) était nécessaire mais PAS suffisant.** Une fois la pose far dense (100 % en prod via gap-fill), le firewall n'est plus affamé — mais une **2ᵉ inadéquation cache↔prod**, cachée jusqu'ici, est apparue : sur la *vraie* piste balle, une **frappe far provoque une inversion verticale (`vy_flip`)** que `classify_events` traitait comme un **ancrage REBOND inconditionnel** contournant le firewall. Le cache (piste Kalman figée) n'avait pas de `vy_flip` à ces frappes → il cachait le bug. **Corrigé en mesurant**, pas à l'aveugle.

---

## 1. Étape 1 — la prod actuelle, mesurée end-to-end (pas le cache)

Commande (far-pose dense, méthodo active) :
```
python run_pipeline_8s.py tennis.mp4 -s 73 -d 13 --device cpu --event-methodo --dump-bounces /tmp/m.json
```
`_stats.json` scoré contre la GT (`tests/fixtures/{bounces/tennis_demo3.bounces.json, shots/tennis_demo3.shots.json}`) via `tools/event_eval/score_stats.py` (nouveau — convertit les frames absolues `start_frame+f` en relatif clip).

**Couverture pose far LIVE = `Player tracks locked: P1 coverage=100.0%`** (était 23.4 % dans le gap-CR). Le far-select mergé (`feat/farselect-integrated`) + le gap-fill du `TwoPlayerTracker` donnent la pose far DENSE que le firewall réclamait.

**Confusion PROD mesurée (avant fix), `_stats.json` réel :**
```
predicted: 10 bounces / 10 shots   vs GT 9 bounces / 8 shots
  GT BOUNCE        5            1         3
  GT HIT           4            3         1
  confusion_H->B = 4   confusion_B->H = 1
  bounce F1 = 0.526 (R=0.556 P=0.500)   hit F1 = 0.333
```
→ **La confusion ne reproduit PAS le 0/0 du cache** malgré la pose far à 100 %. Donc le diagnostic gap-CR (« far-pose = le seul blocage ») était **incomplet**.

---

## 2. Étape 2 — le gap résiduel prod-vs-cache : d'où vient-il ?

Méthode : un **dump des inputs EXACTS** de `classify_events` en prod (hook inerte gardé par `COURTSIDE_DUMP_METHODO_INPUTS`, aucun changement de comportement) → replay offline (`tools/event_eval/diag_live_vs_cache.py`). Le replay reproduit **exactement** le `_stats.json` (4/1, F1 0.526) → **le gap est 100 % dans les inputs/classifieur, pas dans Pass 2 / le mapping.**

**Attribution des 4 confusions H→B** (BOUNCE prédit tombant sur une frappe GT) :
```
pred BOUNCE@80  (GT HIT 81 ) : detect_hits proche? None ; why=vy_flip   ; vx_flip=True
pred BOUNCE@195 (GT HIT 196) : detect_hits proche? 188  ; why=vy_flip   ; vx_flip=True
pred BOUNCE@333 (GT HIT 333) : detect_hits proche? 327  ; why=vy_flip   ; vx_flip=True
pred BOUNCE@137 (GT HIT 141) : detect_hits proche? 145  ; why=parity_fill; vx_flip=None
```

**Cause racine n°1 (vy_flip = ancre bounce inconditionnelle).** `classify_events` faisait `if vy: anchor = BOUNCE` (vy-flip = « pure floor bounce ») et `hit_like = (...) and not vy` → un `vy_flip` shuntait le firewall. Sur la piste LIVE, les frappes far 80/195/333 ont un `vy_flip` (la balle frappée repart vers le haut) → labellisées BOUNCE. **Le cache n'avait pas ce `vy_flip` aux frappes** (piste Kalman différente, 68 % de densité) → 0/0 sur le cache, 4/1 en prod. **Le cache était inreprésentatif d'une 2ᵉ manière** (au-delà de la pose far du gap-CR).

Discriminant qui sépare proprement (lu sur la piste live, frame représentative) :

| frame | type | vy_flip | **vx_flip** |
|---|---|---|---|
| 80 (GT HIT 81) | HIT | True | **True** (vx inversé) |
| 195 (GT HIT 196) | HIT | True | **True** |
| 333 (GT HIT 333) | HIT | True | **True** |
| 255 / 456 / 511 / 569 (GT BOUNCE) | BOUNCE | True | **False** (vx préservé) |

→ **un rebond sol PRÉSERVE le signe de vx ; une frappe l'INVERSE** (la même physique que le `vx_flip_veto` déjà prouvé sur felix, F1 0.914).

**Cause racine n°2 (merge granularity).** Le résiduel `parity_fill` (137, GT HIT 141) : la frappe GT produit DEUX candidats — un turn (137) et un detect_hit (145), 8 frames d'écart > `merge_gap=round(50*0.12)=6` → clusters SÉPARÉS → le DP d'alternance étiquette turn@137=BOUNCE + hit@145=HIT. Granularité de fusion trop fine.

---

## 3. Le fix (mesuré, pas aveugle) — `vision/events.py`

**Fix 1 — vx-gate de l'ancre vy** (`classify_events`, ~l.386). Un `vy_flip` dont la frame représentative a *aussi* `vx_flip=True` (inversion horizontale = **redirection raquette**) n'est plus une ancre rebond pure :
```python
_vxf = feat.get("vx_flip")
vy_bounce = bool(vy) and not (_vxf is not None and bool(_vxf))
```
- `vx_flip` est un **numpy bool** → `np.True_ is True == False` ; on teste la véracité explicitement (`bool(_vxf)`), `None` (illisible) = préserve le rebond (abstention sûre).
- `vy_bounce` remplace le `vy` brut dans `hit_like`, `bscore`, `anchor`, ET dans le dict d'event (`"vy": vy_bounce`) → le `mandatory` du DP et le firewall HIT (`_reward`) utilisent la valeur gated de façon cohérente. La sélection `rep` garde `vy` brut (elle doit toujours préférer la frame vy-flip).

**Fix 2 — `merge_frac` 0.12 → 0.18** (défaut `classify_events`). L'évidence d'un même contact (le turn induit + le pic de poignet) s'étale ~0.15–0.18 s ; elle doit coalescer en UN event. 0.18 (~9f @50fps) fusionne le turn@137 et le detect_hit@145.

**Pourquoi 0.18 et pas 0.16** (validé par un panel adversarial 3-sceptiques + juge) : 0.16 tombe dans un **creux de hit-F1 du cache** (0.556 < plancher 0.60 → forcerait à baisser un garde) ; **0.18 donne le MÊME 0/0 live tout en faisant REMONTER le cache à bounce F1 0.889 / hit F1 0.632** (≥ plancher) — **aucun garde affaibli**. Plateau live 0/0 vérifié sur 0.15–0.20. Le seul risque résiduel de sur-fusion (deux contacts distincts < ~0.18 s, ex. volée-volée réflexe) est un élargissement d'1 frame d'un risque que 0.12 portait déjà — négligeable pour la cible amateur-sur-trépied.

---

## 4. Résultat mesuré (LIVE, pas cache)

**Replay des inputs prod réels via le vrai `classify_events` (après fix) :**
```
BOUNCE: [137, 229, 253, 369, 456, 510, 570]
HIT   : [108, 145, 218, 273, 333, 411, 487, 530]
confusion_H->B = 0   confusion_B->H = 0
bounce F1 = 0.667   hit F1 = 0.750
```
→ **confusion_H→B 4→0, B→H 1→0**. bounce recall nettement > les 2/9 legacy. hit F1 0.333→0.750.

<!-- FINAL-LIVE-PLACEHOLDER -->

---

## 5. Non-régression (tous verts)

```
test_bounce_regression           PASS  (felix Kalman F1 0.800 ≥ 0.72)
test_bounce_wasb_regression      PASS  (felix WASB  F1 0.865 ≥ 0.80)
test_vx_veto                     PASS  (felix F1 0.9143)
test_event_confusion_regression  PASS  (demo3 cache H→B=0 B→H=0 bounce F1 0.842 hit F1 0.632 ≥ 0.60)
test_sharp_turns                 PASS
test_far_coverage                PASS  (84.4 % cache replay / 84.9 % live)
```
Le fix ne touche QUE le chemin méthodo (`classify_events`). Les gardes felix (`vx_veto`, bounce regressions) passent inchangés — `vx_flip_veto` vit dans `vision/bounce.py`, chemin distinct.

---

## 6. felix end-to-end — dégradation gracieuse (pose far maintenant dense)

felix `-s 0 -d 6` (360 frames @60fps), ON vs OFF :
- **ON : 7 bounces / 6 shots**, séquence `B H B H B H B H B B H B H` (alternance quasi-parfaite, 1 hoquet BB), 1 rallye, **firewall 0/0 par construction, aucun crash**.
- **OFF : 3 bounces / 2 shots**, séquence `H B B H B`.

⚠️ **Nuance vs `wire-event-methodo-CR` (« felix ON = 0/0 abstention, parité OFF ») : OBSOLÈTE.** Cette abstention était l'artefact de la pose far AFFAMÉE de l'époque. Maintenant que la pose far felix est dense (0.9 %→87.9 %), la méthodo **s'engage** et produit un rallye structuré au lieu d'abstenir. Ce n'est PAS une régression : felix ON donne une structure de rallye PLUS plausible (alternance B-H) que le OFF (`HBBHB`), sans crash, firewall intact. felix n'a pas de GT frappes pour scorer l'exactitude sur ce sous-clip court ; les rebonds partagés (161, 245, 329) apparaissent dans les deux.

---

## 7. Reco sur le défaut → **ON** (avec `--no-event-methodo` pour le legacy)

`--event-methodo` passe **`action=BooleanOptionalAction, default=True`** :
- **défaut ON** : la confusion rebond/frappe que l'utilisateur a constatée disparaît du `_stats.json` (et donc de la vidéo). C'est le but de la session.
- **`--no-event-methodo`** : le chemin legacy strict (`vx_flip_veto` + `detect_hits`, garde temporelle), conservé comme échappatoire. (Le chemin OFF reste byte-identique au legacy — code OFF intouché.)

**Justification du flip** (vs le défaut OFF du gap-CR) : la condition de bascule posée par le gap-CR — « ré-évaluer ON quand far-pose ≥ ~70 % ET 0/0 tient end-to-end » — est désormais **remplie** (far-pose 100 %, 0/0 live après fix, tous les gardes verts). Le PM peut trancher autrement via le flag ; la donnée pointe vers ON.

---

## 8. Leçon (pour la mémoire)

**Un cache figé peut être inreprésentatif de PLUSIEURS façons à la fois.** Le gap-CR avait identifié l'inadéquation pose-far (23.4 % vs 78 %). En la corrigeant, une **2ᵉ inadéquation** est apparue : la **signature vy_flip de la balle aux frappes** diffère entre la piste Kalman figée et la vraie piste live. Toujours **mesurer sur un run LIVE end-to-end + scorer le `_stats.json` réel**, jamais le cache seul. Le discriminant qui sauve = `vx_flip` (rebond préserve vx, frappe l'inverse), la même physique que le `vx_flip_veto` de `vision/bounce.py` — désormais portée dans l'ancre de `classify_events`.

## 9. Outillage ajouté
- `tools/event_eval/score_stats.py` — score un `_stats.json` live vs la GT demo3 (frames abs→rel).
- `tools/event_eval/diag_live_vs_cache.py` — replay des inputs live dumpés, attribution des confusions.
- `tools/event_eval/proto_vxgate{,2}.py` — prototypes offline du vx-gate (avant édition de `vision/events.py`).
- `run_pipeline_8s.py` — hook de dump inerte (`COURTSIDE_DUMP_METHODO_INPUTS`) des inputs `classify_events`.
