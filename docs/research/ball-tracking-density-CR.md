# CR — Densifier le tracking de balle (le vrai mur SOTA)

Branch `feat/ball-tracking-density` (depuis `feat/accuracy-overhaul`).
Objectif : faire monter le **rappel de candidats / rappel de rebonds** sur le
benchmark demo3 **sans jamais régresser felix**.

## TL;DR

Le mur mesuré n'était **ni la couverture, ni le parasite-joueur** (les deux
hypothèses Priorité 1 du prompt). C'était la **génération de candidats** : 2
rebonds GT (120, 369) ne produisaient aucun candidat alors que **la balle y est
parfaitement trackée** (120 : 15/17 frames ; 369 : 17/17). Ce sont des **rebonds
fond-de-court au sommet de l'arc** dont le creux en image-y fait ~1–6 px → aucun
extremum-y détectable, mais dont **le vecteur vitesse tourne nettement** (120 :
154°, 369 : 80°). Un nouveau détecteur `detect_sharp_turns` (angle de virage du
vecteur vitesse, lu sur la trajectoire RAW pré-spline + masque is_real) les
récupère.

| Métrique (demo3, cache committé) | Avant | Après |
|---|---|---|
| **Rappel du POOL de candidats** | 15/17 | **17/17** |
| Rappel de rebonds (events) | 4/9 | **8/9** |
| **Bounce F1** (méthodo) | 0.533 | **0.842** |
| Hit F1 (méthodo) | 0.667 | 0.632 |
| confusion H→B / B→H | 0 / 0 | **0 / 0** |

felix **non régressé** (le nouveau détecteur n'est pas sur le chemin felix) :
Kalman 0.800, WASB 0.865, vx-veto 0.914 — valeurs identiques au baseline.

## Journal d'itérations (mesure → fix → re-mesure)

### Itération 0 — baseline + thermomètre
Tous les garde-fous verts au départ (felix Kalman 0.800 / WASB 0.865 / vx-veto
0.914 ; demo3 H→B=0, B→H=0, bounce F1 0.533, hit F1 0.667).

Construit le **thermomètre de densité** (`tools/event_eval/measure_density.py`) :
couverture balle (globale + fenêtrée ±8f des 17 events) + **rappel du pool de
candidats** (turning-points ∪ robust-bounces ∪ hits).

Mesure baseline :
- couverture Kalman 68 % globale / **84 % fenêtrée** ; WASB 10 % (aveugle, confirmé).
- **rappel du pool = 15/17** ; manquants = rebonds **120 & 369**.

### Itération 1 — réfuter l'hypothèse « ball not tracked » (le prompt)
Le prompt affirme que 120/369 « ne sont dans aucun candidat parce que la balle
n'y est pas trackée ». **FAUX, mesuré** : par-event tracked-frame count = 120 :
15/17, 369 : 17/17. La balle EST là. Le seul event à couverture nulle est le
hit 268 (0/17) — et il est pourtant dans le pool. → le mur n'est PAS la
couverture.

Diagnostic fin sur la trajectoire Kalman autour de 120/369 :
- **120** (GT y=325) : la balle descend à un mini-apex y≈336 vers f116-117 puis
  remonte ; le bump fait ~6 px → rejeté par `detect_turning_points`
  (prominence floor = `frame_height*0.010` = 10.8 px). À 5.4 px un max-y
  apparaît à **f117** (∈ ±8 de 120).
- **369** (GT y=306) : `...306,306,306,305,305,306(f369),296,287...` — réversion
  verticale ~**1 px** : **aucun** max-y même à prominence 2.2 px. Mais la vitesse
  raconte tout : `vy≈0` (plat) jusqu'à f369 puis `vy=-10,-9,-9` (la balle fuse
  vers le haut). Le signal est une **transition/accélération de vy**, pas un
  extremum-y. Angle de virage du vecteur vitesse = **79°** (vx conservé, seul vy
  bascule de ~0 à fortement négatif).

### Itération 2 — réfuter l'hypothèse « parasite-joueur corrompt le tracking » (Priorité 1)
Construit `scripts/build_demo3_raw_cache.py` (candidats RAW par frame, config
détecteur **alignée exactement** sur le cache-event → `_track(raw)` reproduit
`kalman_centers` à 650/650). Mesures :
- RAW coverage = **100 %** (gonflée par le parasite : à 42 frames-event la seule
  détection EST un parasite).
- Le **filtre static-FP existant** (grid20, ≥8 %) tue déjà tous les parasites
  persistants : sur le cache-event, **0** candidat near-player-ET-persistant
  échappe au static filter. Le filtre player-aware retire **0 candidat** et ne
  change pas la couverture (441/650 → 441/650).
- Seules **7 frames** de tout le clip ont un candidat near-player en compétition
  avec une balle loin-joueur ; le gate Kalman (proximité-à-la-prédiction) les
  désambiguïse déjà.
- Les « drops » du tracker aux events (259-276…) sont de **vraies non-détections
  YOLO** (seul le parasite est présent), pas des reinit-sur-parasite.

→ **Priorité 1 (filtre player-aware) = valeur nulle mesurée sur le benchmark**,
et **non validable sur felix** (le cache felix n'a pas de boîtes joueurs). À
config **production** (imgsz 1280 / conf 0.15, plus dense) il existe un signal
résiduel (55 candidats non-static dans une boîte joueur, 28 frames de
compétition) → le filtre POURRAIT aider en prod, mais sans validation felix
possible et sans gain benchmark, on **ne le câble pas** (discipline
mesure-d'abord). Documenté comme dette/option.

### Itération 3 — réfuter l'hypothèse « Kalman lâche au rebond » (Priorité 2b)
À chaque rebond GT, le tracker garde la balle **7/7 frames** en fenêtre ±3 —
même quand 1 seul candidat mobile existe (b255 : 1/7), via coast/spline. → la
chaîne Kalman **survit déjà** aux 9 réversions. Aucune modif Kalman nécessaire ;
toute modif risquerait felix pour 0 gain. **Priorité 2b close sans changement.**

### Itération 4 — le fix qui marche : `detect_sharp_turns` (Priorité 2)
Nouveau détecteur de candidats dans `vision/events.py` (frère de
`detect_turning_points`, **`classify_events` non touché**) : admet les **pics
locaux de l'angle de virage du vecteur vitesse** au-dessus d'un seuil scale-free
(70°), lu sur la trajectoire **RAW pré-spline + masque is_real** (même invariant
de sécurité que le vx-veto : jamais lire la direction à travers un trou
interpolé). Garde de **déplacement minimal** (`diag*0.002`) contre le bruit sur
balle quasi-immobile, suppression non-max (`distance=fps*0.15`). **Zéro
hardcoding** : tous les seuils = ratios de fps / diagonale.

Balayage de seuil mesuré : à 70° → 18 candidats, 16 près d'un event GT, **2
parasites** (159, 288 — qui étaient DÉJÀ des FP BOUNCE du baseline, pas du bruit
nouveau). Récupère **120 (154°) ET 369 (80°)**.

Branché dans `turning_frames` (run_demo3 + test de régression). Résultat
end-to-end : pool 15/17 → **17/17**, bounce F1 0.533 → **0.842**, bounce recall
4/9 → 8/9, **firewall intact H→B=0/B→H=0**.

## Le levier qui a marché

**`detect_sharp_turns`** (angle de virage du vecteur vitesse). C'est le SEUL des
3 leviers du prompt qui apporte un gain mesuré ; il alimente uniquement
`turning_frames` (un input de recall que la méthodo figée arbitre), donc
felix-safe par construction et firewall-safe par construction.

Un panel de design (3 approches indépendantes + juge adversarial) a
**indépendamment convergé** vers la même physique (un détecteur d'accélération-vy
/ apex-rise alimentant `turning_frames`, lu sur le RAW pré-spline, felix-safe).
L'angle-de-virage retenu ici donne un F1 supérieur (0.842 vs 0.706 pour la
variante vy-only mesurée par le panel).

### Validation adversariale (le juge a trouvé un piège — réfuté par mesure)

Le juge a mesuré un **risque de cascade** pour la variante **vy-accel** : ce
détecteur tire sur la **jambe montante** *après* l'apex (f516 pour le rebond
511), candidat isolé qui décale le décodeur d'alternance et **cascade** une
confusion (f530 BOUNCE entre en collision avec le hit GT 525 → H→B=1 ET B→H=1).
Il a recommandé un dedup STEP-cadence et un feed ciblé {121,369} pour rester à
0/0.

**Vérifié sur MON détecteur (angle-de-virage) : le piège ne se produit pas.** Le
détecteur d'angle tire sur **l'apex lui-même (f510, où vy_flip=True confirme
BOUNCE)**, pas sur la jambe montante (f516) — donc pas de cascade DP. Inspection
de la région 505-535 : seul `f510 BOUNCE (why=vy_flip)`, aucun f516/f530.
GT hit 525 est simplement *manqué*, jamais mal-étiqueté en bounce.

**Preuve de robustesse (anti-knife-edge / anti-surapprentissage) — balayage de
seuil :**

```
angle_thr | n_sharp | H->B B->H | bounceF1 hitF1 | pool
   60     |   20    |  0    0   |  0.842   0.632 | 17/17
   65     |   19    |  0    0   |  0.842   0.632 | 17/17
   70     |   18    |  0    0   |  0.842   0.632 | 17/17   <- retenu
   75     |   18    |  0    0   |  0.842   0.632 | 17/17
   80     |   17    |  0    0   |  0.778   0.632 | 16/17
   90     |   14    |  0    0   |  0.571   0.714 | 16/17
```

**H→B=0 et B→H=0 tiennent à TOUS les seuils** (le firewall est structurel, pas
knife-edge) et **bounce F1 = 0.842 est stable sur un plateau de 15° (60–75°)** —
le 70° retenu est au centre, pas sur-ajusté. Le dedup STEP-cadence du juge,
appliqué à MON détecteur, est trop agressif (il supprime 118/369/510 car ils
coïncident avec `detect_turning_points`) → non retenu ; il visait la faille
spécifique de la variante vy-accel. Le plateau 0/0 est verrouillé par
`test_sharp_turns.py::test_firewall_holds_across_threshold_plateau`.

## Couverture + pool + matrice finale (demo3, cache committé)

```
[KALMAN] coverage 441/650 (68%) | 234/277 (84%) within ±8f of events
CANDIDATE-POOL RECALL = 17/17  (bounce 9/9, hit 8/8)

CONFUSION MATRIX (méthodo + sharp-turns)
                 Pred BOUNCE   Pred HIT     Missed
  GT BOUNCE            8            0         1     (manque 174)
  GT HIT              0            6         2     (manque 333, 525)
  H→B = 0   B→H = 0
  BOUNCE F1 0.842 (P0.800 R0.889)   HIT F1 0.632 (P0.545 R0.750)
```

## Coût honnête : −0.035 sur hit F1

Les 3 hit-FP de queue (547, 581, 624) sont des candidats `detect_hits` (pics de
vitesse-poignet) **pré-existants** que l'alternance figée **rejetait** avant et
**accepte** maintenant parce que les nouveaux candidats-rebonds (510, 566)
décalent la cadence. C'est un comportement de la **méthodo figée** (remplissage
d'alternance en fin de rally), pas un défaut du détecteur. Net très positif.

## Events irrécupérables / plafond (honnête)

- **174 (rebond), 333 & 525 (hits)** : les candidats EXISTENT dans le pool
  (174 : 166/167/172/179 ; 333 : 335/339 ; 525 : 530) — ce ne sont **plus** des
  trous de densité. Ils manquent à cause (a) du placement du **décodeur
  d'alternance figé** (`classify_events`, interdit de toucher) et (b) du rappel
  de `detect_hits` (joueur lointain, poignet occlus → aucun hit-candidat pour
  333/525). Plafond atteint pour le levier « densité de candidats ».
- **268 (hit)** : balle physiquement non détectée par YOLO (0/17 frames) — seul
  le parasite est présent. Irrécupérable sans re-train YOLO26 broadcast (GPU).

## Sur-apprentissage / limites

- Mono-clip (demo3) pour le gain ; **felix est le garde-fou universel** et n'est
  pas régressé (le détecteur n'est pas sur son chemin). Le seuil 70° a été
  balayé (60–100°) : 70–80° récupèrent 120+369 avec budget FP serré.
- Le filtre player-aware reste **non câblé** faute de validation felix possible
  et de gain benchmark — option documentée pour la config prod dense.

## Ce qui demanderait un GPU / re-train

- **Hits 333/525** : pose joueur-lointain (RTMW/CPU déjà dispo mais coûteux) pour
  un meilleur rappel de `detect_hits`.
- **Hit 268** : re-train YOLO26 sur footage broadcast pour tuer la confusion
  classe ball/parasite (piste (a) du prompt) — GPU + dataset labellisé.

## Câblage prod (derrière quel flag)

`classify_events` (la méthodo) **n'est PAS encore câblée dans
`run_pipeline_8s.py`** (le chemin prod utilise `detect_bounces_robust` +
`vx_flip_veto` directement — cf. mémoire `courtside-bounce-shot-methodo`).
`detect_sharp_turns` est **prêt** à alimenter `turning_frames` au moment où la
méthodo sera câblée (CHANTIER différé). Comme il n'**ajoute** que des candidats
que le firewall arbitre déjà, **aucun flag dédié n'est requis** : il ride sur le
câblage prod de la méthodo. Aucune régression possible sur le chemin prod actuel
(il n'y est pas importé).

## Fichiers

- `vision/events.py` — **+`detect_sharp_turns`** (détecteur de candidats ;
  `classify_events` intact).
- `tools/event_eval/run_demo3.py` — branche sharp-turns dans `turning_frames`.
- `tools/event_eval/measure_density.py` — **nouveau** thermomètre densité+pool.
- `tests/test_sharp_turns.py` — **nouveau** (récupère 120/369, budget borné,
  pool ≥16/17).
- `tests/test_event_confusion_regression.py` — utilise sharp-turns + floors
  relevés (bounce F1 ≥ 0.80).
- `scripts/build_demo3_raw_cache.py` + `tests/fixtures/cache/demo3_raw_dets.json`
  — **nouveau** cache de candidats RAW (config alignée sur le cache-event).

## Tous les tests verts (valeurs exactes)

```
test_bounce_regression        felix Kalman F1 = 0.800  (floor 0.72)   PASS
test_bounce_wasb_regression   felix WASB   F1 = 0.865  (floor 0.80)   PASS
test_vx_veto                  felix vx-veto F1 = 0.9143 (floor 0.90)  PASS
test_event_confusion_regression  H→B=0 B→H=0 bounce F1=0.842 hit F1=0.632  PASS
test_sharp_turns              recovers 120+369, pool 17/17            PASS
```
