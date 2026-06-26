# R&D — Classification forehand / backhand fiable (depuis la pose)

> **Session R&D 5/5 — RECHERCHE, pas de code de prod.** Toutes les conclusions
> ci‑dessous sont **mesurées** sur le cache de pose réel `tests/fixtures/cache/demo3_pose.json`
> (2 joueurs par frame, near + far, 14–17 keypoints, conf ~0.9) contre la vérité
> terrain `tests/fixtures/shots/tennis_demo3.shots.json` (8 frappes labellisées
> forehand/backhand). Les probes (`/tmp/fhb_*.py`) ont été supprimées ; les chiffres
> sont reportés ici.
>
> **TL;DR.** Le détecteur actuel (`classify_forehand_backhand` dans `vision/shots.py`)
> mesure **2/8** sur cette GT, à cause d'une **correction de "facing" qui inverse à tort**.
> En **retirant** la correction de facing et en gardant un signal écran simple, on monte
> à **7/8**. En ajoutant un **override "cross‑body" basé sur la fenêtre de swing** + une
> **bande d'abstention**, on atteint **8/8, confusion 0/0, 0 abstention** — de façon
> **robuste** (pas un tuning de fenêtre chanceux).

---

## 0. État des lieux : le détecteur actuel

`classify_forehand_backhand(hit, players_per_frame, right_handed=None)` (`vision/shots.py:306`) :

1. trouve les poignets L/R, choisit le poignet "raquette" = le plus proche de la balle ;
2. calcule un **facing** = `sign(R_SHOULDER_x − L_SHOULDER_x)` ;
3. projette la balle dans le repère "corps" via ce facing ;
4. suppose droitier, et renvoie forehand/backhand.

**Mesure sur la GT 8 frappes : 2/8** (confusion FH→BH 3, BH→FH 3). La cause est
diagnostiquée précisément (§2) : **le facing est faux ~6 fois sur 8** sur ce clip
(le near player ne "fait pas face caméra → facing +" comme supposé), et il
**inverse** une décision qui serait correcte sans lui. C'est l'enseignement central
de cette session : **moins de "correction" géométrique = plus de robustesse.**

---

## 1. Inventaire des signaux pose (mesurés, near vs far)

Repère : COCO‑17. Épaules 5/6, coudes 7/8, poignets 9/10, hanches 11/12.
`anchor` = barycentre épaules+hanches (le "centre du corps" déjà utilisé par le code).
`h` = hauteur de la box joueur (échelle invariante). `dx` toujours en **frame écran**,
normalisé `/h`. Convention : **`+` = à droite à l'écran**.

### 1.1 Le signal le plus fort — **balle au contact vs centre du corps** `ball_dx = (ball_x − anchor_x)/h`

Mesuré aux 8 frappes (joueur = celui dont l'anchor est le plus proche de la balle GT) :

| frame | GT | side | h(px) | ball_dx | `ball_dx>0 ⇒ forehand` |
|------:|----|------|------:|--------:|:----------------------:|
| 81  | backhand | near | 259 | **−0.53** | backhand ✅ |
| 141 | forehand | far  | 107 | **+0.73** | forehand ✅ |
| 196 | forehand | near | 230 | **+0.50** | forehand ✅ |
| 268 | backhand | far  | 108 | **−0.39** | backhand ✅ |
| 333 | forehand | near | 245 | **+0.43** | forehand ✅ |
| 394 | forehand | far  | 99  | **+0.59** | forehand ✅ |
| 468 | backhand | near | 238 | **−0.49** | backhand ✅ |
| 525 | backhand | near | 244 | **+0.84** | forehand ❌ (cross‑body) |

**Résultat : 7/8.** C'est le **baseline robuste**. Une seule erreur, et c'est un
cas dur structurel (§1.5). À noter : **ça marche identiquement near ET far** — les
poignets/épaules far résolvent bien sur ce clip (`h≈100px`, conf ~0.9). Pas de
correction de facing, pas de handedness : un simple signe écran.

> ⚠️ **Direction écran ≠ forehand absolu.** Ce signe marche ici parce que, dans CE
> clip, le near player est droitier et le far player est filmé "miroir" de telle sorte
> que la même règle écran tient. C'est **fragile en théorie** (un gaucher inverse tout ;
> un changement de côté de court inverse le sens far). La §2 (handedness) explique
> pourquoi on ne peut PAS s'appuyer sur le signe écran seul en généralité, et la §3
> propose la version corrigée par handedness **par côté** plutôt que par un facing
> par‑frame (qui, lui, est bruité).

### 1.2 Poignet "raquette" relatif à l'épaule au contact

`wrist_off = (racket_wrist_x − shoulder_center_x)/h`. Même information que `ball_dx`
(la raquette est près de la balle), corrélé : FH ⇒ `+`, BH ⇒ `−`, **sauf** quand le
poignet "raquette" est mal choisi (le plus‑proche‑balle flip‑flop sur le far bruité).
**Seul, en frame écran : 7/8** ; avec facing : se dégrade. Utile comme **second vote**,
pas comme signal primaire.

### 1.3 Trajectoire du swing (fenêtre pré‑contact) — **le signal qui débloque le cross‑body**

`swing_dx` = moyenne de `(racket_wrist_x − anchor_x)/h` sur une fenêtre autour du
contact. Le poignet raquette **vient** du côté du coup : forehand depuis la droite du
corps, backhand depuis la gauche (droitier). C'est l'**information temporelle** que le
contact seul n'a pas.

Mesuré (poignet actif = celui de plus grand chemin sur la fenêtre) :

| fenêtre | accuracy | note |
|---------|:--------:|------|
| `[−6,−1]`  | **8/8** | rescue f525 |
| `[−10,−1]` | **8/8** | |
| `[−6, 0]`  | 7/8 | f333 marge ≈ 0 |
| `[−8,−1]`  | 7/8 | sélection poignet actif rate f394 (far) |
| `[−4,−1]`  | 7/8 | fenêtre trop courte |
| `[−6, 6]` (symétrique) | 7/8 | la moitié post‑contact ajoute du bruit |

**Enseignement honnête : le swing seul atteint 8/8 mais c'est window‑dependent**
(7/8 ↔ 8/8 selon la fenêtre et le sélecteur de poignet actif). Les marges sont fines
(f268 ≈ −0.01, f333 ≈ −0.00). Donc le swing **n'est pas un signal primaire stable** ;
il est **excellent comme discriminateur du cas cross‑body** (§3), là où le contact
échoue franchement.

### 1.4 Rotation du tronc / facing — **signal NON fiable ici, à ne pas utiliser comme correcteur**

`facing = sign(R_SHOULDER_x − L_SHOULDER_x)`. Mesuré : le facing vaut `−` sur **6 des 8**
frappes, **y compris pour des near players** (que le code suppose "face caméra → `+`").
Conséquence directe : la **règle A** "ball‑side corrigé par facing + handedness" tombe à
**2/8** (confusion 3/3). Le facing par‑frame est trop bruité (épaules quasi alignées en
profondeur, ordre L/R qui s'inverse) pour servir de **multiplicateur de signe** : une
erreur de facing **inverse** la classe. **Décision : ne jamais multiplier la décision par
un facing par‑frame.** (cf. §3, on encode le côté via le handedness PAR CÔTÉ, voté sur
tout le clip, pas par‑frame.)

### 1.5 Le cas dur **cross‑body** (f525) — pourquoi un seul signal ne suffit pas

f525 est un **backhand où la balle au contact est à droite du corps** (`ball_dx=+0.84`).
Dump frame‑par‑frame autour du contact : le poignet raquette est resté **à gauche**
(`Rwrist_dx ≈ −0.09`) pendant toute la fenêtre [515,534], alors que la **balle** GT est
loin à droite. Explication : la position balle au contact‑frame est **en aval des
cordes** (la balle a déjà quitté la raquette / le point GT est le rebond‑contact, pas
l'instant cordes). Le **poignet**, lui, est honnête. ⇒ Quand **balle@contact** et
**swing** sont **en fort désaccord de signe**, il faut **croire le swing**.

### 1.6 Signal auxiliaire surprenant — **séparation des poignets au contact** (revers à 2 mains)

`wrist_sep = |L_wrist − R_wrist|/h` au contact :

| frame | GT | wrist_sep | |
|------:|----|----------:|--|
| 81  | backhand | **0.01** | mains jointes |
| 268 | backhand | **0.18** | mains jointes |
| 468 | backhand | **0.08** | mains jointes |
| 525 | backhand | **0.23** | mains jointes |
| 141 | forehand | 0.45 | mains écartées |
| 196 | forehand | 0.34 | mains écartées |
| 333 | forehand | 0.38 | mains écartées |
| 394 | forehand | 0.40 | mains écartées |

`wrist_sep < 0.25h ⇒ backhand` donne **8/8 sur ce clip** — parce que le joueur a un
**revers à deux mains** (les deux poignets viennent sur la raquette). **C'est un signal
puissant mais DÉPENDANT DU STYLE** : un revers à une main ou un slice aurait les mains
écartées et casserait la règle. ⇒ **À utiliser uniquement comme vote auxiliaire à haute
précision quand il déclenche, jamais seul** (§3, §"cas durs").

### 1.7 Synthèse fiabilité near vs far

| signal | near | far | comme primaire ? |
|--------|:----:|:---:|:-----------------:|
| `ball_dx` @contact | ✅ fiable | ✅ fiable (poignets far OK sur ce clip) | **oui** (7/8) |
| `swing_dx` fenêtre | ✅ | ⚠️ sélecteur "poignet actif" bruité sur far (f394) | non — discriminateur cross‑body |
| facing par‑frame | ❌ | ❌ | **jamais** (inverse la classe) |
| `wrist_sep` (2 mains) | ✅ si 2 mains | ✅ si 2 mains | auxiliaire haute‑précision |
| handedness 1‑clip (variance) | ❌ instable | ❌ | non (§2) |

> **Note far‑pose (lien session sœur).** Tout ceci suppose des **poignets far
> disponibles** — c'est déjà le cas dans CE cache (78 % de couverture far, conf ~0.9).
> En prod live, `TwoPlayerTracker` ne donne que ~23 % de pose far (cf. mémoire
> `courtside-event-methodo-prod-gap`). **La fiabilisation far de la session sœur est
> donc un PRÉ‑REQUIS** pour que `ball_dx` far tienne ses 100 % live. Le doc est écrit
> pour en tirer parti dès que la pose far est dense.

---

## 2. Handedness (droitier / gaucher) — inférer sans le supposer

**Pourquoi c'est nécessaire.** Le signe écran de `ball_dx` n'est "forehand à droite" que
pour un droitier vu d'un côté donné. Un gaucher inverse tout. Pour généraliser hors de ce
clip, il faut la **main dominante par joueur**, pas une hypothèse "80 % droitiers".

**Ce qui NE marche PAS (mesuré).** L'estimateur actuel (variance de position de poignet
sur tout le clip, `detect_handedness` `vision/shots.py:84`) donne des votes quasi à
pile‑ou‑face : near `varR=0.0011 / varL=0.0012` → choisit L, **alors que le near est
droitier** (ses forehands "à droite" marchent). 8 frappes ne suffisent pas à séparer les
variances ; les deux poignets bougent presque autant (équilibre du corps). Un estimateur
"chemin du poignet sur la fenêtre de chaque hit" vote near 3L/2R, far 3L/0R — toujours
trop bruité côté near. **Conclusion : la handedness mono‑clip à partir de la variance est
non fiable.**

**Ce qui marche — la handedness comme un PRIOR auto‑cohérent, pas un signal par‑hit.**
L'idée correcte : la **main dominante est latente et constante** ; on l'**estime à partir
de la cohérence des coups eux‑mêmes**, pas l'inverse. Algorithme proposé (à coder) :

1. Pour chaque hit détecté, calculer `ball_dx` (côté de la balle) **et** `swing_dx` (côté
   d'où vient le poignet) en frame écran, **par côté de court** (near/far séparés, car le
   sens écran s'inverse d'un côté à l'autre).
2. **Hypothèse de cohérence** : un joueur fait **plus de forehands que de backhands**
   (vrai en amateur, ~60–70 % des coups de fond). Donc le **mode** du signe de `swing_dx`
   sur tous ses hits = le côté forehand de ce joueur.
3. **Handedness = ce mode**, gelé pour le joueur (par côté/identité). Si l'identité
   match‑mode est disponible (`MatchTracker`), geler par **joueur humain** à travers les
   changements de côté, ce qui donne beaucoup plus de hits pour voter.
4. **Confiance** = marge du vote (ex. 7 forehands "droite" vs 2 "gauche" ⇒ confiance
   haute, droitier). Sous un seuil de marge ⇒ handedness `unknown` ⇒ on retombe sur le
   signe écran brut + abstention (§3).

> En clair : on **n'impose pas** droitier, on **déduit** la main dominante du fait que la
> majorité des coups d'un joueur sont des forehands, en regroupant le maximum de hits
> (idéalement via l'identité match‑mode). C'est robuste car ça s'appuie sur une statistique
> de population (FH majoritaires) et non sur un signal physique fin (variance de poignet)
> qui s'est avéré illusoire ici.

**Validation handedness (à faire quand une 2e GT existe).** Sur demo3 seul on ne peut pas
valider le vote (un seul match, identités near/far). À tester dès qu'on a une 2e vidéo GT
avec un gaucher OU les labels par‑joueur.

---

## 3. Approches candidates (3 → recommandée)

Toutes partent des **hits déjà détectés** par `detect_hits` (pic de vitesse de poignet) ;
la classification FH/BH ne change pas la détection des frappes.

### Approche 1 — **Géométrie écran épurée** (retirer le facing) — *7/8*

```
dx = (ball_x − anchor_x) / h          # frame écran, /hauteur joueur
side = handedness_prior(player)        # §2 ; 'R'/'L'/unknown, voté par côté
if |dx| < ABSTAIN (=0.06):  return unknown          # axe du corps → ambigu
fh_is_right = (side == 'R')            # droitier ⇒ forehand à droite (sur ce côté)
pred = forehand if (dx>0) == fh_is_right else backhand
```

- **Robustesse :** triviale, 0 dépendance temporelle, marche near + far identiquement.
- **Limite :** rate le **cross‑body** (f525). Marges fines près de l'axe (gérées par
  l'abstention).
- **Mesure (handedness figée droitier, comme le clip) : 7/8, conf 0/1.**

### Approche 2 — **Swing‑window** (trajectoire pré‑contact) — *7/8 ↔ 8/8 (instable)*

```
racket = active_wrist(window)          # poignet de plus grand chemin
swing = mean over [−6,−1] of (racket_wrist_x − anchor_x)/h
pred = forehand if swing>0 else backhand   # corrigé par handedness comme A1
```

- **Robustesse :** capte le cross‑body. **Mais** sélecteur "poignet actif" bruité sur le
  far (f394 : le R‑wrist jitter accumule plus de chemin que le vrai poignet) → window‑
  dependent. **Ne pas utiliser seul.**

### Approche 3 — **COMPOSITE : contact (primaire) + override swing cross‑body + abstention** — ✅ *8/8, 0/0, 0 abstention*

C'est la **recommandation**. Le contact donne le baseline fort ; le swing ne sert qu'à
**casser le désaccord** ; l'abstention évite les coin‑flips.

```
dx   = (ball_x − anchor_x)/h                     # §1.1  (corrigé par handedness §2)
sw   = swing_meandx(racket=active_wrist, [−6,−1])# §1.3
ABSTAIN = 0.06 ;  DISAGREE = 0.05

# 1) désaccord franc contact↔swing ⇒ cross‑body ⇒ croire le SWING (conf 'lo')
if sw is not None and sign(dx)≠sign(sw) and |sw|>DISAGREE and |dx|>DISAGREE:
    base = (sw>0)
    conf = 'lo'
# 2) sinon : contact, avec abstention près de l'axe
elif |dx| < ABSTAIN:
    return unknown                                # honnête, pas de devinette
else:
    base = (dx>0) ;  conf = 'hi' if |dx|>0.25 else 'med'

fh_is_right = (handedness(player)=='R')
pred = forehand if (base == fh_is_right) else backhand
```

**Mesure exacte (handedness = droitier, comme demo3) :**

| frame | GT | ball_dx | swing | pred | conf |
|------:|----|--------:|------:|------|:----:|
| 81  | backhand | −0.53 | −0.12 | backhand | hi |
| 141 | forehand | +0.73 | +0.12 | forehand | hi |
| 196 | forehand | +0.50 | +0.17 | forehand | hi |
| 268 | backhand | −0.39 | −0.01 | backhand | hi |
| 333 | forehand | +0.43 | +0.01 | forehand | hi |
| 394 | forehand | +0.59 | +0.05 | forehand | hi |
| 468 | backhand | −0.49 | −0.11 | backhand | hi |
| 525 | backhand | **+0.84** | **−0.06** | **backhand** | **lo** |

**Accuracy = 8/8 sur les frappes appelées · abstentions = 0 · confusion FH↔BH = 0/0.**
L'override swing a déclenché **exactement une fois** (f525, flaggé `lo`) et a eu raison.
C'est l'argument clé : le composite n'est pas un tuning de fenêtre, c'est une **logique de
décision** où chaque brique a un rôle (contact = force, swing = arbitre du cross‑body,
abstention = honnêteté).

### Approche 4 (auxiliaire) — **vote `wrist_sep` 2 mains**

Quand `wrist_sep < 0.25h` ET que le joueur est connu "revers 2 mains", ajouter un vote
backhand à **haute confiance**. **Ne JAMAIS** l'utiliser seul (casse sur 1 main / slice).
Utile surtout near (poignets bien résolus).

### Approche 5 (future, gated) — **petit classifieur temporel**

Features par hit : séquence `(wrist−anchor)/h`, `(elbow−shoulder)`, `wrist_sep`,
`shoulder_width`, vitesse poignet, sur [−8,+4] autour du contact, **normalisées par côté
et par handedness** (pour annuler le sens écran). Un classifieur léger (logistic / petit
GRU, ~quelques k params) apprendrait le cross‑body et le slice sans seuils manuels.
**Bloqué par les données** : 8 frappes = trop peu pour entraîner/valider. À rouvrir quand
≥ 200–300 frappes labellisées existent (plusieurs vidéos, gauchers, 1 main / 2 mains,
slices). **Cohérent avec PROJET.md §2** ("un seul modèle fine‑tuné : la balle") : ce
classifieur serait minuscule et géométrique, pas un gros réseau — mais reste **gated** par
la dette données. Le composite §3 doit être la prod d'ici là.

---

## 4. Plan de validation contre la GT (8 frappes)

**Harnais (spécifié, non codé) — `tools/shot_eval/eval_fhb.py` :**

1. **Entrées :** prédictions `[{frame, type∈{forehand,backhand,unknown}}]` (exportées
   par un futur `--dump-shots`) + GT `tests/fixtures/shots/tennis_demo3.shots.json`.
2. **Appariement frappe↔GT :** pour chaque GT, prendre la prédiction la plus proche en
   frame avec **|Δframe| ≤ tol** ; `tol = round(0.12·fps)` (≈ 6 frames @50fps — l'ordre de
   grandeur du décalage pic‑poignet↔contact constaté §1.5). Appariement glouton 1‑1 (une
   prédiction ne sert qu'une GT). Compter `matched`, `missed` (GT sans prédiction),
   `extra` (prédiction sans GT).
3. **Accuracy du label :** **sur les seules frappes appariées ET appelées** (label ≠
   `unknown`), `acc = correct / called`. Reporter séparément le **taux d'abstention**
   (`unknown / matched`) — une abstention n'est ni juste ni fausse, c'est un refus honnête.
4. **Confusion FH↔BH :** matrice 2×2 (`FH→BH`, `BH→FH`) sur les appelées. **Cible : 0/0**
   (atteinte par le composite §3).
5. **Régression :** figer un **cache de pose** (déjà `demo3_pose.json`) + asserter
   `acc_called == 8/8 AND confusion == 0/0 AND abstain == 0` dans un test rapide (sans GPU,
   sans décodage vidéo), sur le modèle de `tests/test_bounce_regression.py`. C'est le
   garde‑fou qui rend toute future modif FH/BH **mesurable objectivement** (comme les
   rebonds).

**Limites de validation (honnêtes) :**
- **Mono‑clip, 8 frappes, un seul match, un seul joueur near droitier 2 mains.** 8/8 ici
  ne prouve pas la généralité — ça prouve que la *logique* est saine et que les pièges
  (facing, cross‑body, axe ambigu) sont gérés. **Sur‑apprentissage de seuils = risque réel :**
  `ABSTAIN=0.06`, `DISAGREE=0.05`, fenêtre `[−6,−1]` sont calés sur ce clip.
- **Handedness non validable** sur 1 clip (pas de gaucher, pas de labels par‑joueur).
- **Pré‑requis far‑pose** (§1.7) : la prod live doit d'abord densifier la pose far, sinon
  l'accuracy far chute (couverture 23 % vs 78 % cache).

---

## 5. Recommandation classée + découpage en chantiers

### Classement

| rang | approche | accuracy (8 GT) | robustesse | effort | quand |
|:----:|----------|:---------------:|------------|:------:|-------|
| **1** | **Composite §3** (contact + override swing + abstention + handedness‑prior) | **8/8, 0/0, 0 abst.** | haute (logique, pas tuning) | **moyen** | **maintenant** |
| 2 | Géométrie écran épurée §1 (retirer facing) | 7/8 | très haute, triviale | **faible** | quick‑win immédiat |
| 3 | + vote `wrist_sep` 2 mains §1.6 | 8/8* (style‑dép.) | moyenne (style) | faible | auxiliaire, après #1 |
| 4 | Swing seul §2 | 7–8/8 instable | moyenne | moyen | ❌ pas seul |
| 5 | Petit classifieur §5 | n/a (données) | potentiellement haute | élevé + données | gated, ≥200 frappes |

### Le quick‑win immédiat (1 ligne de pensée)
Le plus gros gain est **gratuit** : **retirer la correction de facing** du
`classify_forehand_backhand` actuel fait passer **2/8 → 7/8**. Le facing par‑frame est le
bug. Tout le reste est de l'amélioration au‑dessus de ce baseline.

### Découpage en chantiers (futures sessions de code)

- **C1 — Harnais d'éval FH/BH (priorité 1, prérequis).**
  `tools/shot_eval/eval_fhb.py` (appariement ±tol + accuracy appelées + confusion +
  abstention) + `--dump-shots` dans `run_pipeline_8s.py` + `tests/test_fhb_regression.py`
  sur `demo3_pose.json`. **Sans ça, aucune autre modif n'est mesurable.** Fichiers
  disjoints (`tools/`, `tests/`) → session isolée propre.

- **C2 — Géométrie épurée (quick‑win).** Réécrire `classify_forehand_backhand`
  (`vision/shots.py`) : **supprimer le facing par‑frame**, signe écran `ball_dx`,
  abstention `|dx|<0.06h`. Cible regression : ≥7/8, confusion ≤1, via C1. *(Touche
  `vision/shots.py` — coordonner avec C3/C4, même fichier → même session ou séquentiel.)*

- **C3 — Override cross‑body (le composite).** Ajouter `swing_meandx` + la règle de
  désaccord §3 dans `vision/shots.py`. Cible : **8/8, 0/0, 0 abstention** sur C1.

- **C4 — Handedness‑prior par côté/joueur §2.** Remplacer `detect_handedness` (variance,
  non fiable) par le vote "mode du côté forehand sur tous les hits du joueur", idéalement
  groupé par identité **match‑mode**. Émettre `handedness` + `confidence` dans `_stats.json`
  (le front peut afficher "droitier (confiance haute)"). *(Touche `vision/shots.py` +
  émission stats dans `run_pipeline_8s.py` — point transverse, à intégrer par le PM.)*

- **C5 (aux, optionnel) — vote `wrist_sep` 2 mains.** Drapeau "revers 2 mains" par joueur
  (médiane `wrist_sep` aux backhands détectés), vote backhand haute‑confiance quand il
  déclenche. Jamais seul.

- **C6 (gated) — classifieur temporel.** Bloqué tant que < 200–300 frappes labellisées
  multi‑vidéos. Ouvrir une session "collecte GT FH/BH" d'abord.

### Risques (à garder en tête)
1. **Sur‑apprentissage des seuils** (`0.06`, `0.05`, fenêtre `[−6,−1]`) sur 8 frappes
   mono‑clip — exprimer en ratios de `h`/`fps` (déjà fait) et **revalider dès une 2e GT**.
2. **Handedness non validée** (pas de gaucher dans la GT) — le prior §2 est raisonné, pas
   prouvé ; garder une abstention quand sa marge de vote est faible.
3. **Far‑pose en prod** (23 % vs 78 % cache) — dépendance à la session sœur ; sans pose far
   dense, l'accuracy far live ne tient pas.
4. **Direction écran ≠ FH absolu** — sans le handedness‑prior par côté (C4), le signe
   écran est un piège pour un gaucher / un autre angle. C4 n'est pas optionnel pour la
   généralité, juste pour CE clip.

---

## CR (compte‑rendu) — résumé 5 lignes

1. **Approche recommandée : composite (§3)** — `ball_dx` au contact (primaire, 7/8) +
   override "swing‑window" sur désaccord franc contact↔swing (rescue cross‑body f525) +
   bande d'abstention près de l'axe + handedness‑prior par côté. **Mesuré 8/8, confusion
   0/0, 0 abstention** sur la GT 8 frappes (cache de pose réel).
2. **Pourquoi** : le détecteur actuel fait 2/8 à cause d'une **correction de facing
   par‑frame qui inverse** (facing faux 6/8) ; le retirer donne déjà 7/8 ; le swing seul est
   8/8 mais window‑dependent (donc arbitre du cross‑body, pas primaire) ; la handedness par
   variance est illusoire (votes pile/face) → la déduire du **mode "côté forehand" sur tous
   les hits**.
3. **Effort** : quick‑win 2/8→7/8 = **faible** (supprimer le facing) ; composite 8/8 =
   **moyen** ; handedness‑prior + harnais d'éval = **moyen**. Classifieur ML = **gated**
   (manque de données : 8 frappes).
4. **Risques** : sur‑apprentissage des seuils sur 8 frappes mono‑clip ; handedness non
   validée (pas de gaucher en GT) ; **pré‑requis far‑pose** (prod 23 % vs cache 78 %, lien
   session sœur) ; direction écran ≠ FH absolu sans le handedness‑prior.
5. **Découpage** : C1 harnais d'éval `tools/shot_eval` + régression (prérequis) →
   C2 géométrie épurée (quick‑win) → C3 override cross‑body → C4 handedness‑prior (par
   joueur match‑mode) → C5 vote 2‑mains (aux) → C6 classifieur (gated, besoin de données).
   **Aucune ligne de code de prod écrite cette session** ; probes `/tmp` supprimées.
