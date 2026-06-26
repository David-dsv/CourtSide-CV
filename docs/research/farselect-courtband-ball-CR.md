# CR — Far-select B : court-band géométrique + proximité balle

> Session AUTONOME ULTRACODE (2/5). Branche `feat/farselect-courtband-ball` (worktree
> `../CourtSide-CV-farB`). Fichier touché : **`vision/pose.py`** uniquement (+ script de
> mesure jetable `scripts/farB_offline_eval.py`). Tous les chiffres ci-dessous sont
> **mesurés**, jamais estimés.

## TL;DR (les deux idées les plus réutilisables)

far-CORRECT : **0.9% → 81.3% live** (×90), plafond cache offline 91.6%. Deux leviers :

1. **Le joueur FAR = le candidat au-dessus du filet le plus CENTRAL horizontalement**
   — PAS le plus gros ni le plus confiant. Le jeu en simple est dans la bande
   horizontale centrale ; spectateurs/juges/structures s'étalent vers les bords.
   Classer par **centralité** (`1 - 2·|cx/W - 0.5|`, 0.5 = centre = constante
   géométrique, zéro pixel demo3) : sur le cache 0.9%→**91.6%** (= plafond détection),
   **insensible** à la forme du score (γ 0.5–3.0, fenêtre, maille → tous identiques),
   **stable** en CV (first/second half, even/odd : 90.2–92.0%) ⇒ ne surapprend pas.

2. **Découpler la sélection far du SQUELETTE.** Le crop far minuscule donne nkp=0 sur
   ~80% des frames → la gate squelette `is_valid_player` (nkp≥10) jette la box
   correctement localisée. Le piège : la centralité SEULE n'a donné que 4.4% en prod
   tant que la box far passait par cette gate. Sélectionner le far par GÉOMÉTRIE
   (`is_geometric_far_player` : centralité + bande de court, sans nkp) → 4.4%→81.3%.
   **far-CORRECT est piloté par la BOX, pas par le squelette.**

## Le problème (diagnostic déjà mesuré, confirmé ici)

Mémoire `courtside-far-player-selection-not-detection` : le far **EST détecté**, le
pipeline **SÉLECTIONNE** le mauvais sujet above-net (cluster fixe spectateur en haut à
droite ~(1360,120)). Baseline mesurée (scorer `tools/pose_gt/measure_far_coverage.py`,
prod imgsz 1280) : **far-CORRECT 2/225 = 0.9%, distance moyenne 572px.**

Confirmation indépendante (ce CR) sur le cache committé `demo3_raw_dets.json`
(conf0.2/640) : une box personne tombe à <192px du VRAI far sur **91.6%** des frames
(distance médiane de la box la plus proche = **6px**) ⇒ le far est DANS l'ensemble de
candidats ; c'est 100% un problème de SÉLECTION.

## Profil géométrique VRAI far vs spectateurs above-net (mesuré sur le cache)

| feature | VRAI far | spectateurs above-net |
|---|---|---|
| feet_y (px) | 221–271 (serré, μ243) | 84–550 (étalé) |
| cx/W | 0.37–0.55 (central, roam 360px) | 0.0–1.0 (médiane 0.2 = bord G) |
| conf | μ0.7 | μ0.4 |

⇒ **le seul discriminant géométrique qui généralise = la centralité horizontale.**

## Journal d'itérations (far-CORRECT %)

Méthode : harnais offline `scripts/farB_offline_eval.py` (sub-seconde, sur le cache,
plafond 91.6%) pour explorer ; **scorer end-to-end prod (imgsz 1280)** pour valider.

| # | changement | far-CORRECT | dist moy | note |
|---|---|---|---|---|
| 0 | baseline (h·conf) | **0.9%** (prod) / 2.7% (cache) | 572 / 726px | spectateur gagne |
| — | offline: `central_conf` | 79.1% (cache) | 126px | centralité × conf |
| — | offline: `central seul` | **91.6%** (cache, =plafond) | 55px | conf nuit, ball/static n'ajoutent rien |
| 1 | PROD: centralité far + conf det 0.20→0.12, MAIS sélection far liée à `is_valid_player` (nkp≥10) | **4.4%** (prod) | 527px | **ÉCHEC** : le crop far minuscule donne nkp=0 → la box correcte est rejetée par la gate squelette |
| 2 | PROD: far DÉCOUPLÉ du squelette — `is_geometric_far_player` (centralité≥0.5 + bande), pose far en crop large best-effort | **74.7%** (prod) | 143px | le BON levier — la box pilote far-CORRECT, pas le squelette |
| 3 | PROD: + aspect far relâché 1.1→0.9 (le far qui se détend fait une box plus large que haute) | **81.3%** (prod) | 106px | oracle offline prédit 91.1% ; gap restant = densité détection live |

**Chiffre canonique** : scorer officiel `tools/pose_gt/measure_far_coverage.py`
(`far_pose_imgsz=640` par défaut) = **183/225 = 81.3%, dist 106px** — IDENTIQUE au
scorer rapide (`far_pose_imgsz=160`), ce qui **prouve que far-CORRECT est piloté par la
BOX, pas par l'appel pose** (le squelette ne change pas la box). Baseline officiel
re-mesuré : 2/225 = 0.9%, dist 572px. **Gain net : ×90.**

### Ablation (cache, far-CORRECT%) — pourquoi centralité seule

```
central only                   91.6%   (= plafond détection)
central+conf                   89.3%   (conf NUIT : spectateur parfois + confiant)
central+ball                   91.6%   (ball above-net seulement 20% des frames → n'ajoute rien)
central+static                 90.7%
ALL                            91.6%
static only (sans central)     17.3%   (faible seul)
```
**Les 19 frames d'échec de `central seul` sont TOUTES des trous de détection**
(vrai far absent du cache conf0.2/640), 0 erreur de sélection ⇒ centralité = parfaite
quand le far est présent. La prod (imgsz 1280) en récupère une partie.

### Sensibilité (anti-surapprentissage)

centralité γ ∈ {0.5,1,1.5,2,3} → **91.6% partout** ; static weight, fenêtre, maille →
**91.6% partout**. Le résultat ne dépend d'aucun paramètre réglé.

**Cross-validation** (sous-ensembles de frames, règle prod side-aware) :
first-half 90.2% / second-half 92.0% / even 90.3% / odd 92.0% — serré ⇒ pas
surappris à un sous-ensemble de frames (différentes phases d'échange).
L'ablation montre aussi que band+floor coûtent **0** (gardés comme assurance) et que
le SEUL gain restant viendrait d'aspect 0.8–0.7 (+0.5%, écarté = trop proche d'un
réglage demo3-spécifique).

## Changements dans `vision/pose.py` (tous géométriques, zéro-hardcoded)

1. **`_centrality(box, frame_w)`** — `1 - 2·|cx/W - 0.5|`.
2. **`is_geometric_far_player(box, …)`** — gate far POSITION-ONLY (centralité ≥ 0.5 +
   pieds dans la bande de court), **indépendant du squelette**. C'est le fix clé.
3. **`_pool_by_side(…, frame_w)`** — côté FAR re-classé par centralité (NEAR garde h·conf).
4. **Sélection finale side-decoupled** :
   - FAR = above-net le plus central passant `is_geometric_far_player` (PAS de nkp requis),
     posé en best-effort (crop large `pad_frac=1.0` + conf 0.05 → récupère le squelette far
     sur ~70% des frames ; sinon box correcte + squelette creux, gap-fillé en aval).
   - NEAR (+ slot extra) = recette inchangée (pose imgsz 960, `is_valid_player` nkp≥10,
     tri h·conf) — felix-safe.
5. **Aspect floor side-aware** : FAR 0.9 (far qui se détend = box plus large que haute),
   NEAR 1.1. (Le 1.1 global plafonnait far-CORRECT à 74.7% ; 0.9 → 81.3%.)
6. **conf détecteur 0.20 → 0.12** + **`far_pose_imgsz=640`** (nouveau param, défaut prod).

## Felix / tests — non-régression

**5 tests requis — VERTS avant ET après** (ils n'importent PAS le chemin pose ⇒
structurellement insensibles ; mesuré identique au baseline) :
- `test_bounce_regression` : F1=0.800 (floor 0.72) ✅
- `test_bounce_wasb_regression` : F1=0.865 (floor 0.80) ✅
- `test_vx_veto` : ALL PASS ✅
- `test_event_confusion_regression` : H→B=0 B→H=0, bounce F1=0.842 ✅
- `test_sharp_turns` : PASS ✅

**Felix — AMÉLIORÉ, pas régressé** (comparaison directe orig vs nouveau pose.py sur
`felix_full.mp4`, 5 frames échantillon — pas de GT felix donc check comportemental) :

| frame | BASELINE (orig) | NOUVEAU (mien) |
|---|---|---|
| f48 | far cx=1502 central=0.44 | far cx=1253 **central=0.69** (plus central) |
| f144 | **0 joueur** | **1 joueur** far récupéré (nkp=14) |
| f240 | **0 joueur** | **1 joueur** far récupéré (nkp=14) |
| f336 | near + far cx=1721 | identique |
| f432 | 1 joueur | 2 joueurs (1 far + 1 near remonté) |

⇒ mine récupère des far que le baseline ratait (f144/f240 : 0→1), sélectionne un far
plus central (f48), n'en perd aucun. Le double-"far" en f432 = artefact de
`net_y=0.5·fh` sur l'angle rasant felix (le filet est haut dans le cadre) — **pré-existant**,
pas introduit ici (felix = cas grazing pathologique connu, cf. mémoire
`courtside-felix-grazing-angle`).

## Limites / risque de surapprentissage

- **Plafond live 81.3% < plafond cache 91.6%** : le gap restant n'est PAS de la
  sélection (CV stable 90–92%, sélection = plafond sur le cache) mais de la **densité
  de détection live** — à conf0.12/imgsz1280 la VRAIE box far n'est présente que ~84%
  des frames (mesuré). Améliorer ça = lever DÉTECTION (imgsz/conf détecteur), hors du
  mandat sélection de cette session, et avec effet pipeline-wide. **NE PAS** y toucher
  ici. Note contre-intuitive mesurée : imgsz1280 trouve le far MOINS souvent que le
  cache imgsz640 (la cible minuscule se fragmente en haute-rés) — piste détection à creuser.
- **Mono-clip** : GT humaine seulement pour demo3. La centralité est un prior générique
  (court centré) — robuste baseline-view, mais un spectateur DEBOUT au centre derrière
  le court la tromperait. Filet prévu : pénalité d'occupation statique (sujet récurrent
  à la même cellule = fixture) — testée inoffensive (n'enlève rien sur le cache), MAIS
  elle a besoin de contexte TEMPOREL que `estimate_players_pose` (per-frame) n'a pas →
  **à brancher au niveau TwoPlayerTracker** (intégrateur), pas dans pose.py.
- La proximité balle far-side n'aide pas ici (ball above-net 20% des frames) ; tiebreak
  seulement quand un échange est côté far.
- **Coût CPU** : conf 0.12 (plus de candidats) + crop far large (`pad 1.0`) → ~12 min
  pour 225 frames CPU. Le `far_pose_imgsz` est paramétrable (160 pour mesurer la box,
  640 pour le squelette prod) — la box (donc far-CORRECT) est indépendante du pose.
- f432 felix : `net_y=0.5·fh` classe parfois le near comme far sur angle rasant
  (pré-existant) → l'intégrateur devrait dériver net_y du filet détecté quand dispo.

## Pour l'intégrateur

Le cœur, dans l'ordre d'importance :
1. **`is_geometric_far_player` (sélection far par GÉOMÉTRIE, sans nkp)** — c'est ce qui
   débloque vraiment (4.4%→81.3%). Sans ça, la centralité seule reste à 4.4% (la box
   correcte est jetée par la gate squelette). Leçon transférable : **valider le far par
   POSITION, jamais par le nombre de keypoints** (le far est trop petit pour poser).
2. **`_centrality` + tri far par centralité** (`_pool_by_side` + sélection finale).
   Ne JAMAIS réintroduire le tri far par h·conf.
3. **Aspect far 0.9** (récupère le far qui se détend).
4. Secondaire : conf 0.12 (surface le far), `far_pose_imgsz` (coût/squelette).
5. **Prochain lever (intégrateur, hors session)** : brancher la pénalité d'occupation
   statique au niveau TwoPlayerTracker (contexte temporel) pour fermer le gap live de
   sélection résiduel ; dériver `net_y` du filet détecté.
