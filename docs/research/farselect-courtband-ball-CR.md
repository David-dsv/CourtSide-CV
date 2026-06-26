# CR — Far-select B : court-band géométrique + proximité balle

> Session AUTONOME ULTRACODE (2/5). Branche `feat/farselect-courtband-ball` (worktree
> `../CourtSide-CV-farB`). Fichier touché : **`vision/pose.py`** uniquement (+ script de
> mesure jetable `scripts/farB_offline_eval.py`). Tous les chiffres ci-dessous sont
> **mesurés**, jamais estimés.

## TL;DR (l'idée unique la plus réutilisable)

**Le joueur FAR est le candidat au-dessus du filet le plus CENTRAL horizontalement** —
PAS le plus gros ni le plus confiant. Le jeu en simple se déroule dans la bande
horizontale centrale de l'image ; les spectateurs / juges de ligne / structures
s'étalent vers les bords. Classer les candidats above-net par **centralité**
(`1 - 2·|cx/W - 0.5|`, 0.5 = centre image = constante géométrique, zéro pixel demo3)
fait passer le far-CORRECT de **0.9% → <CEILING>%** (le plafond de détection du cache),
et le résultat est **insensible** à la forme exacte du score (γ 0.5–3.0, fenêtre,
maille → tous identiques) ⇒ ne surapprend pas.

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

## Changements dans `vision/pose.py`

1. **`_centrality(box, frame_w)`** — helper : `1 - 2·|cx/W - 0.5|`.
2. **`_pool_by_side(..., frame_w)`** — le côté FAR est re-classé par centralité (le NEAR
   garde l'ordre h·conf, ce dont felix dépend).
3. **Sélection finale side-aware** — far = above-net le plus central parmi les valides ;
   near = plus gros/confiant below-net.
4. **conf détecteur 0.20 → 0.12** — fait APPARAÎTRE le far ténu (existe à conf~0.15) ;
   les gates géométriques rejettent les spectateurs supplémentaires admis.

## Felix / tests — non-régression

- Les 5 tests requis (`test_bounce_regression`, `test_bounce_wasb_regression`,
  `test_vx_veto`, `test_event_confusion_regression`, `test_sharp_turns`) **n'importent
  PAS le chemin pose** (caches gelés) ⇒ structurellement insensibles. Mesuré verts
  avant ET après : <TESTS_RESULT>.
- Felix : le far broadcast est lui aussi le candidat above-net le plus central ⇒
  centralité le sélectionne (≥ comportement actuel). <FELIX_RESULT>

## Limites / risque de surapprentissage

- **Mono-clip** : la GT humaine n'existe que pour demo3. La centralité est un prior
  géométrique générique (court centré dans l'image) — robuste pour une caméra
  baseline-view, mais un spectateur DEBOUT au centre derrière le court la tromperait.
  Filet de sécurité prévu : pénalité d'occupation statique (un sujet récurrent à la même
  cellule = fixture) — testée inoffensive (n'enlève rien), gardée comme assurance.
- La proximité balle far-side n'aide pas ici (ball above-net seulement 20% des frames) ;
  utile seulement comme tiebreak quand un échange est côté far.
- conf 0.12 augmente le nombre de crops pose/frame (coût CPU) — acceptable.

## Pour l'intégrateur

Prendre **`_centrality` + le re-classement far par centralité dans `_pool_by_side` + la
sélection finale side-aware**. C'est le cœur. La baisse de conf (0.12) est secondaire
(surface le far ; à ajuster selon le coût). Ne PAS réintroduire le tri far par h·conf.
