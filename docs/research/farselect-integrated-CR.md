# CR — Far-select INTÉGRATEUR : `hcv` cross-geometry selector + B's skeleton recovery

> Session AUTONOME ULTRACODE (3/5, INTÉGRATEUR). Branche `feat/farselect-integrated`
> (worktree `../CourtSide-CV-farINT`, depuis `feat/accuracy-overhaul`). Fichier
> produit livrable : **`vision/pose.py`** + **`tests/test_far_coverage.py`** (lock) +
> `tools/pose_gt/build_far_select_cache.py` (fixture builder). **Tous les chiffres
> ci-dessous sont MESURÉS sur la prod (`estimate_players_pose`), jamais estimés.**

## TL;DR

J'ai lu les DEUX CR finaux et combiné le meilleur de chacun en UN fix intégré :

| approche | demo3 far-CORRECT | felix far-CORRECT | squelette far |
|---|---|---|---|
| baseline (avant) | **0.9%** (2/225, 572px) | 4.5%* | creux |
| **B** (courtband-ball) | 81.3% | non testé (pas de GT felix) | wide-crop (dense) |
| **A** (conf-motion, `hcv`) | 84.9% | 87.9% | crop uniforme (creux sur le far minuscule) |
| **INTÉGRÉ (ce CR)** | **84.9%** (191/225, 121px) | **87.9%** (58/66, 91px) | **wide-crop (dense)** |

\* 4.5% = score centralité-pur (commit `c50b00b`, style B) tel que mesuré par A sur felix.

**L'intégré domine STRICTEMENT les deux** : il égale le meilleur far-CORRECT (A, 84.9% /
87.9%) — donc **bat B sur demo3 (81.3%→84.9%)** ET sur felix (le style centralité de B
s'effondre à 4.5% sur l'angle de coin) — **tout en ajoutant la récupération de
squelette far dense de B** (que A n'avait pas). C'est le sur-ensemble propre.

## Comment j'ai combiné (et pourquoi PAS ma 1ère idée)

Mon brouillon initial (centralité + la gate dure `is_geometric_far_player` de B)
**aurait régressé felix à ~4.5%** : le CR FINAL de A le prouve — les deux clips ont une
géométrie far OPPOSÉE (demo3 = central+minuscule+ne pose pas ; felix = décentré
xr~0.76 + plus gros + pose), donc **tout score spécifique à une géométrie surapprend
une seule scène**. La centralité pure gagne demo3 (84.9%) mais casse felix (4.5%, elle
choisit une structure centrale) ; une gate keypoint dure gagne felix mais casse demo3
(26.7%). J'ai donc jeté le brouillon et reconstruit autour du **`hcv` de A**, qui est
le seul score validé sur les DEUX clips.

### Ce que j'ai pris de A (le cœur porteur)

Le **score unifié `hcv`** sur les candidats above-net on-court (`vision/pose.py`,
`_hcv_far_score`), avec `hmax` = la plus grande box far on-court de la frame :

```
on-court := cy >= 0.25·net_y  ET  0.12 < cx/W < 0.88
S = 1.0·(h / hmax) + 0.3·conf + 1.2·central + 0.3·valid     # central = 1 − |xr−0.5|·2
```

- **height/hmax** = le signal CROSS-GÉOMÉTRIE porteur : le VRAI far player est la box
  above-net la plus HAUTE on-court (un *joueur* distant > un *spectateur* assis
  distant) — vrai sur demo3 ET felix. Normalisé → scale-free (zéro pixel demo3).
- **central (wc=1.2)** = demo3 a besoin (son joueur est central) ; felix tolère
  (plateau jusqu'à wc~1.3, cliff à 1.5).
- **valid (0.3)** = bonus SOFT pondéré, **JAMAIS une gate dure** — c'est le fix du
  backfire de la gate keypoint (le far minuscule ne pose pas ; une gate nkp≥10 rend le
  slot au spectateur net). Load-bearing pour felix (le retirer : felix 88%→62%).
- **side-bucketing par CENTRE de box** (`cy >= net_y`, pas les pieds) — sûr sur angle
  rasant (les pieds du far felix projettent SOUS net_y mid-frame ; le bucketing par
  pieds le mal-routerait côté near et le slot far le perdrait).
- **pool far = union (size-rank ∪ geom-rank)** — garantit que le vrai far est POSÉ
  quelle que soit la géométrie. + **det_conf 0.08** (rend le far minuscule détectable).

`hcv` **subsume déjà les deux idées de B** : la centralité (terme `1.2·central`) et le
découplage squelette (`valid` est un terme pondéré, jamais une gate). A a aussi un
far-CORRECT demo3 supérieur (84.9% vs 81.3%) ET est le seul prouvé non-régressif felix.

### Ce que j'ai greffé de B (complémentaire, que A n'a pas)

**Récupération du SQUELETTE far par wide-crop** (`pad_frac=1.0`, `far_pose_imgsz=640`,
`conf=0.05`) sur la box far DÉJÀ choisie. A pose les candidats far au crop uniforme
(pad 0.25, imgsz 960) qui rend souvent 0 keypoint sur le far minuscule → squelette
creux. La greffe re-pose la box far choisie en crop large best-effort **uniquement si
le squelette courant est creux (<10 kpts) ET si le wide-crop fait STRICTEMENT mieux** :
- elle ne change JAMAIS la box → **far-CORRECT structurellement inchangé** (la sélection
  est découplée du squelette — l'idée porteuse de B) ;
- elle ne jette jamais un bon squelette que A avait déjà.
C'est un gain additif pur sur la densité de squelette far. **Skeleton far recovery
MESURÉ (38 frames GT échantillonnées) : squelette far VALIDE (≥10 kpt) 18%→84%
(7/38→32/38, +25 frames, +66pp)** — A laissait le far creux sur 82% des frames ; la
greffe le récupère sur la grande majorité. far-CORRECT inchangé (84.9% avec/sans).

### Ce que j'ai écarté (sur preuve)

- Mon brouillon centralité-`_far_score` (régresse felix).
- La gate dure `is_geometric_far_player` de B (A prouve : gate-centralité → demo3 26.7%).
- La pénalité motion/static de A (A a mesuré qu'elle NUIT sur demo3 ; reste OFF derrière
  `track_state`, dispo mais non câblée).

## Chiffres canoniques (prod, scorers officiels)

- **demo3** : `tools/pose_gt/measure_far_coverage.py` → far DETECTED 225/225, far
  **CORRECT 191/225 = 84.9%**, mean 121px. (baseline re-mesuré même scorer : 2/225 =
  **0.9%**, 572px.)
- **felix** : `tools/pose_gt/_measure_felix.py` (proxy-GT 66 frames, construit par A
  depuis les picks fiables de l'ancien sélecteur) → far DETECTED 66/66, far **CORRECT
  58/66 = 87.9%**, mean 91px.

Le far-CORRECT intégré = **identique à A sur les deux clips** (la greffe préserve la
sélection à l'octet, elle ne touche que le squelette) ⇒ on hérite des bornes de A :
demo3 plafond détection 89.3% (84.9% = 95% du plafond atteignable), plateau anti-overfit
(±grid des poids reste ≥80% sur les deux clips). **vs B : +3.6pp demo3 ET felix non
cassé** (B n'avait pas de garde felix).

## Felix / tests — non-régression (VERTS avant ET après)

5 tests de régression obligatoires, mesurés VERTS :
- `test_bounce_regression` : F1=0.800 (floor 0.72) ✅
- `test_bounce_wasb_regression` : F1=0.865 (floor 0.80) ✅
- `test_vx_veto` : PASS ✅
- `test_event_confusion_regression` : H→B=0 B→H=0, bounce F1=0.842 ✅
- `test_sharp_turns` : PASS ✅

(Le changement far-sélection ne touche AUCUN chemin bounce/event ⇒ structurellement
insensible ; mesuré identique au baseline.) Felix far-CORRECT **87.9%** (≥ A) prouve
zéro régression sur l'angle de coin.

## Nouveau test de lock — `tests/test_far_coverage.py`

FAST (no GPU / no video) : rejoue le cache committé de candidats far posés par frame GT
(`tests/fixtures/pose_gt/demo3_far_select_cache.json`) à travers la **fonction de
scoring PROD partagée** `vision.pose._hcv_far_score` + la même on-court gate, et asserte
far-CORRECT ≥ **plancher 0.80** (sous le 84.9% atteint, au-dessus du 81.3% de B et très
loin du 0.9% baseline). La fonction de scoring étant PARTAGÉE entre prod et test, le
test **ne peut pas dériver** de la prod. Il verrouille aussi les deux propriétés
porteuses : height-dominance (une box plus haute gagne) et valid-soft (une box haute
INVALIDE bat une courte VALIDE). Cache régénérable via
`tools/pose_gt/build_far_select_cache.py` (225 frames, 5439 candidats far, 24.2/frame).
**Résultat replay : 190/225 = 84.4%** (≥ floor 0.80, cohérent avec le live 84.9% à la
densité de détection près — identique à la prédiction cache→live de A : 84.4%→84.9%).
Les 3 fonctions de test PASS.

## Limites / honnêteté

- **Sélection = identique à A** : mon gain net SUR LA SÉLECTION vs A = 0 (par design —
  `hcv` est déjà au plateau, et toucher la détection était hors mandat). Mon apport
  propre = (1) le sur-ensemble strict (battre B + garde felix dans UN fix), (2) la
  densité de squelette far (greffe B), (3) le refactor `_hcv_far_score` partagé + le
  test de lock fast.
- **felix GT = proxy** (66 frames, depuis les picks de l'ancien sélecteur), pas une GT
  humaine. Valide « intégré ≈ A sur felix », pas un absolu. Ne pas sur-optimiser dessus.
- **2 clips seulement.** Zéro-hardcoding respecté (tout ratio de frame/net_y/hmax), mais
  la généralisation à une 3e géométrie reste non prouvée. La bande latérale on-court
  (0.12–0.88) est lâche par design pour qu'un far modérément décentré qualifie.
- Plafond détection demo3 89.3% (24/225 frames sans aucun candidat près du vrai far) —
  fermer le reste = lever DÉTECTION (imgsz/conf), pipeline-wide, hors mandat sélection.

## Pour le PM / la suite

- Le livrable : `vision/pose.py` (hcv + greffe squelette far + `_hcv_far_score` partagé),
  `tests/test_far_coverage.py` (lock), `tools/pose_gt/build_far_select_cache.py` + fixture.
- Prochain lever (hors session) = DÉTECTION du far (le plafond, pas la sélection) :
  B a noté imgsz640 trouve le far PLUS souvent qu'imgsz1280 (la cible minuscule se
  fragmente en haute-rés) — piste à creuser, effet pipeline-wide.
