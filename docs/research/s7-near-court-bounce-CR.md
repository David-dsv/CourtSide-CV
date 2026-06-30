# CR — S7 : récupérer les rebonds NEAR-COURT rasants (le trou de génération mesuré)

**Branche :** `feat/s7-near-court-bounce` (depuis `feat/accuracy-overhaul`), worktree
`../CourtSide-CV-s7nearbounce`. **Fichiers de prod touchés : `vision/events.py` + le câblage
1-ligne dans `run_pipeline_8s.py`** (+ tests/outils force-add). ULTRACODE, measure-driven,
vérifié par replay contrôlé à track-identique.

## TL;DR

Le trou mesuré (CR `event-rootcause-2026-06-28`, retour utilisateur du hero render) : les rebonds
**near-court rasants b174 et b308 n'ont AUCUN candidat** dans le pool. La balle est parfaitement
suivie à l'impact (4-6 px du GT) mais **aucun générateur ne fire** :
- **Pas de vy-flip** : la balle descend vite puis **repart vers le bas LENTEMENT** (vy passe de
  ~+30 à ~+3 mais **ne s'inverse pas**) → `detect_turning_points` (extremum-y) et
  `detect_bounces_robust` (vy-flip) sont aveugles.
- **Angle de virage trop doux** : b174 = 36-38°, b308 = 53-55° → **sous** le seuil 70° de
  `detect_sharp_turns`, ET près du bas de l'image la série d'angle est étalée donc le coude rasant
  n'est **pas un pic proéminent** pour `find_peaks` (baisser l'angle seul ne couvre que 1/3 et
  commence à flooder le far — vérifié, cf. §pistes rejetées).

**Le fix = un générateur dédié `detect_near_court_knees` (vision/events.py)** qui capte la
signature physique du **« genou » de vy** : `vy_pre` fortement positif (descente) puis `vy_post`
qui **se tasse vers ~0 SANS s'inverser**, **gaté au near-court** (`y > net_y`). Il **alimente
`turning_frames`** comme `detect_sharp_turns` ; le firewall de `classify_events` arbitre toujours
BOUNCE vs HIT (un contact raquette near décélère aussi la balle → un genou n'est jamais
auto-labellisé BOUNCE).

**Résultat — REPLAY CONTRÔLÉ (à track-identique WITH vs WITHOUT, PAS un run live), sur
l'instrument fidèle (track hero canonique post-S1, 90 % de couverture) :**

| | bounce F1 | hit F1 | conf H→B | conf B→H | bounce misses | spurious |
|---|---|---|---|---|---|---|
| **SANS knee** | 0.750 | 0.625 | 0 | 1 | **[174, 308]** | 3 |
| **AVEC knee** | **0.889** | 0.625 | **0** | 1 | **[]** | **3** |

⚠️ **Pas de gain F1 LIVE démontré** : ce +0.139 est un gain de REPLAY contrôlé (isolation propre
du changement). Sur le seul track live mesuré (MPS 69 %, byte-déterministe sur ce box) le knee est
un **no-op byte-identique** (§Validation LIVE). La conclusion honnête = « récupère le trou de
génération là où il existe (hero), ne coûte rien là où le track le couvre déjà (live) ».

→ **bounce F1 0.750 → 0.889 (+0.139)**, b174+b308 **tous deux récupérés et classés BOUNCE**,
hit F1 **tenu**, **firewall H→B = 0 tenu**, **spurious COUNT identique (le fantôme @231 N'EMPIRE
PAS)**, B→H = 1 inchangé (résidu S2 pré-existant). **Zéro flood** : `near_knees` ne fire QUE sur
des frames GT (0 spurious sur tout le track, plateau anti-surapprentissage de 36 combinaisons).

## Le diagnostic exploité (mesuré, pas re-supposé)

Probe sur le dump canonique post-S1 (`tools/.../probe`, RAW + spline), série vy autour de chaque
GT bounce (k=3, médiane de pente par côté) :

```
  b62  NEAR : vy_pre=+25.3 vy_post=+14.0   KNEE (descend → se tasse, pas de flip)
  b174 NEAR : vy_pre=+13.7 vy_post= +2.7   KNEE  ← TARGET (angle 36-38°)
  b308 NEAR : vy_pre=+17.0 vy_post= +1.3   KNEE  ← TARGET (angle 53-55°)
  b456 NEAR : vy_pre=+29.3 vy_post= -6.0   FLIP (caught by robust_bounce/turning_points)
  b120 FAR  : vy_pre= +0.0 vy_post=-11.0   FLIP (caught by sharp_turns)
  b255 FAR  : vy_pre= -1.3 vy_post=-88.3   FLIP
  b369 FAR  : vy_pre= +1.0 vy_post= -9.0   FLIP
```

La signature **descend-puis-se-tasse-sans-s'inverser** sépare proprement les 2 cibles des autres
rebonds (qui flippent et sont déjà attrapés). Un ballon en plein vol a vy ~constant (pas de
genou) ; un rebond franc flippe (genou + inversion) ; **seul un rebond rasant** montre
descend→tasse-sans-flip.

## Le générateur retenu — piste A (vy-knee), et POURQUOI les 2 autres perdent

Mesuré sur le dump canonique (couverture par-cible + FP near-court) :

| piste | mécanisme | b174 | b308 | spurious (FP) | verdict |
|---|---|---|---|---|---|
| **A — vy-knee** | descend (`vy_pre≥0.012·fh`) → tasse (`|vy_post|≤0.010·fh`, **non flippé**) | ✅ | ✅ | **0** | **RETENUE** |
| B — angle local near | seuil d'angle abaissé, gaté near-court | ✅ (k=2/3 @35°) | ✅ | 0 | perd b174 dès 45° (b174 à 36° = bord du plancher) → **fragile** |
| C — courbure `|v×a|/|v|³` near | pic de courbure near-court | ✅ | ✅ | **1-3** (f322, f586, f186) | **flood** à tous les seuils → rejetée |

- **Piste B** marche techniquement (angle≥35° near-court, k=2/3) mais b174 est **à 36°, sur le
  bord du plancher** : à 45° elle ne garde que b308. C'est le même virage que la piste A mesuré
  via une primitive plus fragile (find_peaks sur une série d'angle plate au near-court). Marge
  trop mince.
- **Piste C** récupère les 2 cibles mais génère **systématiquement 1-3 faux candidats**
  (f322/f586/f186) — la courbure pique aussi sur le bruit de tracking near-court. Inacceptable
  vu la contrainte « zéro flood » (le DP global est couplé : chaque candidat parasite peut
  cascader un mislabel, cf. [[courtside-s3-event-recall-negative]]).

La piste A est **la plus propre (0 FP), la plus physique** (capte directement la signature
identifiée), et **la plus robuste** : sur un balayage de **36 combinaisons** (k∈{2,3,4},
`desc_min_frac`∈{0.008…0.015}, `settle_max_frac`∈{0.008…0.012}) elle récupère **toujours** les 2
cibles avec **0 FP et conf H→B = 0** (bounce F1 0.889 plat partout). Ce n'est **pas** un magic
number réglé sur 3 frames — c'est une signature à grande marge (`vy_pre` 13-29 vs `vy_post` ~0-7).

**Garde anti-téléport (durcissement mesuré sur le track live, leçon S3).** Sur le track live
dégradé un **glitch de tracking** (saut vertical de +202 px en 1 frame à f291, +185 à f122) faisait
fire le knee hors-GT (lecture de pente faussée par le glitch). Mesure décisive : les vrais knees
ont `max|dy|` 1-frame **≤ 45 px (≤4.2 % fh)** ; les glitches **≥ 185 px (≥17 % fh)** — marge ×4.
Une garde « rejeter si un pas vertical 1-frame > 8 % fh » (scale-free) **supprime f291+f122 sans
toucher aucune cible** (canonique inchangé 0.889, plateau 36-combos toujours propre). Le knee ne
fire désormais QUE sur des genoux physiquement plausibles → claim no-flood rendu rigoureux.

## Zéro hardcoding

Tous les seuils en fractions : `net_y_frac=0.47` (= la ligne de filet déjà dérivée upstream à
`frame_height*0.47`), `desc_min_frac=0.012·fh`, `settle_max_frac=0.010·fh`, `win_frac=0.06·fps`,
`min_gap_frac=0.12·fps`. Aucun pixel d'un clip. La condition « near » = `y > net_y` (pas une bande
tunée). Lecture **RAW pré-spline + masque is_real** avec les 3 ancres (f-k, f, f+k) réelles — même
invariant de sécurité que `detect_sharp_turns`/le veto : aucun genou fabriqué à travers un gap
interpolé.

## Validation LIVE (l'arbitre final) + l'honnêteté sur la variance de track

⚠️ **La piste de balle live VARIE run-à-run** (non-déterminisme cross-run YOLO/MPS, documenté
[[courtside-hits-overfire-device]] / [[courtside-s3-event-recall-negative]]). C'est pourquoi la
**bonne mesure d'un générateur est un replay CONTRÔLÉ à track-identique** (WITH vs WITHOUT sur le
MÊME dump), pas 2 runs live séparés (leçon de vérif S2/S6).

### Replay contrôlé à track-identique (l'isolation propre du changement)

**Track HERO canonique (dump post-S1 committé, 90 % cov = ce dont l'utilisateur s'est plaint) :**

| | bounce F1 | hit F1 | H→B | B→H | bounce misses | spurious |
|---|---|---|---|---|---|---|
| SANS knee | 0.750 | 0.625 | 0 | 1 | **[174, 308]** | 3 |
| AVEC knee | **0.889** | 0.625 | 0 | 1 | **[]** | 3 |

→ Le knee **récupère b174+b308** (`near_knees` fire @ 63,79,172,194,307,331,455,568 — TOUS sur
GT), **+0.139 bounce F1**, 0 flood (spurious=3 inchangé, fantôme @231 intact), firewall tenu.

**Tracks LIVE de ce box (3 runs : run1+run2 `--device mps`, run3 `--device cpu`,
`tennis.mp4 -s73 -d13 --match-mode`, `_stats.json` réel scoré par `score_stats.py`) :** les 3 runs
sont **byte-identiques** (track à **69 % de couverture** — sur ce box MPS≡CPU, déterministe). Replay
contrôlé WITH-vs-WITHOUT sur chacun :

| run (track) | | bounce F1 | hit F1 | H→B | B→H | bmiss | spurious | near_knees |
|---|---|---|---|---|---|---|---|---|
| run1/2/3 (69 %) | SANS | 0.556 | 0.444 | 0 | 4 | [] | 6 | — |
| run1/2/3 (69 %) | AVEC | 0.556 | 0.444 | 0 | 4 | [] | 6 | @63,455 (post-garde) |

→ Sur ce track dégradé, **WITH ≡ WITHOUT byte-identique sur les 3 runs** : le knee est un **no-op
propre (0 régression, 0 flood)**. Raison mesurée : sur ce track bruité les `turning_points` denses
(30 cands vs 16 sur le hero) couvrent **déjà** 174/308, donc `bmiss=[]` sans le knee ; le mur ici
est le **placement du DP global** (B→H=4, 6 spurious **non issus du knee** — `near_knees` ne fire
que sur 2 frames GT, @63/@455, après la garde anti-téléport qui a éliminé f291/f122), exactement le
mur hors-scope S7 documenté [[courtside-s3-event-recall-negative]] /
[[courtside-hits-overfire-device]]. Le `_stats.json` de prod scoré directement **= le replay
byte-pour-byte** (bounce F1 0.556, B→H 4, spurious identiques) → l'instrument de replay est fidèle.

**Synthèse honnête :** le knee **récupère le trou de génération near-court là où il existe (track
hero, +0.139 bounce F1 en replay contrôlé) et ne coûte rien là où le track le couvre déjà par
hasard (no-op byte-identique sur 3 runs live)**. **Aucun gain F1 LIVE démontré** (le track live de
ce box ne reproduit pas le trou hero) — seulement une **non-régression prouvée** sur les 3 runs +
le gain de génération prouvé en replay contrôlé sur l'instrument fidèle. Aucun run, aucune des 36
combinaisons de seuils, ne le voit flooder ou casser le firewall. La variance de track (69 %↔90 %)
est **cross-environnement** (build/CI), pas run-à-run ni cross-device sur ce box.

## Tests verts

`test_near_court_knee.py` (nouveau, 8 tests) : récupère b174+b308, **test discriminant**
(le pool pré-fix turning_points+sharp_turns NE couvre PAS 174/308 → un revert échoue), no-flood
(0 spurious, ≤12 cands), gate near-court, **garde anti-téléport** (rejette un knee à glitch
1-frame), plateau anti-surapprentissage (36 combos), bout-à-bout (bounce F1 ↑, hit F1 tenu,
H→B=0), entrées dégénérées.

Suite de régression (toute verte) :
- `test_event_confusion_regression` : cache **0/0, bounce F1 0.889, hit F1 0.737** (inchangé — le
  knee n'est pas sur le chemin du cache).
- `test_bounce_regression` : felix **F1 0.800 / 0.774** (inchangé).
- `test_bounce_wasb_regression` : felix **0.865** (inchangé).
- `test_vx_veto`, `test_sharp_turns`, `test_static_fp_reject` (13/13), `test_ball_anchored_prox`,
  `test_robust_slope` : verts.

**felix prouvé inchangé** : `detect_near_court_knees` est une fonction NOUVELLE qui n'alimente que
`turning_frames` ; le chemin felix (vx-veto/robust) ne l'appelle pas, et les tests felix sont
byte-identiques.

## Vérification adversariale (3 lentilles indépendantes)

Un workflow de 3 reviewers adversariaux (mécanisme / no-flood / honnêteté-CR) a indépendamment
reproduit les 2 lignes du replay canonique (0.750→0.889) et confirmé : firewall H→B=0 robuste même
sous élargissement forcé des seuils (9 configs), fantôme @231 prouvé NON issu du knee (source =
turning_points+sharp_turns), zéro hardcoding, invariant de sécurité tenu. **Un vrai bug trouvé +
corrigé : la garde anti-téléport était gatée sous `if win_real:`** → un gap dans la fenêtre d'un
glitch désactivait silencieusement la garde (un track-reinit est souvent flanqué de gaps). Fix =
mesurer le pas vertical sur les **paires consécutives RÉELLES** sans pré-condition all-real ;
nouveau test adversarial `test_teleport_guard` couvre le cas glitch+gap. Verdict final = **GO**.

## Reusable lessons

1. **Une signature physique à grande marge bat un seuil réglé** : le genou de vy (descend→tasse
   sans flip) a une marge telle que 36 combinaisons donnent le même résultat. Quand le balayage
   est plat, ce n'est pas du surapprentissage.
2. **Juger un générateur par replay contrôlé à track-identique**, jamais par 2 runs live séparés
   (la couverture de balle varie 69 %↔90 % d'un run à l'autre et noie le signal du générateur).
3. **Le mur restant est le placement DP sur les tracks dégradés** (B→H, spurious), pas la
   génération de candidats — confirme [[courtside-s3-event-recall-negative]]. Le knee ferme
   proprement le trou de génération near-court sans toucher ce mur (no-op sur track bruité, gain
   sur track hero).
4. **⚠️ Le piège cwd-reset est destructeur** : un `ln -sf <abs> tennis.mp4` relatif lancé quand le
   cwd a silencieusement reset au MAIN tree a créé un symlink auto-référent qui a **détruit le
   tennis.mp4 réel** (restauré depuis `../CourtSide-CV-legacysep/`). **Toujours chemins absolus**
   pour les artefacts hors-worktree. Étend [[courtside-edit-targets-main-tree]].
