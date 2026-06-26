# Auto-Homographie Court sur Angles Obliques / Amateurs — Design de Recherche

> **Statut : RESEARCH-ONLY.** Ce document synthétise quatre fils de recherche (classique no-GPU, pilote GPU/RF-DETR, licences, validation) et leurs critiques adversariales. **Aucun code prod n'est écrit ici.** Toutes les estimations chiffrées de succès, latence et coût sont des **cibles à mesurer**, pas des faits, sauf les *sondes mesurées en live cette session* (tableau §2) et les *licences vérifiées contre les fichiers LICENSE réels* (§7).

---

## 1. TL;DR / Résumé

- **Recommandation : la voie classique no-GPU (line + RANSAC VP + warp-back self-score) D'ABORD.** Pure OpenCV/numpy/Apache-MIT-BSD, zéro GPU, zéro donnée à labelliser, license-clean — elle remplace le modèle keypoint mort *sur la classe d'angle exacte que le produit vise* (amateur derrière la ligne de fond).
- **Pourquoi** : les sondes live prouvent que le modèle keypoint broadcast (`yastrebksv`) **s'effondre par domain-shift** sur l'angle cible (Djokovic court-level **0–2/14**, felix grazing **1–3/14**) alors que le masque ligne-blanche extrait un **quasi-wireframe complet** là où le CNN donne 0/14. La géométrie (court ITF rigide, caméra statique, une bonne frame suffit) est précisément le régime où Farin/VP gagne.
- **Honnêteté sur la cible réelle** : le classique n'est **PAS un win propre sur Djokovic**. Le VP-spread mesuré y est **857/702 px** (vs 7 px sur tennis) — un ordre de grandeur plus lâche, propageant une grosse erreur en champ lointain. Djokovic est le cas **TBD / la question ouverte**, possiblement seulement LOW-confidence. Et **felix (grazing) est insoluble par le classique** (1 seule famille d'orientation) → reste sur la calibration manuelle sidecar.
- **Effort estimé** (jugement d'ingénierie, non mesuré) : ~6 chantiers (§9), chacun testable indépendamment sur les 3 clips-sondes déjà en main ; voie classique livrable sans GPU ni donnée.
- **Risques ouverts** : (a) l'association *ligne détectée → ligne ITF nommée* est le vrai cœur non résolu, pas le fit d'homographie ; (b) le gate grazing repose dangereusement sur un seul veto (famille-oblique=0) ; (c) blow-up métrique en champ lointain sur VP lâche ; (d) ambiguïté flip 180°.
- **Pilote GPU (RF-DETR keypoint, Apache-2.0)** : gardé en **pilote gated**, lancé seulement après un **smoke-test MPS de 1 h**. La dépense réelle = **labelliser ~300–1500 frames obliques**, pas le GPU ($15–60 all-in). C'est le plafond de précision, pas le déblocage immédiat.

---

## 2. Diagnostic — voie actuelle + preuve mesurée

### 2.1 Cascade actuelle (citations file:line)

La voie homographie actuelle est un fallback en cascade, câblé dans `run_pipeline_8s.py:722-755` :

1. **Keypoint** — `run_pipeline_8s.py:722` : si `--homography`, `CourtKeypointDetector.compute_homography(first_frame, frame_diag)` (`models/court_detector.py`). 14 keypoints ITF → solve LMEDS. Si **`confidence == 'fallback'`** (le détecteur émet ça quand `n < 4`), `court_homography = None` (`run_pipeline_8s.py:731`).
2. **Calibration sidecar** — `run_pipeline_8s.py:745` : si `court_homography is None and not args.no_calibration`, `load_calibration(video_path)` charge `calibrations/<name>.court.json` (ou `<video>.court.json`). Semi-auto 4-corner manuel via `models/court_calibrator.py::homography_from_points`, qui solve `cv2.findHomography(..., ransacReprojThreshold=0.10)` — **seuil en MÈTRES (10 cm), pas en pixels** — puis `_assess_conditioning(H_inv, world_pts, img_arr, n, rms)` (`court_calibrator.py:109/121`) flague le grazing via aspect-ratio de l'image-court.
3. **Scalar** — sinon, `estimate_px_per_meter` (court ≈ 58 % hauteur frame) ; le pipeline flague honnêtement `speed_source=scalar`.

**Point d'insertion proposé** pour le détecteur auto-classique : **entre `:738` et `:745`**, *avant* le bloc sidecar, donc ordre = keypoint(722) → **auto-line(nouveau)** → sidecar(745) → scalar. Drop-in complet : `vision/court_zones.py::img_to_world` est **matrix-agnostic** → n'importe quel H propre (classique OU modèle) tombe dedans sans aucun changement downstream (minimap, speed, depth). C'est l'argument de déployabilité le plus fort.

> Fait vérifié cette session : `calibrations/felix.court.json` contient **3 coins doubles + 1 marque de service** — `{far_bl_left, far_bl_right, near_bl_left, far_svc_center}` — il **manque `near_bl_right`** (near-right occlus, cas grazing). Le JSON **n'a pas de champ `confidence`** (le pipeline le dérive à `low` au chargement). C'est l'inventaire de départ réel pour le builder de GT (§8), pas "4 coins".

### 2.2 La preuve mesurée — le modèle keypoint domain-shift sur l'angle cible

Sondes exécutées **en live cette session** sur les vraies vidéos du repo (1920×1080, MPS, pas de CUDA) :

| Clip | Angle | Keypoint (n/14) | Conf. | Masque ligne-blanche | Familles orient. | VP récupérable ? |
|---|---|---|---|---|---|---|
| **tennis.mp4** | broadcast oblique élevé | **13–14/14**, rms ~4.6px | HIGH | 37–43 lignes | 2 (H + obliques) | oui, **spread 7 px (serré)** |
| **felix.mp4** | grazing side-on (caméra haute, regard rasant) | **1–3/14** → FALLBACK | — | 26–35 lignes mais **toutes ~horizontales** | **1 (effondrée)** | **non** (sidelines écrasées) |
| **Djokovic vs Ruud court-level 4K** | LOW derrière ligne de fond, court complet, 4 coins visibles = **cible SaaS réelle** | **0–2/14** → FALLBACK | — | **96–221 lignes**, wireframe quasi-complet | 2 (H=71, S=25) | oui, **spread 857/702 px (lâche mais présent)** |

> ⚠️ Résolution Djokovic : sondé "4K" mais lignes/VP rapportés comme sur frame 1920×1080. Le **VP-spread absolu (857/702 px) n'a de sens que rapporté à la diagonale** : sur diag 2203px (1080p) c'est **32–39 % de la diagonale**. Si la frame réelle est 3840×2160, *tous* les seuils px (ε, d_tol, distance_threshold) **et** ce 857 doivent être re-normalisés. La discipline percentile/ratio du design absorbe la plupart, mais **état de la résolution à figer dans le cache** (§8).

**Conclusion du diagnostic** : c'est un **domain-shift du front-end appris**, pas la pathologie de felix. Le modèle broadcast collapse sur **la classe d'angle exacte** que le produit cible (court-level amateur). Mais — point que la critique impose d'expliciter — **les DEUX** (keypoint ET VP classique) sont faibles sur Djokovic. Le classique n'est donc **pas un fallback fiable garanti sur la cible** ; il est *meilleur que rien et license-clean*, à valider.

---

## 3. Inventaire des signaux disponibles par classe d'angle

Ce que chaque signal apporte, et sa fiabilité mesurée par classe :

| Signal | broadcast (tennis) | court-level (Djokovic) | grazing (felix) | Usage |
|---|---|---|---|---|
| **Lignes blanches (masque HSV+ridge)** | 37–43, 2 familles | 96–221, wireframe quasi-complet | 26–35 mais **toutes horizontales** | matière première du fit ; survit où le CNN meurt |
| **Coins = intersections de lignes** | nets | présents (au moins la moitié near) | near-right **occlus** | sous-pixel via `cross(L_A,L_B)` ; cibles du solve |
| **Familles d'orientation → VP** | 2 (tight 7px) | 2 (lâche 857/702px) | **1 seule** → abstain | gate de dégénérescence + réducteur de recherche correspondance |
| **Net line / net posts** | visibles | visibles | partiel | points bien-écartés en profondeur (améliorent le conditionnement far-field) |
| **Lignes de service (T marks)** | visibles | visibles | `far_svc_center` cliqué | idem ; **séparent la mesure d'erreur des coins du solve** (hold-out, §8) |
| **Zones court (modèle HF)** | domain-shift, retourne {} | idem | idem | non fiable (même domain-shift) ; non utilisé comme signal primaire |
| **Côté joueur / scoreboard** | présent | présent | présent | désambiguïsation du **flip 180°** seulement |

**Fiabilité par classe (mesurée)** : broadcast et court-level ont **2 familles de lignes** (condition nécessaire pour 2 VP) ; grazing **n'en a qu'une** → l'axe de profondeur est inobservable, aucune quantité de lignes ne le sauve. C'est la frontière physique entre "le classique peut tenter" et "abstain obligatoire".

---

## 4. Approche 1 — Classique no-GPU (line + RANSAC + warp-back self-score)

Backbone : les 4 étapes de **Farin** (détecteur lignes-blanches → params lignes RANSAC → **fit combinatoire du modèle court scoré par back-projection** → tracking Chamfer), dont la seule implémentation open-source de référence est **gchlebus/tennis-court-detection** (C++, BSD-3-Clause — *permissif, renforce le récit license-clean*), spécialisée amateur par **2404.06977** (Agrawal et al., "Accurate Tennis Court Line Detection on Amateur Recorded Matches").

### 4.1 Front-end masque ligne-blanche

Masque combiné `M = M_bright ∧ M_ridge ∧ M_notcourt`, tous seuils en **percentiles** (zéro hardcoding) :
- `M_bright = (V > p80) & (S < 60)` — peinture blanche brillante/désaturée.
- `M_ridge = top-hat morphologique > p95` sur une **échelle de kernels** {3,5,7}px (largeur de ligne inconnue avant l'échelle px) — tue les gros blobs brillants (maillots, ciel), garde les ridges fins.
- `M_notcourt` = **le test couleur-court 7×7 de 2404.06977** : `court_color` = mode de 1000 px échantillonnés ; un pixel est ligne ssi `|px − court_color| > τ` ET ≥4 de ses 7×7 voisins sont à τ de `court_color` (centre PAS court-coloré). **C'est la sous-composante classique qu'on adopte** (la critique impose de ne PAS la survendre comme "la contribution cœur" du papier — le headline réel du papier est son pipeline ML : shadow-removal MTMT/ShadowFormer + crop net YOLOv5, qu'on **n'utilise pas**).
- Shadows : approximation classique **retinex / flatten d'illumination** (division par un gros flou gaussien) avant le test ridge. **Pas de shadow-removal ML** (GPU).

### 4.2 Extraction de segments + fusion colinéaire

- **HoughLinesP** (`minLineLength=10% diag=220px`) est ce que les sondes ont utilisé et qui a marché → **chemin par défaut, totalement Apache-clean** (jamais eu le problème de licence du LSD).
- Option : **FLD (`cv2.ximgproc`) / LSD (`cv2.createLineSegmentDetector`)** pour endpoints sous-pixel et moins de segments parasites. ⚠️ **Caveats** : (a) LSD a été **retiré d'OpenCV 4.1.0–4.5.3** (conflit code-licence GPL de la routine NFA), **restauré en 4.5.4+** sous réimplémentation MIT — exiger `opencv>=4.5.4` ; (b) **FLD/ximgproc nécessite `opencv-contrib-python`**, qui n'est **pas** dans `requirements.txt` (seulement `opencv-python>=4.8.0`), et contrib/non-contrib coexistent mal → **fallback HoughLinesP** couvre la correction ; le changement de dépendance est un item de chantier (§9). Le repo a aujourd'hui cv2 4.13 + ximgproc incidemment, mais l'environnement est fragile (Python 3.14 beta, pas de lockfile).
- Fusion : clusterer les segments par `(θ ±2°, ρ ±0.5% diag)`, refit chaque cluster par **total-least-squares** (`cv2.fitLine` DIST_L2) sur tous les pixels-masque dans une bande → ~10–25 **candidats de ligne maximaux**, chacun avec `(params, endpoints, support = #px masque ±2px)`. Le support est la monnaie du scoring downstream.

### 4.3 Classification en 2 familles + VP RANSAC + gate d'abstention

Le court ITF en perspective a **exactement 2 pinceaux** de lignes concourant en 2 VP : famille A "longueur" (sidelines + ligne de service centrale) → VP_len ; famille B "largeur" (baselines, lignes de service, filet) → VP_wid.

```
# VP par famille (closed-form + RANSAC consensus)
vp = argmin_v Σ_i (L_iᵀ v)²  s.t. ‖v‖=1   # plus petit vecteur singulier droit de [a b c]
inliers = {L : point_line_dist(vp, L) < ε≈0.3% diag}
```

**Gate d'abstention (AVANT tout fit)** — retourne `{"confidence":"fallback","H":None}` si :
- **#familles < 2** (felix : tout-horizontal) ← **c'est le veto primaire et propre** qui sépare felix des deux autres ;
- min(lignes/famille) < 2 (impossible de former un quad) ;
- `VP_spread/diag > τ_vp` (cloud trop lâche).

> ⚠️ Réconciliation des seuils (la critique l'impose) : le **veto famille-count** fait le travail propre. Le seuil de *spread* `τ_vp≈0.6` est libre mais **doit être réconcilié avec le 0.32–0.39 mesuré de Djokovic** : un τ_vp de 0.6 *laisserait passer* un VP bien pire. Donc le gate repose **primairement sur famille-count**, le spread n'est qu'un feature secondaire. **Renvoyer `fallback` est le comportement CORRECT**, pas un échec : felix a déjà son sidecar manuel qui prend le relais à `:745`.

### 4.4 Fit du modèle : correspondance + warp-back self-score

4 correspondances point = 2 lignes famille-A × 2 lignes famille-B → 4 intersections, matchées aux 4 intersections monde. **Réduction VP-contrainte** : les VP ordonnent les pinceaux (distance signée → gauche..droite, far..near), réduisant l'assignation à ~10–50 hypothèses H au lieu de milliers.

> 📐 Cadrage géométrique honnête (la critique l'impose) : **2 VP ne donnent PAS H seul.** Deux points de fuite donnent la rotation/rectification métrique *up-to-scale* ; il faut **une longueur connue (ou 4 points)** pour fixer échelle+translation (Liebowitz & Zisserman). Ici les VP sont un **réducteur de recherche + gate de dégénérescence**, et le H final vient de `cv2.findHomography` sur 4+ points (mieux conditionné, cohérent avec le repo). Ne PAS dire "les VP donnent l'homographie".

**Warp-back self-score** (le gate de confiance) — pour chaque candidat H, rendre les lignes du modèle ITF via H_inv et mesurer l'accord avec le masque blanc M, via Chamfer / distance-transform (concept emprunté à Nie WACV2021 — qui est un **réseau profond** ; on n'emprunte que l'idée géométrique DT/Chamfer, pas la méthode ML) :

```
DT = distanceTransform(255 − M)
proj_pts = [world_to_img(H_inv, p) for p in sample_along(MODEL_LINES, 0.25m) if in_frame(p)]
if len(proj_pts) < N_min: score = −inf            # court projette hors-cadre → reject
chamfer  = mean(DT[p] for p in proj_pts)          # ↓ meilleur
coverage = frac(proj_pts with DT[p] < d_tol≈0.4% diag)   # ↑ meilleur
SCORE = coverage − λ·(chamfer/diag)               # λ≈1.0, scale-free
H* = argmax SCORE
```

**Veto de sanité géométrique** (avant scoring) : les 4 coins monde via H_inv doivent former un quad **convexe, non auto-intersecté** → réutiliser **`court_calibrator._assess_conditioning` verbatim** (encode déjà le test grazing aspect-ratio + convexité), pour que la confiance auto soit comparable à la calibration manuelle.

### 4.5 Correspondance avec la cascade existante

Mapping de confiance vers le champ `confidence ∈ {high, low, fallback}` que le pipeline branche déjà :
```
SCORE ≥ τ_hi ET conditioning high  → "high"
SCORE ≥ τ_lo ET conditioning ≠ low → "low"
sinon                              → "fallback"   # abstain → sidecar/scalar
```
`τ_hi, τ_lo` calibrés une fois sur tennis(HIGH) / Djokovic(doit passer ou LOW) / felix(doit `fallback`). Le pipeline ne distingue que `fallback` vs non à `:731/:754` → drop-in.

### 4.6 Coins auto + émission du dict (zéro changement downstream)

Émettre les **4 coins doubles** en px, identique à un clic humain dans `tools/calibrate_court.py`. Coin = **intersection analytique de deux lignes refit** (`cross(L_A, L_B)`, déshomogénéisé) → **sous-pixel intrinsèque** (raison pour laquelle le classique *peut* battre le centroïde de connected-component du keypoint — **scope : sur les clips fallback uniquement**, PAS sur tennis où le keypoint gagne déjà à 13-14/14). Deux sorties supportées par le repo :
1. **Dict H direct** — même forme que `compute_homography` : `{H, H_inv, confidence, rms_px, n_used, source:"autoline", landmarks, landmark_px}`.
2. **Seed des landmarks nommés** → appeler `court_calibrator.homography_from_points(image_points)` sous les clés `WORLD_LANDMARKS` (`far_bl_left/right`, `near_bl_left/right`, + `*_svc_*`, `net_left/right` si détectés → plus de points bien-écartés = meilleur conditionnement far-field). **Réutilise solver + RMS + conditioning + `save_calibration` (cache une fois par clip) verbatim — le moins de code neuf.**

**Flip 180°** : un rectangle symétrique se mappe sur lui-même tourné 180°. Résolu par : max-SCORE warp-back, + net à Y=0, + near-baseline plus bas/grand dans l'image, + côté joueur/scoreboard si dispo.

### 4.7 Issue attendue par classe + modes d'échec

| Classe | Évidence ligne (mesurée) | Issue attendue (**non testée, cible**) | Pourquoi |
|---|---|---|---|
| broadcast élevé (tennis) | 2 familles, VP 7px | HIGH (keypoint gagne déjà ; classique = alt license-clean) | — |
| **court-level (Djokovic = cible SaaS)** | 2 familles (71/25), VP 857/702px | **TBD — possiblement LOW seulement** | VP lâche (32–39% diag) → erreur far-field ; à valider, **pas un win promis** |
| amateur tripod mi-hauteur | typiquement 2 familles | à mesurer | cas déployable commun |
| **grazing (felix)** | **1 famille**, VP irrécupérable | **ABSTAIN → fallback (correct)** | axe profondeur inobservable ; sidecar/scalar prend le relais |

**Modes d'échec & mitigations** : grazing→1 famille = abstain (mitigation = sidecar manuel, pas un solve) ; clay usé/courts colorés = le test couleur-court 7×7 + retinex ; **court partiel (moitié far occluse)** = le scénario de 2404.06977 — leur trick "crop near-court + extension des lignes au far" utilise **YOLOv5 (AGPL-3.0)** ⇒ **re-dériver l'extension classiquement** (prolonger géométriquement les lignes famille détectées, sans ML) pour rester license-clean ; alley vs singles = ratio d'espacement (1.37m) résolu par warp-back ; faux VP foule/panneaux = consensus RANSAC + `M_notcourt` + veto quad-convexe.

---

## 5. Approche 2 — Pilote GPU-gated (RF-DETR keypoint / heatmap / line-segment)

**Thèse : la dépense est la DONNÉE, pas le compute.** GPU fine-tune = bruit ($15–60). Le coût = labelliser ~300–1500 frames obliques. Et **avant toute donnée**, un smoke-test MPS de 1 h décide si RF-DETR est viable local-first.

### 5.1 Tableau costé des architectures

| Arch | License | Données nécessaires | GPU$ train | MPS-infer | Risque |
|---|---|---|---|---|---|
| **RF-DETR keypoint** (DINOv2-w/-registers + skeleton 14-kpt arbitraire + incertitude/kp) | **Apache-2.0** (code+weights preview) | ~300–1500 obliques | $15–60 | ⚠️ **vérifier** (#427 *corrigé*, valider en 1.8.2) | Med ; churn preview |
| **TrackNet-heatmap court** (même arch que yastrebksv, *tes* données Apache) | code MIT/Apache + **données propres** | ~300–1500 obliques | $15–50 | ✅ **prouvé** (yastrebksv tourne déjà sur ton MPS) | **Low-med ; drop-in dans `court_detector.py`** |
| **RTMPose/ViTPose court-skeleton** | Apache-2.0 (**rtmlib déjà installé**) | ~300–1500 | $15–50 | ✅ rtmlib CPU/MPS | Med ; cadrage "court-as-object" gauche |
| **DeepLSD → lignes → fit géométrique** | **MIT** | **0 (pré-entraîné)** | **$0** | ⚠️ tester (PyTorch, probable MPS-ok) | Low ; upgrade le classique ; **felix toujours dur** |
| **M-LSD** | **Apache-2.0** | **0 (pré-entraîné)** | **$0** | TF→ONNX port | Low-med ; felix toujours dur |
| **Régresseur-H + template overhead** | per-paper | seg labels | $20–80 | ✅ petit net | Med ; pipeline plus lourd |
| **Synthetic homography-warp pretrain** | propre | **génère la donnée** | +$5–20 | n/a (outil) | Low ; réduit *matériellement* le labeling réel ; gap sim2real |

**Lecture** : **TrackNet-sur-tes-données = la voie GPU la plus basse en risque** (drop-in dans `CourtKeypointDetector` inchangé, MPS déjà prouvé, blanchit la licence avec tes labels Apache-clean). Elle perd l'incertitude par-keypoint qui rend RF-DETR attractif. **DeepLSD/M-LSD = 0 entraînement, 0 donnée** — upgrade SOTA du masque classique cette semaine ; **ne résolvent PAS le grazing felix** (1 famille = géométrie, pas détection) mais aident Djokovic.

> 📌 Corrections adversariales appliquées : (a) **MPS #427 est CORRIGÉ** (grid_sample + bicubic interpolation tournent maintenant nativement sur MPS via impl custom ; l'op fautive `aten::_upsample_bicubic2d_aa.out` patchée) → reframé en "vérifier que le fix a atterri en 1.8.2", **pas** "risque binaire non prouvé". (b) **arxiv 2511.04126 mal caractérisé** : ce n'est PAS un nouveau dataset/SOTA court-keypoint ni du classical-Hough — c'est un pipeline étudiant (ResNet50 keypoints) **réentraîné sur le MÊME dataset broadcast 8841-images yastrebksv** ⇒ il **re-confirme le domain-shift broadcast**, ce n'est pas une solution. (c) **L'incertitude par-keypoint** de RF-DETR : présentée comme "covariance 2D calibrée + findable/visible" avec plus de spécificité que le blog cité ne supporte → softené en "incertitude/confidence par-kp (vérifier que c'est une covariance utilisable, pas un scalaire)" ; l'avantage gating-homographie est **non confirmé**.

### 5.2 Données : pas de dataset oblique clean prêt à l'emploi

| Dataset | Frames | Angle | License | Utilisable ? |
|---|---|---|---|---|
| yastrebksv train set | 8841 @1280×720, 14kpt | **broadcast YouTube** (domaine qui collapse) | **NONE** = blocker SaaS | ❌ mauvais angle + licence |
| Roboflow Universe (desertsloth1 2524, Pei Ling 535, EpuResearchSpace 403, …) | 191–2524 | mixte, surtout broadcast, minorité amateur | per-dataset (souvent CC-BY) | ⚠️ **auditer pour frames obliques** |
| Gholamreza HF | court kpts | surtout broadcast | à vérifier (HF) | ⚠️ probe |

**Combien de frames** (jugement d'ingénierie, **non sourcé**) : RF-DETR fine-tune depuis un prior DINOv2+COCO fort → **centaines, pas dizaines de milliers**. Floor ~300–500, robuste ~800–1500. Court rigide+planaire = efficacité de label énorme : **labelliser 1 frame par clip statique**, propager via ta propre homographie (pseudo-labels) ⚠️ **valable caméra STATIQUE seulement** — casse dès que la caméra bouge (amateur handheld, une part réelle de la cible). **Coût labeling** (estimations, non faits) : 14 kpts/frame ~30–60s → 1000 frames ≈ 8–17 h-personne (1–2 j) ou ~$150–450 outsourcé.

**Synthetic** : warp d'homographie du template ITF top-down sur la variété de viewpoints obliques + domain randomization → labels gratuits (recette prouvée par Amazon sports-field / Rink-Agnostic / ML6, **mais le multiplicateur "2–3×" est non supporté** → "réduit matériellement le labeling réel"). ~1 jour d'ingé, à faire en parallèle.

### 5.3 Coût GPU (réaliste, RunPod/Vast mi-2026)

L4 24GB $0.39/h (community), L40S $0.99/h, 4090 $0.69/h, A100 $1.39–1.49/h. Fine-tune ~1000 frames, 576², 20–50 epochs, 1–3h → **L4 ≈ $1.20**, +3–4 re-runs ≈ $5–15, **all-in incl. re-train churn = $15–60. Le GPU est du bruit.** Deploy : local-first MPS (contingent au smoke-test) ; web-tier serverless CUDA (Modal/Replicate, court solve = fractions de cent/clip) **sidestep le risque MPS entièrement**.

### 5.4 STOP-gate MPS (FAIRE EN PREMIER, ~1h, $0, AVANT toute donnée)

```
pip install rfdetr==1.8.2
model = RFDETRKeypointPreview()        # classe directe, PAS get_model()
model.model.to('mps')                  # vérifier mps, pas cpu
# warmup 5 frames, timer 30 frames @576² sur vraie frame repo 1080p
```
**STOP (→ tuer la voie RF-DETR, pivoter TrackNet) si** : (1) ne `.to('mps')` pas / op non-implémentée non résolue par `PYTORCH_ENABLE_MPS_FALLBACK=1` ; (2) avec fallback=1, une op tourne silencieusement sur CPU (`next(model.parameters()).device` + wall-clock) ; (3) **>~400 ms/frame** = signature CPU-fallback #427. **Barre PASS heuristique (non mesurée) : <150 ms/frame @576².** Note : la calibration court statique tourne **une fois par clip**, donc même 300ms est *utilisable* — le vrai job du smoke-test est de détecter le **mode CPU-fallback**, pas le temps-réel. **Ne labelliser aucune frame avant que le Gate 1 passe.**

---

## 6. Approche 3 — Hybride (classique seeds + modèle pour les cas durs)

Composition recommandée, par ordre de coût croissant et de risque décroissant :

1. **Classique seul** (Approche 1) sur tout. Émet `high`/`low`/`fallback`.
2. **DeepLSD/M-LSD (pré-entraîné, $0)** remplace HoughLinesP dans le front-end quand le masque classique est pauvre (lignes manquantes) — *même pipeline VP/fit downstream*, juste un meilleur extracteur de lignes. Aide Djokovic (lignes abondantes), n'aide pas felix.
3. **Modèle fine-tuné (TrackNet/RF-DETR)** appelé **seulement sur les cas où le classique s'abstient OU émet `low`** — c'est-à-dire la classe court-level lâche-VP (Djokovic) et les amateurs durs. Pas sur tennis (keypoint gagne) ni felix (rien ne sauve le grazing automatiquement).

**Arbitrage de confiance** : chaque source émet le même dict `{H, confidence, ...}` ; la cascade existante `:722-755` branche déjà sur `fallback` vs non. L'hybride n'ajoute que des *niveaux* :
```
keypoint(high) → classique-auto(high) → modèle-finetune(high/low) → sidecar manuel → scalar
```
Le **warp-back self-score (§4.4) est l'arbitre commun** : quelle que soit la source, on rend H_inv et on mesure l'accord au masque blanc ; on garde le H de plus haut SCORE qui passe le veto conditioning. Cela rend les confidences de sources hétérogènes **comparables sur une même échelle physique** (px d'écart au wireframe réel), sans GT.

> Note câblage (la critique l'impose) : dans l'ordre prod, le keypoint tourne *en premier* ; sur tennis il retourne HIGH ⇒ **le classique n'est jamais atteint sur tennis**. Le "cross-check classique vs keypoint sur tennis" ne peut donc se faire qu'en **forçant** le classique (test unitaire isolé), pas dans le chemin prod. Le documenter, ou faire tourner le classique *avant* le keypoint uniquement pour le cross-check de validation.

---

## 7. Licence — l'audit full-stack (vérifié contre les LICENSE réels)

**Bottom line** : `yastrebksv/TennisCourtDetector` ship **AUCUNE LICENSE** (re-confirmé : pas de LICENSE, pas de README, pas de SPDX) → "tous droits réservés" par défaut → **blocker SaaS dur**. Remplacement clean : **RF-DETR Keypoint (Apache-2.0 code+weights)** fine-tuné sur données court commercialement-propres, avec **voie classique 100% permissive** (OpenCV Apache + LSD-MIT restauré) en interim no-GPU.

| Composant | Rôle | License | SaaS-OK | Note (vérifiée) |
|---|---|---|---|---|
| **yastrebksv/TennisCourtDetector** | keypoints court actuels | **NONE** | ❌ **BLOCKER** | pas de LICENSE/README/SPDX. Usage **interne eval-only** OK (jamais redistribué/servi/dockerisé). |
| yastrebksv/TrackNet (arch source) | impl TrackNet | **NONE** | ❌ | même problème propagé. |
| **RF-DETR (rfdetr core)** | remplaçant det/kpt | **Apache-2.0** | ✅ | "Copyright 2025 Roboflow, Apache 2.0", pas de carve-out poids nano→large. |
| **RF-DETR Keypoint (Preview)** | **le remplaçant** | **Apache-2.0 (code+weights)** | ✅ | skeletons arbitraires → 14-pt ITF. Commercial-OK, no copyleft, closed-source-deployable. |
| RF-DETR-XL/2XL (`rfdetr_plus`) | variantes larges | **PML-1.0** (*Platform Model License* Roboflow : acceptation de termes + compte Roboflow) | ⚠️ éviter | rester nano→large pour garder Apache. |
| **DINOv2-with-registers** (backbone RF-DETR) | ViT backbone | **Apache-2.0** | ✅ | Meta relicencé CC-BY-NC→Apache-2.0 **en août 2023** (pas de jour précis sourcé). ⚠️ **métadonnées PyPI périmées (CC-BY-NC)** — épingler les poids GitHub. |
| DINOv3 | backbone récent | **Custom (DOB + approval)** | ❌ éviter | **commercial-permis mais legal-review-required** + gate DOB/approval (PAS flatement non-commercial). Ne pas pull. |
| **OpenCV (≥4.5)** | cœur classique | **Apache-2.0** | ✅ | relicencé BSD→Apache en 4.5.0. |
| cv2 LSD | détection lignes | **MIT** (restauré) | ✅ | retiré 4.1.0–4.5.3 (conflit GPL NFA), restauré 4.5.4+ (NFA re-MIT par von Gioi). |
| **HoughLinesP / FLD / top-hat** | lignes classiques (sondes actuelles) | **Apache-2.0** (OpenCV core) | ✅ | jamais eu le problème LSD. **Stack classique mesuré = 100% clean.** |
| NumPy | math | **BSD-3-Clause** | ✅ | — |
| rtmlib / RTMPose / RTMW / MMPose | pose (déjà utilisé) | **Apache-2.0** | ✅ | déjà dans le stack. |
| WASB-SBDT | ball tracking | **MIT** (`vendor/wasb/LICENSE.md`) | ✅ | (datasets broadcast référencés = termes propres, ne pas redistribuer). |
| M-LSD | line segments (option GPU) | **Apache-2.0** | ✅ | navervision. |
| DeepLSD | line refinement (option GPU) | **MIT** | ✅ | cvg, © Rémi Pautrat 2022, commercial-OK. |
| gchlebus/tennis-court-detection | réf. Farin C++ | **BSD-3-Clause** | ✅ | template de structure permissif. |
| **PnLCalib** | registration field | **GPL-2.0** | ⚠️ **idée-seule** | copyleft. Ne pas linker/vendoriser ; ré-implémenter l'objectif point+line (math non breveté). |
| SoccerNet sn-calibration | baselines calib | **NDA / research-only** (data) | ❌ | ne pas entraîner de poids commerciaux dessus. |
| **Ultralytics YOLO / YOLO-pose** | court via pose | **AGPL-3.0** | ❌ **TRAP** | AGPL s'applique au SaaS/network ; poids fine-tunés = dérivés. ⚠️ **Obligation EXISTANTE** : le produit ship déjà YOLOv8-pose+YOLOv8m, et `Davidsv/CourtSide-Computer-Vision-v1` (YOLO11n via `models/yolo_detector.py:59`, *vérifier l'arch sur la carte HF*) est AGPL-taint — item d'audit séparé, pas une considération RF-DETR-only. |
| Roboflow Universe court datasets | données train | surtout **CC-BY-4.0**, certains **CC-BY-NC/unstated** | ⚠️ per-dataset | CC-BY-4.0 = commercial-OK+attribution ; CC-BY-NC = ❌ taint les poids. Enregistrer la string de licence exacte par dataset. |

**Voie clean-room permissive** : Tier A (no-GPU, ship now) = classique OpenCV-Apache/LSD-MIT/Numpy-BSD ; Tier B (plafond précision) = RF-DETR keypoint Apache fine-tuné sur **données auto-labellisées** (les tiennes, possédées outright). **Hard avoids** : Ultralytics YOLO-pose (AGPL — retirer aussi le modèle court YOLO11n pour purger le taint), DINOv3, code PnLCalib, toute donnée CC-BY-NC/SoccerNet/NDA. **La donnée-licence propage aux poids** : avant tout fine-tune, enregistrer la string de licence de chaque dataset + garder les attributions dans la model card.

> ⚠️ Le **long pole Tier-B n'est pas le GPU mais le dataset oblique labellisé clean** : aucun n'existe (les sets broadcast sont mauvais-angle ET license-taint). C'est la dépendance gating, pas un footnote.

---

## 8. Plan de validation

### 8.1 Ground Truth (hand-clické — la GT NE PEUT PAS être bootstrappée du keypoint mort)

GT en **`tests/fixtures/court/<clip>.court.json`**, byte-compatible `calibrations/<name>.court.json` (consommé par `load_calibration`/`homography_from_points` sans code neuf), étendu de 3 champs additifs :
```json
{ "video":"felix.mp4", "frame_wh":[1920,1080], "frame_index":0,
  "difficulty":"grazing", "annotator":"davidsv",
  "license":"repo-internal eval only", "source_url":"",
  "points": { "far_bl_left":[...], "far_bl_right":[...], "near_bl_left":[...],
              "near_bl_right":[...], "far_svc_center":[...], "near_svc_center":[...],
              "net_left":[...], "net_right":[...] },
  "point_quality": { "near_bl_right":"estimated" } }
```
- `difficulty ∈ {broadcast, elevated_oblique, court_level, grazing}` → drive le **split métrique par classe**. tennis=`elevated_oblique`, Djokovic=`court_level`, felix=`grazing`.
- `frame_index` explicite (felix utilise implicitement frame 0) → reproductibilité du cache (§8.4).
- **Inventaire de départ réel de felix (vérifié)** : `{far_bl_left, far_bl_right, near_bl_left, far_svc_center}` = **3 coins doubles + 1 marque service**, manque `near_bl_right` (near-right occlus). La GT **ajoute `near_bl_right`** (meilleure estimation, taggé `point_quality:"estimated"`). Le sidecar prod reste tel quel (c'est l'*input produit*) ; la GT est la *vérité*, un superset.
- **6–8 points/clip** : 4 coins doubles (obligatoires) + ≥2 mid-court (`far/near_svc_center`, `net_left/right`) qui **étendent l'axe profondeur** pour mesurer l'erreur far-field *indépendamment des coins du solve*.

### 8.2 Métriques (évaluateur `tools/court_eval/eval_homography.py`, pur comparateur, no-GPU/no-video, mirror `eval_bounces.py`)

Le candidat émet `{H (img→m), H_inv (m→px), confidence, rms_px, n_used}`.

**(a) Erreur de reprojection des coins** — projeter les 4 coins-monde via H_inv, comparer aux clics GT. `corner_rms_px`, `corner_rms_pct = rms/diag*100`, + `corner_max_px` (les échecs grazing se cachent dans un seul coin far). PASS ≤0.5%, MARGINAL ≤1.2%, FAIL sinon. ⚠️ Le seuil 0.5% est **ancré faiblement** : le 4.6px keypoint est un *résidu de fit LMEDS sur ses propres 14 points*, PAS une erreur vs GT indépendante (4.6/2203=0.21% est arithmétiquement correct mais n'est pas une preuve d'atteignabilité sur la classe cible) → traiter comme cible, pas garantie.

**(b) Erreur métrique / mètres-court — la métrique PRIMAIRE** (c'est de quoi speed/depth dépendent). Projeter les **clics GT** via H, comparer des distances ITF connues (angle-invariantes) :
```
baseline_far/near  = ‖P(*_bl_left) − P(*_bl_right)‖   truth = 10.97 m
sideline_left/right= ‖P(far_bl_*)  − P(near_bl_*)‖    truth = 23.77 m
svc_to_svc         = ‖P(near_svc_center) − P(far_svc_center)‖   truth = 12.80 m
```
⚠️ **BUG CORRIGÉ (critique C1)** : la distance `net_to_far_svc = 6.40 m` du spec est **FAUSSE**. `net_left/right = (±5.485, 0)`, `far_svc_center = (0, +6.40)` → distance réelle `hypot(5.485, 6.40) = 8.43 m`, pas 6.40. 6.40 ne vaut que depuis `net_center (0,0)` **qui n'est pas un landmark du dict**. ⇒ utiliser `svc_to_svc = 12.80 m` (ou ajouter un landmark `net_center`, ou corriger la vérité à 8.43 m). Tel qu'écrit, ce check inflerait chaque score de ~24%.

`metric_err_pct = mean(err_pct)`. PASS ≤3% mean ET ≤5% max (3% ≈ ±5 km/h sur un service 160 km/h). **C'est la cible d'opération du gate.**

**(c) Warp-back IoU / Chamfer (no-GT self-score = aussi le gate live)** — projeter le polygone court ITF via H_inv, comparer au polygone GT (offline) ou au **masque ligne-blanche détecté** (live gate, no-GT). ⚠️ **IoU retiré comme pass-gate** (critique M5) : sur grazing un H métriquement cassé score IoU 0.92 et passerait → **Chamfer (px) seul est discriminant** ; IoU reste descriptif. PASS Chamfer à calibrer.

**(d) Split far/near (obligatoire grazing)** — splitter (a)(b) par moitié de court (Y<0 near, Y>0 far). Reporter le **ratio far/near** : felix attendu >>3, tennis ~1. Le ratio est lui-même un feature de gate.

**(e) Protocole solve-honnête** — un fit 4-pt est *exact à ses points de contrôle* → corner-RMS=0 si on score les mêmes coins qu'on solve. ⇒ **leave-one-out** si ≥5 points (solve all-but-one, score le held-out, rotate). Si exactement 4 points : corner-RMS non informatif → retomber sur (b)+(c). ⚠️ Le re-solve LOO **doit mirrorer `homography_from_points` exactement** : `cv2.findHomography(..., cv2.RANSAC, ransacReprojThreshold=0.10)` — **0.10 en MÈTRES** (vérifié `court_calibrator.py:93`), pas en pixels, sinon le H ne matche pas le produit.

### 8.3 Calibration du gate

Feature-vector (tout calculable à l'inférence, no-GT) : `warpback_chamfer_px` (primaire), `aspect`/`x_span/y_span` (de `_assess_conditioning`), `farnear_ratio` (ratio px far/near baseline vs perspective attendue), `n_lines_oblique_family` (felix=0→LOW instantané), `vp_spread_px` (tennis 7, Djokovic 700–850, felix irrécupérable).

Procédure : pour chaque clip → label vrai `good = (metric_err_pct ≤ 3%)` (utilise GT, seulement pour calibrer) → sweep τ sur `warpback_chamfer_px` → table de confusion (le **false-accept = on a menti sur la vitesse = le dangereux**) → choisir τ le plus permissif tel que **précision-de-HIGH ≥ 0.95**. **Vetos hard AND-és** (n'importe lequel → force LOW) : `n_lines_oblique_family==0`, `farnear_ratio>3`, `aspect<0.18`, quad-non-convexe.

> ⚠️ Honnêteté statistique (critique U1/U3/M2) : **avec 3 clips et ~6 features, on NE PEUT PAS établir un point d'opération à 0.95-précision** (ROC à 3 points). Le "≥0.95 gate-HIGH precision" est une **CIBLE, pas une garantie certifiée**. Djokovic→HIGH est **TBD / la question ouverte**, pas un win présumé (VP 32–39% diag = faible). felix metric "~8+ far" = **estimation non mesurée** (pas de GT métrique committé). Et le veto grazing repose **dangereusement sur le seul `oblique_family==0`** : une ligne oblique parasite (joueur/ombre/panneau) le retourne → ajouter `farnear_ratio>3` + `aspect<0.18` comme **confirmations indépendantes** et tester que felix abstient même avec bruit oblique injecté. Les 3–5 clips amateurs sourcés (§8.5) sont ce qui rend le 0.95 signifiant.

### 8.4 Test de régression rapide (`tests/test_court_regression.py`, mirror `test_bounce_regression.py`, <2s, no-torch/no-VideoCapture)

Cache committé = sorties HoughLinesP+masque déjà mesurées cette session, figées (`cache/<clip>_lines.json` : `{width, height, frame_index, white_line_segments, vp_candidates}`), pour exercer **fit+gate** sans re-Hough ni décoder. Stocker seulement le **line-cache dérivé** pour Djokovic (pas la frame, droits source) ; frames felix/tennis OK (les nôtres).
```python
assert tennis.metric_err_pct   <= 3.0     # solvable: reste résolu
assert djokovic.metric_err_pct <= 6.0     # cible: floor marginal-or-better
assert tennis.corner_rms_pct   <= 0.6
assert felix.gate == "LOW"                # LA ligne la plus importante: ne jamais mentir sur le grazing
false_accepts = sum(1 for c in clips if c.gate=="HIGH" and c.metric_err_pct > 3.0)
assert false_accepts == 0
```
Floors fixés *sous* les premières valeurs mesurées (philosophie `F1_FLOOR=0.72`). ⚠️ **`tennis.gate=="HIGH"` est un positive-control du chemin CLASSIQUE en isolation, PAS une assertion prod** : en prod le keypoint gagne tennis en premier (`:722-755`), le gate classique n'y est jamais exercé.

### 8.5 Sourcing de clips obliques diversifiés (clean/cheap)

Critère d'échantillonnage = clips où keypoint=FALLBACK mais court complet (court-level/behind-baseline). Cible 3–5 : 2× behind-baseline court-level (idéal SaaS), 1× side-grazing (grossir la classe abstain), 1–2× phone-tripod. Source CC/public-domain ou auto-filmé (idéal, zéro droits). **Committer seulement les artefacts dérivés** (`*.court.json` + `*_lines.json`), jamais les frames brutes de clips tiers. Labeling via `tools/calibrate_court.py` (~60s/clip). QA pré-commit : `homography_from_points` sur la GT doit s'auto-rapporter `confidence ≥ medium` (ou `low` documenté pour les négatifs-gate intentionnels).

---

## 9. Recommandation classée + découpage en chantiers

### 9.1 Recommandation classée

1. **Approche 1 (classique no-GPU) D'ABORD** — license-clean, $0, déployable, remplace le keypoint mort sur la cible. *Caveat honnête* : win sur tennis/amateur-élevé, **TBD/LOW sur Djokovic court-level (VP lâche)**, **abstain sur felix grazing**.
2. **DeepLSD/M-LSD ($0, pré-entraîné)** — upgrade SOTA gratuit de l'extracteur de lignes du (1) pour Djokovic ; à spiker cette semaine.
3. **Pilote GPU RF-DETR/TrackNet** — *gated* derrière le smoke-test MPS ; plafond de précision ; dépense = données, pas compute ; si MPS échoue → **TrackNet-sur-tes-données** (drop-in, MPS-prouvé, license-clean) plutôt que se battre avec RF-DETR.
4. **Calibration manuelle sidecar** — reste la voie pour felix-grazing et tout `fallback`, déjà construite (`_assess_conditioning` flague déjà le grazing).

### 9.2 Chantiers ordonnés (chacun testable, sur les 3 clips-sondes ; effort = jugement, non mesuré)

- **C1 — Masque ligne-blanche** `vision/court_lines.py::white_line_mask(frame)→mask` (§4.1 : percentile bright+ridge + test couleur 7×7 + retinex). *Test* : dump PNG ; Djokovic≈wireframe quasi-complet, felix≈horizontal. *Effort : S.*
- **C2 — Extraction segments + fusion** `detect_line_candidates(mask)→[Line]` (HoughLinesP, fallback ; FLD si contrib dispo — **ajouter le caveat dépendance opencv-contrib + floor 4.5.4 en item**). *Test* : counts dans les plages mesurées ; chaque candidat porte support + endpoints sous-pixel. *Effort : S-M.*
- **C3 — Familles + VP RANSAC + gate d'abstention** `estimate_vps(candidates)→(vpA,vpB,families)|ABSTAIN` (§4.3). *Test* : tennis spread≈7px, Djokovic présent-lâche, **felix→ABSTAIN (famille-count)**. **C'est le test de sûreté felix — doit passer avant qu'aucun H ne soit émis.** *Effort : M.*
- **C4 — Correspondance + warp-back self-score** `fit_court(families,vps,mask)→(H,score,corners)` (§4.4, VP-pruned + Chamfer + veto `_assess_conditioning`). ⚠️ **Le vrai cœur dur = l'association ligne→ligne-ITF-nommée** (quelle Hough est quelle ligne ITF), pas le fit ; ne pas l'assumer résolu. *Test* : sur tennis, auto-H ≈ keypoint-H (oracle, *forcer* le classique). *Effort : L.*
- **C5 — Coins auto + émission dict + sidecar** `AutoCourtDetector.compute(frame)→dict` (§4.6 : 4 coins nommés → `homography_from_points`, `save_calibration`, dict standard). *Test* : dict byte-compatible `load_calibration` ; re-run caché. *Effort : S-M.*
- **C6 — Câblage pipeline** insérer entre `:738`–`:745` derrière `--auto-court` (**default OFF**, discipline `--event-methodo`). *Test* : OFF = byte-identique ; ON, Djokovic obtient un H métrique là où il tombait scalar, felix charge son sidecar (auto a abstain), tennis inchangé. + **régression rapide §8.4**. *Effort : S.*
- **C-VAL — Harness validation** (parallélisable) : GT fixtures §8.1, `tools/court_eval/eval_homography.py` §8.2 (**avec le fix distance 12.80m, pas 6.40**), `tests/test_court_regression.py` §8.4. *Effort : M.*
- **C-GPU0 — Smoke-test MPS RF-DETR** (§5.4, 1h, $0, **avant toute donnée**). STOP-gate. *Effort : XS.*

**Séquençage** : C1→C3 *prouvent le gate d'abstention avant qu'aucun H ne soit émis* (sûreté felix d'abord) ; C4–C5 réutilisent `court_calibrator` verbatim ; C6 derrière flag default-OFF (jamais de régression du legacy byte-identique). C-GPU0 indépendant, à lancer en parallèle. **Ce document est research-only : aucun code prod n'est écrit.**

---

## 10. Risques

- **Grazing (felix) insoluble par le classique** — 1 seule famille d'orientation, axe profondeur inobservable. **Aucune quantité de lignes ne le résout** ; nécessite >4 points manuels (sidecar existant) ou un modèle (et même le modèle keypoint donne 1–3/14). Documenté honnêtement : felix reste sur la calibration manuelle, le gate doit l'**abstain** (`fallback`). Le flagship local-test échoue Tier-A — c'est *correct*, pas un bug.
- **Gate grazing = single point of failure** — le veto `oblique_family==0` fait tout le travail ; une ligne oblique parasite le retourne. Mitigation : vetos indépendants (`farnear_ratio>3`, `aspect<0.18`) + test avec bruit oblique injecté.
- **Licence** — yastrebksv (NONE) = blocker SaaS ; YOLO/Ultralytics (AGPL) = trap **déjà actif** dans le produit (pose + modèle court YOLO11n à retirer) ; DINOv3/PnLCalib-code/CC-BY-NC/SoccerNet = à éviter. Le **long pole Tier-B = dataset oblique labellisé clean** (inexistant).
- **Généralisation** — tout repose sur 3 clips-sondes ; les pourcentages de succès sont des **cibles non mesurées**. Les 3–5 clips amateurs (§8.5) sont nécessaires pour que le gate-precision ≥0.95 ait un sens.
- **Blow-up métrique far-field** — sur VP lâche (Djokovic 857/702px = 32–39% diag), l'erreur en champ lointain explose ; le split far/near (§8.2d) et le ratio>3 le détectent ; possible que Djokovic ne dépasse jamais LOW.
- **Flip 180° / ambiguïté gauche-droite** — rectangle symétrique ; résolu par max-SCORE warp-back + net-à-Y=0 + near-baseline-plus-bas + côté joueur/scoreboard. Reste un risque résiduel sur courts parfaitement symétriques sans cue d'asymétrie.
- **Caméra mobile / handheld** — toute l'économie (court solve une fois/clip, pseudo-label propagation, latence tolérable) **assume une caméra STATIQUE (tripod)**. Le behind-baseline amateur peut être handheld → cas **non résolu**, à carve-out explicitement.
- **MPS RF-DETR (preview churn)** — #427 *corrigé* mais à re-vérifier en 1.8.2 ; l'API/poids preview bougent → épingler `rfdetr==1.8.2`, budgéter un re-train.
