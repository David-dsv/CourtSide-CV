# Distinguer REBOND et FRAPPE par la direction de déplacement de la balle

> **TL;DR — décision PM.** À implémenter en #1 : **A3 ⊕ A4 fusionnées** = veto vx-flip *one-directional* (ne retire qu'un candidat dont la composante horizontale s'inverse, sinon s'abstient), **deadband relatif** à la vitesse propre de la balle (échelle-libre), **CONFIRM exigeant aussi le vy-flip**. **Bloqué sur le CHANTIER 0** (exposer les centres tracker *pré-spline* + un masque d'interpolation ; ne jamais lire la direction sur le tableau re-lissé du détecteur). Gain rebond mesuré **sur la production** : **F1 0.811 → 0.882** (run 1280×720) / **0.865 → 0.914** (run natif 1080p), **0 vrai rebond perdu**, en tuant les FP de frappe `f820`/`f1609` (et `f105` au format 1280×720). C'est un **filtre de précision sur le chemin dense (WASB)**, pas un détecteur universel.

> **Session RESEARCH-ONLY** (aucun fichier de production modifié). Toutes les valeurs ci-dessous sont **reproduites** sur les caches commités `felix` (WASB dense, 60 fps, `tests/fixtures/cache/felix_wasb_centers.json`, 1578/1920 frames détectées) et `demo3` (pose verrouillée, 50 fps, `tests/fixtures/cache/demo3_pose.json`) — probes jetables exécutées sous `python3` (numpy/scipy), aucune dépendance GPU/vidéo. Branche `feat/accuracy-overhaul`.

---

## 1. Diagnostic — le vrai problème est en DÉTECTION, pas en affichage

Le correctif précédent (formes orthogonales rebond vs frappe dans l'overlay) était **purement cosmétique** : il a séparé visuellement deux marqueurs, mais l'erreur sous-jacente persiste — **le détecteur de rebonds émet de vraies frappes raquette comme si c'étaient des rebonds de sol**. La confusion est *upstream*, dans la couche de décision, pas dans le rendu. Les mesures de §3 le prouvent : sur la production `detect_bounces_robust`, parmi les faux positifs de rebond figurent `f820` et `f1609`, dont la balle **inverse sa direction horizontale à ~180°** (signature physique d'un renvoi raquette, jamais d'un rebond).

### État actuel du code

**Détection de REBOND** — source = trajectoire de la balle uniquement :
- `vision/bounce.py:112` `detect_bounces_from_trajectory` (chemin Kalman / curve-fit, défaut sur trajectoire éparse). **Retourne uniquement `chosen`** (l.236) — aucun tableau de centres exposé.
- `vision/bounce.py:484` `detect_bounces_robust` (chemin WASB / dense). Logique : despike → re-lissage → maximum local de y + ajustements balistiques descente/montée + **garde de restitution d'énergie déjà présente** : `hit_ratio_max=1.8` (l.501) rejette les candidats dont la remontée est > 1.8× la descente (une frappe injecte de l'énergie → remontée raide → rejetée) ; `min_v_excursion_frac=0.055` (l.506) rejette les micro-oscillations. **Retourne `chosen, re_smoothed, n_despiked`** (l.613).
- `vision/bounce.py:348` `despike_trajectory`, `vision/bounce.py:442` `_robust_fit`, `vision/bounce.py:20` `smooth_ball_trajectory`.

**Détection de FRAPPE / HIT** — source = squelette (pose) :
- `vision/shots.py:200` `detect_hits` : pic de **vitesse du poignet-raquette** (kps COCO 9/10), filtré par FWHM (`fwhm_min = max(3, int(fps*0.06))`, l.221 — un vrai swing fait 4-8 frames, le jitter 1-2) + proximité balle↔joueur (`best_d / h > 2.5` rejette, l.272).
- `vision/shots.py:128` `_wrist_speed_series`, `vision/shots.py:184` `_fwhm_at`.

**Les deux détecteurs sont INDÉPENDANTS.** La seule protection croisée est une **garde temporelle aveugle** : `bounce_guard = int(fps * 0.10)` (`vision/shots.py:214`), appliquée à `vision/shots.py:290` — elle supprime une frappe trop proche d'un rebond connu (±0.10 s), sans aucune notion de physique.

### Le signal manquant

**AUCUNE logique de changement de direction n'existe.** Le filtre de Kalman calcule bien `(vx, vy)` puis **les jette** : seule la norme (vitesse km/h) est conservée — `run_pipeline_8s.py:1015-1017` :

```python
vx = float(ball_kf.kf.statePost[2][0])
vy = float(ball_kf.kf.statePost[3][0])
speed_px = np.hypot(vx, vy)   # ← seule la norme survit ; vx, vy signés perdus
```

L'ancienne règle « vy descendant→montant » a été abandonnée car le spline la lissait (fragile). **Appels rebonds** : `run_pipeline_8s.py:1100-1105` (robust sur chemin WASB, l.1104 curve-fit sinon). **Appels frappes** : `run_pipeline_8s.py:1270-1295`.

### La physique à exploiter (intuition utilisateur, mesurée)

| Événement | Composante horizontale (vx) | Composante verticale (vy) |
|---|---|---|
| **REBOND sol** | **signe PRÉSERVÉ** (la balle continue de traverser dans le même sens) | **FLIP** (+→− : descendante puis montante, repères image) |
| **FRAPPE raquette** | **signe INVERSÉ / dévié fort** (le joueur renvoie) | change aussi |

Le discriminant central est donc **le signe de la composante horizontale `vx`**, que la production calcule puis jette (l.1015-1017). C'est exactement le levier sous-exploité. **Mesure de cadrage (W=7 frames ≈ 117 ms) :** l'angle pré/post du vecteur vitesse aux rebonds GT s'étale d'environ **28–30° à ~140–157°** (la borne exacte dépend de la méthode d'estimation de vitesse : mean-of-deltas vs pente moindres carrés) — l'angle *seul* ne sépare donc PAS proprement (un drop-shot rebondit à ~30°, un renvoi croisé frappe à >150°). Le séparateur propre est **le signe de vx combiné au flip de vy**, pas l'amplitude de l'angle.

---

## 2. Inventaire des signaux disponibles au moment de la détection

Pour chaque signal : d'où il vient, fiabilité **broadcast** vs **amateur**, et ce que les probes disent réellement.

### 2.1 Vitesse signée `(vx, vy)` recalculée par moindres carrés sur la trajectoire — LE signal clé

C'est le signal porteur. Il faut le **recalculer** (la production le jette, l.1015-1017), trivialement, par pente des moindres carrés de `x(t)`/`y(t)` sur une fenêtre de points réels.

**Signature directionnelle aux 16 rebonds GT de `felix` (fenêtre ±7 frames), reproduite :**

| Substrat | mesurables | vx-signe PRÉSERVÉ | vy-FLIP | vy-FLIP @ négatifs (mi-vol) |
|---|---|---|---|---|
| **WASB dense brut** (1578/1920) | **15/16** | **15/15 (100 %)** | **15/15 (100 %)** | ~5-6 % |
| WASB dense + spline | 15/16 | 14/15 | 15/15 | ~10 % |
| Kalman épars brut (`felix_dets`) | — | ~80 % | ~73 % | ~8 % |
| Kalman épars + spline | — | ~88 % | ~62 % | ~9 % |

> **1/16 rebond GT est irréductiblement occlus : `f424`** (trous WASB des deux côtés → < 2 points réels par fenêtre → **abstention**). **Les 15 autres sont mesurables**, y compris `f978` (vx +2.72→+0.92 PRÉSERVÉ, vy +3.00→−3.45 FLIP) qui compte dans le 15/15. Toute logique de direction doit dégrader gracieusement sur `f424` (pas de données → pas de verdict → repli y-max + énergie).

**Verdict :** le discriminant est **porté par {vx-préservé ET vy-flip} conjointement** — car vx-signe est préservé ~90 % du temps **aussi en mi-vol** (la balle garde trivialement son sens horizontal), donc vx seul ne sépare pas un rebond d'un point quelconque de la trajectoire ; et vy-flip seul classe « rebond » des frappes qui réfléchissent aussi verticalement (cf. `f820`/`f1609`, vy-flip présent). La règle composée `{vx-préservé ET vy-flip}` = **100 % rebond / ~1 % mi-vol** sur WASB dense. **MAIS** la puissance de `vx-signe` comme *rejeteur de frappe* ne se réalise que **contre de vraies frappes** (où vx s'inverse). Faute de **GT de frappes sur felix**, elle est prouvée **indirectement** par les 3 FP-frappes capturés en §3 — voir le fil consolidé en encadré §2.3.

- **Broadcast (felix WASB) : FIABLE.** Fenêtre optimale **W = round(fps×0.08…0.12) = 5-7 frames** (50–117 ms : 15/15 stable). À **W=9 (150 ms) → 14/15**, **W=12 (200 ms) → 13/15** : une fenêtre trop large brouille la réflexion et érode vx-préservé.
- **Amateur/épars (Kalman) : MARGINAL.** vy-flip tombe à 62-73 %, `{vx∧vy}` autour de 53-67 % — Kalman+spline sur-lissent la réflexion verticale (la fragilité même qui a tué l'ancienne règle vy). Utilisable comme **indice confirmant**, jamais autoportant.

### 2.2 Trajectoire LISSÉE (spline) vs BRUTE (centres tracker pré-spline)

**Le spline DÉTRUIT le signal de frappe exactement là où il compte.** Aux rebonds GT, WASB-brut pré-spline = **15/15 vx-préservé + 15/15 vy-flip**, alors que WASB-spline **dégrade vx-préservé à 14/15** : le lissage cubique « arrondit » la réflexion, en particulier quand elle tombe sur un trou comblé.

Le chemin « raw highest-score » (max-confiance sans tracker) est **inexploitable** : épinglé sur des FP de logo de court statiques. **Un tracker est obligatoire** pour ne serait-ce que choisir la balle. Le cadrage « brut vs lissé » de l'utilisateur doit donc se lire **« centres tracker pré-spline vs post-spline »**.

> ⚠️ **Conséquence directe pour l'implémentation (vérifiée sur source) :** `detect_bounces_robust` despike les centres *déjà splinés* (commentaire l.516 « despike the raw (already spline-smoothed) centers ») puis **re-lisse** (`re_smoothed = smooth_ball_trajectory(...)`, l.521) et **retourne `re_smoothed`** (l.613) — un tableau post-spline où **le masque « point réel vs interpolé » est ÉRASÉ**. De plus `detect_bounces_from_trajectory` ne retourne **aucun** tableau de centres (l.236). **Le contrat de sécurité « ne jamais lire la direction à travers un gap comblé » n'est donc PAS implémentable en lisant la sortie du détecteur** — il faut remonter aux centres tracker bruts AVEC un masque d'interpolation, threadés explicitement (CHANTIER 0).

> *Note de sourçage honnête :* l'affirmation forte « le spline efface activement la signature de frappe » est démontrée **au niveau des rebonds GT** (14/15 vs 15/15 ci-dessus) et par construction (un spline cubique monotonise une inversion à cheval sur un gap). Une statistique précise du type « N inversions de signe effacées sur M gaps, dont la majorité sont des non-rebonds » n'a **pas** été extraite d'une probe commitée ici ; ne pas la citer comme un chiffre dur. Ce qui est **reproduit** : (a) 15/15 brut → 14/15 splinné aux GT, (b) un tracker est indispensable.

### 2.3 Poses (squelettes verrouillés P1/P2) → proximité balle↔joueur

La distance min balle↔poignet, normalisée par la hauteur de boîte du joueur, est un **signal complémentaire** : là où la frappe a lieu, la balle est tout près d'un poignet.

- **Cas dense / proche : fort.** Les frappes se groupent à faible `dist_norm`, les rebonds à grande distance → seuil de l'ordre de `~0.45×hauteur_boîte`.
- **Deux modes d'échec structurels que la direction (vx) NE partage PAS :**
  1. **Rebond au pied d'un joueur** → la proximité crie « frappe » à tort ; la règle `{vx-préservé + vy-flip}` le rattrape correctement en « rebond ».
  2. **Côté lointain/épars** : la balle est souvent **non détectée exactement aux frappes lointaines**, ET la petite boîte du joueur lointain (≈3× plus petite) gonfle la normalisation → signal **manquant ET bruité** là où la détection de rebond est déjà faible.

→ **La proximité couvre le proche-court dense ; la direction couvre le lointain et le rebond-au-pied.** Complémentaires.

> ⚠️ **Sourçage honnête (à corriger vs un brouillon antérieur) :** le cache `demo3_pose.json` ne contient **qu'UN seul rebond** (`bounce_frames = [457]`) et 8 frappes GT (`data/output/tennis_demo3_annotations.json`, `strikes` = 8). Une statistique de séparabilité du type « 92 % séparable sur 9 rebonds GT » **n'est PAS reproductible** depuis ce cache (il n'y a pas 9 rebonds). La proximité reste un argument **structurel** valable (proche-dense fort, lointain faible), mais **sans chiffre de séparabilité chiffré ici** : tout claim quantitatif de proximité exige d'abord un cache multi-rebonds annoté.

> **Fil consolidé « vx-signe prouvé indirectement » (à lire d'un bloc) :** felix n'a **pas** de GT de frappes. La puissance de vx-signe à *rejeter de vraies frappes* est donc établie **uniquement** via les 3 FP de rebond que §3 identifie comme des frappes (`f105`/`f820`/`f1609`, réversions horizontales 153–180°). Ces 3 FP constituent le **bootstrap minimal de GT-frappes** (cf. §5.1). Tant que ce GT ne contient pas **au moins une frappe down-the-line** (vx préservé), le claim « le veto tue les vraies frappes » reste **NON testé pour le cas tueur**.

### 2.4 Filet (`net_y = frame_height*0.47`, hardcodé `vision/shots.py:210`)

**NON discriminant** : les rebonds surviennent des **deux côtés** du filet. **À ne pas utiliser comme séparateur.** La recommandation finale ne s'appuie pas dessus.

### 2.5 Homographie / mètres-par-pixel

Disponible seulement sur broadcast standard (`--homography`) ou clip calibré (`models/court_calibrator.py`). **Utilité ici :** fournir une échelle physique pour exprimer le deadband de vx (voir §6, hardcoding). **Non requise** pour le signal de base, mais **recommandée comme proxy d'échelle** quand présente.

### Synthèse de fiabilité

| Signal | Broadcast dense (WASB) | Amateur/épars (Kalman) |
|---|---|---|
| **vx signe (préservé/flip)** | **FIABLE** (15/15) | marginal (~80 %, abstention fréquente) |
| **vy flip** | **FIABLE** (15/15 vs ~5-6 % mi-vol) | dégradé (62-73 %) |
| Trajectoire pré-spline + masque | requis & disponible (tracker) | requis ; rarement assez de points réels |
| Proximité pose | fort en proche | s'effondre en lointain |
| Filet | non discriminant | non discriminant |
| Homographie | parfois | rarement (sauf calibration) |

---

## 3. Approches concrètes

Cinq approches stress-testées. **Elles convergent toutes sur le même cœur physique** (veto vx-flip) ; elles diffèrent par le **périmètre** et la **sécurité** (one-directional vs re-labellisation bidirectionnelle), pas par la faisabilité brute. Défauts adversariaux repliés.

> **Note de provenance des chiffres (lire avant les scores).** Les F1 ci-dessous sont obtenus en **exécutant la production `detect_bounces_robust`** sur le cache `felix_wasb_centers.json`, puis en scorant contre `felix.bounces.json` avec un matcher greedy 1-1, tolérance `round(0.15*fps)=9` frames (identique à `tools/bounce_eval/bounce_io.py`). **Découverte importante : la composition de la liste de candidats DÉPEND de la résolution passée** (les seuils fractionnaires internes du détecteur tirent à cette échelle). Je rapporte donc les **DEUX** runs :
>
> | Run | Baseline détecteur | FP baseline | Après veto vx-flip | FP survivants | TP perdus |
> |---|---|---|---|---|---|
> | **1280×720** (résolution des probes) | pred=21, TP=15, FN=[1253], **F1=0.811** | `[105, 602, 820, 1050, 1609, 1859]` | drop `[105, 820, 1609]` → TP=15, FP=3, **F1=0.882** | `[602, 1050, 1859]` | **0** |
> | **1080p natif** (felix réel) | pred=21, TP=16, FN=[], **F1=0.865** | `[602, 820, 1050, 1609, 1826]` | drop `[820, 1609]` → TP=16, FP=3, **F1=0.914** | `[602, 1050, 1826]` | **0** |
>
> Commande de reproduction : charger le cache → `detect_bounces_robust(centers, speeds_px, fps=60, frame_height, frame_width)` → matcher greedy vs GT. Le **gain est réel et reproduit sur la production**, mais **plus modeste que des estimations antérieures** (0.811→0.882 / 0.865→0.914, **pas** 0.857→0.938). Les FP horizontalement inversés `f820` (vx −11.61→+11.65, 180°) et `f1609` (vx −16.19→+10.88, 178°) sont tués aux deux résolutions ; `f105` (vx +2.16→−8.83, 153°) n'apparaît comme candidat qu'au format 1280×720. **0 vrai rebond perdu** aux deux résolutions. **À re-valider en 1080p natif** avant tout merge (CHANTIER 5).

### APPROCHE 1 — Direction-Gate minimal (veto vx-flip lu sur l'array du détecteur)

- **Principe :** ajouter un gate additif dans la boucle d'acceptation ; accepter seulement si vx préservé ET vy réfléchit, sinon rejeter. Lit « l'array que le détecteur tient déjà ».
- **⚠️ Défaut :** lire `re_smoothed` (post-spline, masque effacé, l.613) = mesurer la direction sur le substrat qui dégrade le signal aux GT (15/15→14/15) ; et `detect_bounces_from_trajectory` ne retourne aucun array (l.236). **Non par faisabilité intrinsèque** (le CHANTIER 0 rend tout array disponible), **mais par sécurité** : sans le masque réel/interpolé, le gate peut lire à travers un trou comblé. Sa vraie faiblesse face à A3/A4 est **bidirectionnelle / non scoped** (il agit comme un gate dur des deux côtés).
- **Score PM-readiness : 5/10** (différence de **sécurité**, pas de faisabilité — voir §4).

### APPROCHE 2 — Arbitre directionnel complet (veto + arbitrage hit↔bounce + proximité)

- **Principe :** détecteurs = générateurs de candidats ; un arbitre recalcule `(vx,vy)` pré-spline, **re-labellise** chaque candidat, **remplace le `bounce_guard` aveugle** (l.214/290) par un arbitrage physique, et fusionne la proximité comme gate de confirmation des frappes.
- **Robustesse :** dense net-positif ; épars en abstention-lourde (no-op sûr).
- **Coût :** ~40 lignes + 3 sites d'appel + threader les centres pré-spline + masque (CHANTIER 0).
- **⚠️ Défauts :** la **re-labellisation est bidirectionnelle** (elle peut *créer* des rebonds ET en retirer → peut faire baisser la recall, contrairement à A3) ; et l'**arbitrage hit↔bounce est NON VALIDÉ** : demo3 n'a **aucun** chevauchement hit/bounce dans la garde (frappe la plus proche à ~0.40 s de l'unique rebond f457). Net-cord vers l'avant non géré. Hardcoding EPS (voir §6).
- **Maturité : needs-prototype. Score : 5/10.**

### APPROCHE 3 — VX-SIGN Reflection Gate (filtre FP one-directional, scoped) ★

- **Principe :** garder la liste de candidats du détecteur, ajouter **UN** gate de confirmation qui recalcule vx sur la trajectoire **pré-spline** et **REJETTE uniquement les candidats dont vx s'inverse** (= frappe). *One-directional, conservateur* : il ne retire que des vx-flip et **S'ABSTIENT** dès que trop épars/occlus → il **ne peut qu'ajouter de la précision, jamais retirer un vrai rebond**.
- **Pseudo-formule :**
  ```
  half    = round(0.10 * fps)        # ±0.10 s ; 0.08-0.12 s = 15/15 stable ; 0.20 s dégrade (13/15)
  EPS_vx  = frame_width * 0.0004      # ⚠️ MAGIC felix-fit : 0.0004 ré-ingénieré sur felix.
                                      #    À re-dériver cross-clip ; préférer le deadband RELATIF de A4.
  vx_pre  = polyfit pente x(t) sur ≤half points RÉELS dans [f-half..f]   (≥2 requis)
  vx_post = polyfit pente x(t) sur ≤half points RÉELS dans [f..f+half]   (≥2 requis)
  si vx_pre|vx_post None              -> ABSTAIN (garde ; trop épars)        # ex : f424
  sinon si |vx_pre|≤EPS ou |vx_post|≤EPS -> ABSTAIN (horizontal quasi-nul)   # drop-shot lent
  sinon si sign(vx_pre)==sign(vx_post)   -> CONFIRM rebond
  sinon                                  -> REJECT (frappe)                  # f820/f1609
  tag UI : "metric" si CONFIRM, "unconfirmed" si ABSTAIN
  ```
- **Robustesse (vérifiée end-to-end sur la production, voir encadré de provenance) :** **F1 0.811→0.882 (1280×720) / 0.865→0.914 (natif)**, **0/TP perdu**, tue `f820`/`f1609` (+`f105` en 1280×720). Stable 50–117 ms.
- **Coût :** ~30 lignes, un post-filtre ; `numpy.polyfit` ; aucun modèle/pose/homographie.
- **Cas durs — explicitement :**
  - **Smash** : vx dévie fortement + énergie de remontée → REJECT (et la garde `hit_ratio_max=1.8` aide déjà). ✅ géré.
  - **Net-cord (let) latéral** : la balle dévie horizontalement → vx flip → REJECT. ✅ géré pour le cas latéral.
  - **Volley (volée)** : c'est une frappe **sans rebond**. Le gate ne s'applique **qu'aux candidats rebond émis par le détecteur** ; or une volée ne crée pas de y-maximum au sol → **aucun candidat n'existe à cet instant** → **hors périmètre par construction** (le gate ne le voit jamais ; il n'a donc rien à « tuer » ni à confirmer). ✅ non concerné.
  - **Contact occlus** : < 2 points réels → ABSTAIN (ex : `f424`). ✅ dégrade proprement.
  - **Rebond au pied d'un joueur** : la direction le sauve là où la proximité mentirait. ✅ géré.
- **Cas durs NON gérés (faux négatifs assumés) :**
  - **Frappe down-the-line** (vx préservé) → gate aveugle. **Le cas tueur.** Non couvert.
  - **Net-cord vers l'avant** (la balle continue tout droit, vx inchangé) → gate aveugle.
  - **Drop-shot / rebond lointain lent** (vx<EPS) → ABSTAIN (ni gain ni dégât).
  - **FP vx-PRÉSERVÉ** : ex `f602` (glide vx +41→0, **pas** de vy-flip) **survit** — c'est exactement la classe que la greffe (b) « vy-flip requis au CONFIRM » attaque.
- **✅ Verdict :** aucun défaut fatal sur le périmètre revendiqué. Plafond honnête : **filtre FP one-directional sur la liste existante**, gain sur **chemin dense seulement**, validation **mono-clip (felix, 1 surface, 60 fps)** où les FP capturés sont **tous des réversions cross-court** (biais de sélection — voir §6.1).
- **Greffe avant ship :** (a) re-valider EPS_vx + liste de candidats en **1080p natif** ; (b) **ajouter vy-flip REQUIS au CONFIRM** → vise les FP vx-préservés sans réflexion verticale (`f602`/`f1859`/`f1826`/`f1050`) ; ⚠️ **n'adresse PAS down-the-line** (qui a un vrai vy-flip de rebond) — ne pas confondre les deux ; (c) documenter down-the-line + net-cord-avant comme faux-négatifs connus.
- **Maturité : pm-ready. Score : 8/10.**

### APPROCHE 4 — VX-SIGN Arbiter à deadband RELATIF (échelle-libre) ★

- **Principe :** identique au cœur one-directional de A3, mais **remplace le deadband absolu en pixels par un deadband RELATIF à la vitesse locale propre de la balle** → échelle-libre across proche/lointain et toute résolution. Conserve vy-flip comme **co-signal de confiance** (pas un gate dur → le plancher 0.72 n'est jamais érodé par un vy-flip manquant).
- **Pseudo-formule (le delta vs A3) :**
  ```
  half        = round(fps * 0.12)
  speed_scale = median(|x[i+1]-x[i]| sur paires consécutives RÉELLES dans [f-half..f+half])
  eps         = max(0.15 * speed_scale, W * 1e-3 * (60/fps))
                #  ⚠️ 0.15 ET 1e-3 sont des MAGIC felix-provisionnels — à re-dériver cross-clip.
                #  sur felix : speed_scale médian = 9.54 px/f -> eps ≈ 1.43 px/f
                #  (< réversions frappe 8-20 px/f, > bruit) : sépare proprement.
  ABSTAIN si la fenêtre traverse un gap interpolé > round(fps*0.05)
  vx_FLIP -> DROP ; sinon KEEP ; vy_FLIP loggé en CONFIANCE seulement
  ```
- **Robustesse :** **mêmes chiffres vérifiés que A3** (le veto est le même ; seul le seuil change, et `eps≈1.43 px/f` reproduit le même tri sur felix). TP lents lointains tombent sous eps → abstention correcte. **0/TP perdu.**
- **Cas durs :** identiques à A3, **plus** robustesse d'échelle supérieure (le deadband suit la vitesse propre, pas la largeur de frame).
- **✅ Verdict :** aucun défaut fatal. Même plafond mono-clip que A3.
- **Greffe :** mêmes (a)/(b)/(c) que A3 ; le deadband relatif **adresse partiellement** le grief de hardcoding (§6.3) — mais 0.15 et 1e-3 restent à valider cross-clip.
- **Maturité : pm-ready. Score : 8/10.**

### APPROCHE 5 — Direction-Veto avec tags de confiance UI

- **Principe :** A3/A4 + sortie explicite de **niveaux de confiance** dans le badge UI existant (`CONFIRMED` / `unconfirmed`(abstain) / `weak`). Bonne idée — mais **à greffer sur A3/A4**, pas un packaging séparé.
- **⚠️ Si packagée comme « lit l'array du détecteur »**, elle hérite du défaut de sécurité de A1. **Garder l'idée des tags (CHANTIER 2), jeter ce packaging. Score : 5/10.**

### Tableau comparatif

| # | Cœur | Sens du gate | Threading pré-spline+masque (CHANTIER 0) | Deadband | Verdict adversaire | Score |
|---|---|---|---|---|---|---|
| 1 | veto vx-flip (lit array détecteur) | **bidirectionnel / dur** | requis (comme tous) | abs. px | non scoped, lit le spline | 5/10 |
| 2 | arbitre complet + proximité + arbitrage | **bidirectionnel (re-label)** | requis | abs. px | re-label non validé, peut baisser recall | 5/10 |
| **3** | **veto vx-flip one-directional, scoped** | **one-directional + abstention** | requis | abs. px | **pas de défaut fatal** | **8/10** |
| **4** | **idem + deadband RELATIF** | **one-directional + abstention** | requis | **relatif** | **pas de défaut fatal** | **8/10** |
| 5 | veto + tags confiance | one-dir si greffé sur A3/A4 | requis | abs. px | OK comme greffe, KO en standalone | 5/10 |

> **Pourquoi 8/10 vs 5/10 — c'est la SÉCURITÉ, pas la faisabilité.** Le CHANTIER 0 (centres pré-spline + masque) est requis **pour les cinq** : ce n'est donc pas un différenciateur. Le vrai écart est que **A3/A4 sont *one-directional avec abstention*** — ils ne **retirent** un candidat que sur un vx-flip *confiant et mesurable*, et **gardent** (en taguant « unconfirmed ») dans tous les cas douteux → **ils ne peuvent pas faire descendre la recall sous le plancher**. A1/A2/A5(standalone) appliquent un jugement **bidirectionnel/dur** (re-labelliser ou gater des deux côtés) → ils **peuvent** retirer un vrai rebond sur chemin épars (où vx-préservé ~80 % et vy-flip ~62-73 %), donc franchir le plancher 0.72. C'est cet axe de sécurité qui justifie le delta.

---

## 4. Recommandation classée + découpage en chantiers

### Classement (pm-ready → spéculatif)

1. **A3 ⊕ A4 = la recommandation** (8/10 chacune, aucun défaut fatal). Fusionner : **veto one-directional scoped de A3 + deadband relatif échelle-libre de A4 + greffe (b) « vy-flip REQUIS au CONFIRM »**. Seul périmètre où l'adversaire n'a pas trouvé de défaut fatal **et** où le gain dense est **reproduit sur la production** (0.811→0.882 / 0.865→0.914, **0 TP perdu**).
2. **A5 (tags de confiance UI)** — bonne idée à **greffer** sur A3/A4 (CHANTIER 2). Jeter le packaging standalone.
3. **A2 (arbitrage hit↔bounce + proximité)** — spéculatif : physiquement fondé mais **NON validé** (aucun GT avec chevauchement) et bidirectionnel (peut baisser la recall). **À différer** jusqu'à un GT riche.
4. **A1** — dominé par A3/A4 (même cœur, mais bidirectionnel/dur et lit le spline). À ne pas implémenter tel quel.

### Découpage en CHANTIERS (pour de FUTURES sessions de code)

> **Invariant transverse (le plus important) :** le gate doit lire la direction sur les **centres tracker BRUTS pré-spline AVEC un masque d'interpolation**, threadés en entrées explicites — **jamais** sur `re_smoothed`/`chosen` retournés par le détecteur. Tant que ce masque n'existe pas, **rien ne doit être mergé**.

- **CHANTIER 0 — Exposer la trajectoire pré-spline + masque (PRÉ-REQUIS bloquant).**
  Conserver, à côté de `all_ball_centers`, le **chemin tracker-sélectionné avant `smooth_ball_trajectory`** + un **masque booléen `is_real[i]`** (détection réelle vs comblée). Le threader jusqu'au point d'appel rebond (`run_pipeline_8s.py:1100-1105`). **Sans ce chantier, A3/A4 sont impossibles.** Le chemin **WASB** expose des centres denses ; le chemin **Kalman** (`detect_bounces_from_trajectory`, l.112) ne retourne rien → CHANTIER 0 doit aussi décider du **SKIP explicite du chemin Kalman** (voir CHANTIER 3).

- **CHANTIER 1 — Helper vitesse signée + gate veto (le cœur A3/A4).**
  Helper `numpy.polyfit` (pente moindres carrés sur points réels) + veto one-directional : `half=round(fps*0.10…0.12)`, **deadband RELATIF** `eps=max(0.15*speed_scale, W*1e-3*(60/fps))` **(0.15 et 1e-3 = felix-provisionnels, à re-dériver cross-clip — CHANTIER 5)**, `REJECT` sur vx-flip confiant mesurable, **ABSTAIN** sinon. Slot : post-filtre après le retour de `detect_bounces_robust`. **CONFIRM exige vy-flip** (greffe b) ; sinon le candidat **reste** mais tagué « unconfirmed » (ne le retire pas → ne touche pas la recall).

- **CHANTIER 2 — Tags de confiance UI (greffe A5 sur A3/A4).**
  Propager `direction_state ∈ {CONFIRMED, unconfirmed/abstain, weak}` vers le badge de confiance existant. Aucun risque détection (présentation seulement).

- **CHANTIER 3 — Décision chemin Kalman (curve-fit).**
  `detect_bounces_from_trajectory` (l.112) n'expose aucun array → **SKIP explicite du veto sur ce chemin** jusqu'à ce qu'il expose trajectoire+masque (sinon, gater sur un spline re-dérivé où vy-flip recall = 62-73 % risque de descendre **sous 0.72**). Documenter ce skip comme dette.

- **CHANTIER 4 (différé, spéculatif) — Arbitre hit↔bounce (A2).**
  Remplacer le `bounce_guard` aveugle (`shots.py:214/290`) par l'arbitrage physique `vx_flip → HIT, vx-pres+vy-flip → BOUNCE`. **Bloqué sur** un GT avec chevauchements hit/bounce réels (cf. §5). Ne pas livrer sans ce GT.

- **CHANTIER 5 (validation, transverse) — Re-score natif + re-dérivation des constantes.**
  Re-valider EPS / `speed_scale` / liste de candidats en **1080p natif** (les probes tournaient en 1280×720 ; la **liste de candidats change avec la résolution** — `f105`/`f1859` vs `f1254`/`f1826`). Re-dériver les facteurs `0.15` et `1e-3` sur ≥2 clips. Re-run `tools/bounce_eval` + `tests/test_bounce_regression.py` + `tests/test_shots_regression.py` avant tout merge.

---

## 5. Plan de validation — métrique de confusion REBOND↔FRAPPE

Le harnais bounce existe déjà (`tools/bounce_eval/eval_bounces.py` + `bounce_io.py`, `tests/test_bounce_regression.py` floor 0.72). **Il manque la moitié frappe et la métrique de confusion.** À spécifier (pas à coder) :

### 5.1 GT de frappes — format

Créer `tests/fixtures/shots/<clip>.shots.json` (miroir de `tests/fixtures/bounces/felix.bounces.json`) :
```json
{ "fps": 60, "width": 1920, "height": 1080,
  "shots": [ { "frame": 1609, "x": 847, "y": 641, "player": "near", "kind": "groundstroke" } ] }
```
**Clip prioritaire = `felix`.** Les FP confirmés `f105`/`f820`/`f1609` (réversions horizontales 153–180°, §3) sont déjà identifiés comme frappes → ils deviennent le **bootstrap minimal de GT-frappes**, ce qui ferme la boucle « vx-signe prouvé seulement indirectement » (§2.3). **Exigence dure :** ce GT doit inclure **au moins une frappe down-the-line** (vx préservé) ; sinon le claim « le veto tue les vraies frappes » reste **NON testé pour le cas tueur** (le veto est aveugle au down-the-line par construction).

### 5.2 Métrique de CONFUSION bounce↔shot (le livrable manquant)

Au-delà de la F1 par classe, une **matrice de confusion croisée** au matcher greedy 1-1 (tolérance `round(0.15*fps)=9` frames, identique à `bounce_io`) :

|  | Prédit REBOND | Prédit FRAPPE | Non détecté |
|---|---|---|---|
| **GT REBOND** | TP_b | **confusion B→H** | FN_b |
| **GT FRAPPE** | **confusion H→B** ★ | TP_h | FN_h |

★ **`confusion_H→B`** = nombre de frappes GT émises comme rebonds = **exactement le bug utilisateur**. C'est la métrique cible.

> **Règle de désambiguïsation (à spécifier explicitement, sinon `confusion_H→B` n'est PAS déterministe) :** quand un **même prédit** tombe à ≤ tolérance d'une GT-rebond **ET** d'une GT-frappe (l'ambiguïté bounce-vs-hit elle-même), trancher dans cet ordre : **(1) frame la plus proche gagne** ; **(2) en cas d'égalité de distance, la classe est celle du label PRÉDIT** (un prédit étiqueté « rebond » s'apparie d'abord à la GT-rebond) ; **(3)** un GT ne peut être apparié qu'une fois (greedy 1-1, le plus proche d'abord, comme `bounce_io`). Sans cette règle, l'assertion 5.3-Nouvelle-1 (`==0`) serait instable.

### 5.3 Assertions de régression (sans coder)

- **Existante (à préserver) :** `bounce_F1 ≥ 0.72` (`tests/test_bounce_regression.py`, chemin **Kalman/curve-fit**, cache replay ~0.80).
- **Nouvelle 1 :** `confusion_H→B == 0` sur le cache felix WASB (les frappes `f820`/`f1609`(+`f105`) ne doivent plus apparaître comme rebonds).
- **Nouvelle 2 :** `bounce_F1 ≥ 0.90` sur felix WASB **après gate**. ⚠️ **Le plancher est fixé À LA VALEUR POST-GATE (≥0.90)**, pas en-dessous : un seuil ≤0.865 ne verrouillerait **rien** (il est sous la baseline pré-gate 0.865 natif). Choisir 0.90 (< le 0.914 mesuré natif, marge de bruit) verrouille effectivement le gain. **Re-confirmer la valeur exacte en 1080p natif** (CHANTIER 5) avant de figer le seuil.
- **Existante (à préserver) :** `shot_recall ≥ 6/8` sur demo3 (`tests/test_shots_regression.py`, `RECALL_FLOOR=6`, `MAX_HITS=16`) — A3/A4 ne touchent pas `detect_hits`, donc inchangé **par construction**.
- **Différée (CHANTIER 4) :** une assertion d'arbitrage nécessitera un GT avec chevauchements (inexistant aujourd'hui).

### 5.4 Cache de détection

Comme pour les bounces, committer un cache de centres **pré-spline + masque** felix (pas de GPU/vidéo au test) pour rendre la régression de confusion rapide et déterministe. **Fixer aussi la résolution du cache à 1920×1080 natif** (la liste de candidats en dépend, cf. CHANTIER 5).

---

## 6. Risques / ce qui pourrait ne pas marcher

1. **Validation mono-clip (risque #1).** Tout repose sur **felix, 1 surface, 60 fps**, et **les FP capturés sont tous des réversions cross-court 153–180°** — biais de sélection qui flatte le gate. **Un clip avec des échanges down-the-line montrerait le gate capturant bien moins de FP frappe** (vx préservé → gate aveugle, *cas tueur*). Mitigation : valider sur ≥2 clips additionnels + labelliser ≥1 frappe down-the-line (§5.1) avant tout claim « résout bounce-vs-hit ».

2. **Contrat de sécurité inimplémentable si mal câblé (CHANTIER 0).** Lire `re_smoothed` (post-spline, l.613) au lieu des centres bruts+masque = mesurer vx sur le substrat qui **dégrade le signal aux GT (15/15→14/15)** et lit à travers les trous comblés → le gate devient bruité et **pourrait DROP un vrai rebond** sur chemin épars (vy-flip recall 62-73 %, vx-préservé ~80 % aussi en mi-vol) → **descente sous le plancher 0.72**. CHANTIER 0 strictement bloquant.

3. **Hardcoding du deadband (grief réel).** `EPS=frame_width*0.0004` (A3/A5) est du felix-fit : la vitesse-par-frame dépend de fps **ET** des mètres-par-pixel (zoom/homographie), **pas** de la largeur de frame seule (le facteur 0.0004 → 0.512 px/f en 4K/30fps ferait dériver des rebonds lointains lents réels en zone d'abstention). **A4 atténue** via un deadband **relatif à la vitesse propre** (`0.15*speed_scale`, ≈1.43 px/f sur felix), bien plus défendable — **mais les facteurs `0.15` et `1e-3` restent non validés cross-clip** (admis aussi au site d'implémentation, CHANTIER 1). **Fix recommandé :** exprimer EPS via fps **et** un proxy d'échelle (médiane du déplacement-par-frame du clip, ou mètres/pixel via homographie quand présente), avec re-dérivation cross-clip des facteurs.

4. **Cas durs structurellement non couverts.** **Down-the-line hit** (vx préservé) et **net-cord vers l'avant** (vx inchangé) sont des **angles morts** : vx-flip est *nécessaire-non-suffisant* pour une frappe. ⚠️ **La greffe (b) « vy-flip requis au CONFIRM » N'ADRESSE PAS down-the-line** (qui possède un vrai vy-flip de rebond) — elle ne capture que les **FP vx-préservés sans réflexion verticale** (ex `f602` glide). Ne pas conflater. **Drop-shot / lointain lent** : vx<EPS → abstention (ni gain ni dégât). **Volée** : hors périmètre (aucun candidat rebond n'existe).

5. **Gain dense seulement.** Sur un clip où Kalman s'effondre (broadcast épars), le gate **abstient massivement** : la confusion utilisateur n'est que **partiellement** corrigée. Le fix complet exige la trajectoire dense **WASB** (CLAUDE.md recommande déjà `--ball-tracker wasb`). **Ne pas sur-promettre sur les clips amateurs épars.**

6. **Arbitrage hit↔bounce non validé (CHANTIER 4).** demo3 n'a **aucun** chevauchement hit/bounce dans la garde (frappe la plus proche à ~0.40 s de l'unique rebond f457). La règle est physiquement fondée mais **non mesurée end-to-end**, et **bidirectionnelle** (peut désormais garder les deux, plus correct mais à régression-gater). **À différer.**

7. **Résolution de validation (tension à lever honnêtement).** Les probes ont tourné en **1280×720**, alors que felix est nativement **1920×1080**. Conséquence **mesurée** : la **liste de candidats du détecteur change avec la résolution** (1280×720 produit `f105`/`f1859` ; natif produit `f1254`/`f1826`) car les seuils fractionnaires internes (`min_drop_frac`, `min_v_excursion_frac=0.055`, …) tirent à cette échelle. Le gain est **reproduit aux DEUX résolutions** (0.811→0.882 et 0.865→0.914, 0 TP perdu), mais le **chiffre exact à figer dans l'assertion 5.3-Nouvelle-2 doit être pris du run 1080p natif** (CHANTIER 5). C'est pourquoi le doc dit « reproduit sur la production » **et** « à re-confirmer en natif » : ce n'est pas une contradiction, c'est la même mesure aux deux échelles avec figeage du seuil en natif.

8. **Occlusion irréductible.** **1/16 rebond GT (`f424`)** est non mesurable dans **tous** les substrats (trous WASB des deux côtés). Toute logique de direction doit **dégrader gracieusement** : pas de détection → pas de direction → repli sur y-max + énergie ; **ne jamais affirmer une frappe sur données manquantes**.

---

### Conclusion pour le PM

Le bug est réel et **upstream** : la production calcule `(vx,vy)` puis les jette (`run_pipeline_8s.py:1015-1017`). Le signal qui le corrige existe et est **reproduit sur la production** : un **veto vx-flip one-directional** (A3 ⊕ A4) fait passer la F1 rebond felix **0.811→0.882** (run 1280×720) / **0.865→0.914** (run natif) en tuant les FP de frappe `f820`/`f1609` (réversions ~180°) **à coût TP nul**. C'est **pm-ready et scoped** — à trois conditions : (1) respecter l'invariant bloquant (**CHANTIER 0** : centres pré-spline + masque, jamais l'array splinné), (2) exprimer le deadband en **relatif** (A4) en re-dérivant `0.15`/`1e-3` cross-clip, (3) **ne pas sur-promettre** : c'est un **filtre de précision sur le chemin dense**, pas un détecteur universel. Les angles morts (down-the-line, net-cord-avant, épars) et la validation **mono-clip** imposent une **labellisation de frappes felix incluant ≥1 down-the-line** (§5) avant tout claim « résout bounce-vs-hit ».
