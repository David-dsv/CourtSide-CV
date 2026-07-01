# CR — Accuracy overhaul 2026-07-02 : événements quasi parfaits + hero métrique + refonte web

**Branche :** `feat/accuracy-overhaul`. **Méthode :** mesure LIVE d'abord (≥2 runs frais par étape,
jamais un cache seul), replays contrôlés sur les dumps `COURTSIDE_DUMP_METHODO_INPUTS`,
zéro constante nouvelle non ancrée. GT = les fixtures humaines demo3 (9 bounces + 8 hits + FH/BH)
et felix (16 bounces).

## TL;DR

| métrique (demo3 live, `tennis.mp4 -s73 -d13` mps) | avant | après |
|---|---|---|
| bounce F1 | 0.941 (R 0.889) | **0.947 (R 1.000 — 9/9)** |
| hit F1 | 0.625 (R 0.625, 3 fantômes) | **0.941 (R 1.000 — 8/8)** |
| confusion H→B / B→H | 0 / 1 | **0 / 0** |
| marqueurs fantômes | 2-3 | **0** (les 2 "extra" = le rituel pré-service + LE SERVICE, réels, vérifiés à l'image) |
| vitesse | scalar (aucune homographie) | **homography(high)** — 14/14 keypoints, rms 3.5 px, km/h perspective-corrigés |

Le hero (`data/output/tennis_hero_FINAL_annotated.mp4`) porte maintenant : rappel événements
**17/17**, marqueur de SERVICE détecté (non annoté dans la GT), zones court perspective-exactes,
vitesses métriques, `homography_H` émis dans le `_stats.json`.

## Les 6 causes racines corrigées (dans l'ordre de découverte)

1. **`vision/events.py` — évidence de frappe non corroborée.** Un « swing » de `detect_hits`
   comptait comme évidence HIT même quand le ballon tracké était à 1.25–2.4 box-heights de tout
   poignet. Les clusters fantômes absorbaient les slots du DP d'alternance et déplaçaient les
   vrais contacts (pred@65 vs GT@81 ; pred@487 vs GT@468). **Fix :** un swing n'est une évidence
   que si le ballon est à portée de CONTACT (`dmin < near_wrist`) ou non mesurable ; la mi-bande
   (0.5–1.2) ne fait que renforcer. Mesuré : toutes les vraies frappes GT ≤ 0.49.
2. **`vision/events.py` — cluster réel non-étiquetable (guard B→H trop strict).** Au vrai contact
   @81, le détecteur de rebond tire (le redirect raquette ressemble à un rebond) → `bnc sans
   swing` interdisait HIT pendant que `hit_like` interdisait BOUNCE → cluster mort. **Fix :**
   exception *redirect* — vx inversé AU poignet = physique de frappe (le vx_flip_veto felix), HIT
   redevient légal.
3. **`vision/events.py` — parité knife-edge.** `k = round(gap/step)` : un vrai couple consécutif
   à ~1.56 beats devenait k=2 (parité paire) → label interdit. **Fix :** k ∈ {floor, ceil}, le
   meilleur k parité-légal gagne, la pénalité temporelle continue de payer l'écart.
4. **`vision/shots.py` — `contact_frame = j` (bug de variable de boucle).** La frame émise était
   la DERNIÈRE du window de recherche (+0.2 s systématique), pas l'argmin ballon↔poignet. Tous
   les candidats hit étaient en retard ; l'aval s'était calibré autour. **Fix :** argmin poignet
   (fallback torse), position ballon à cette frame.
5. **`vision/shots.py` + `run_pipeline_8s.py` — les frappes far étaient structurellement
   indétectables.** (a) le serve-gate frame-fraction (`wy < 0.30·fh`) est vrai pour TOUT contact
   far → `far_aware=True` (gate player-relative + NMS par côté) pour le chemin methodo ;
   (b) le plancher d'amplitude frame-scaled (8.64 px/f) est aveugle au joueur far minuscule
   (swing réel @141 : pic 7.57, FWHM 19) → plancher proportionnel à la taille de boîte médiane
   du côté (le plus grand joueur garde exactement l'ancien plancher — zéro constante) ;
   (c) `detect_hits` recevait les poses BRUTES (far éparse/bruitée) → il consomme maintenant les
   pistes LOCKED gap-filled (les mêmes que la proximité du classifieur).
6. **`vision/pose.py` — le rectangle ITF rejetait le joueur near (LE bug --homography).** Avec
   une homographie précise, la bande « pieds dans court+4% » rejetait le vrai joueur near debout
   1-2 m DERRIÈRE la ligne de fond (997 px vs 810+43) → piste near vide → événements affamés
   (bounce 0.706 / hit 0.333 au 1er run --homography !). **Fix :** la bande adaptative (dérivée
   des pieds des candidats — la vérité de la scène) est TOUJOURS utilisée ; les zones ne servent
   plus qu'au net_y. ⚠️ felix (sidecar calibration → zones) était potentiellement affecté pareil.
7. *(présentation)* **HUD « Balle max » spiké** (224 km/h sur un rallye à ~100) : la série km/h
   par-frame porte des spikes 1-2 frames aux jonctions de spline. **Fix :** médiane glissante
   0.10 s pour l'AFFICHAGE seul (gauge + pic d'échange) ; les stats gardent la série brute.

## Le piège cache, encore (2 caches reconstruits)

`demo3_event.json` / `demo3_pose.json` dataient d'avant farselect/ball-density : leurs poses
plaçaient de vrais contacts à 2+ box-heights du ballon (physiquement impossible). Tout
changement qui FAIT CONFIANCE à la mesure était puni par le fixture et récompensé en live.
Reconstruits depuis le dump d'un run canonique live (`scripts/rebuild_demo3_event_cache_from_dump.py`) ;
le replay du cache égale maintenant le run live (WASB consistency 100 %). Planchers re-verrouillés
À LA HAUSSE : confusion 0/0 (dur), bounce ≥ 0.90, hit ≥ 0.80 ; shots-regression : 8/8 strikes
détecteur-seul, FH/BH ≥ 6 corrects / 0 faux.

## Ce que la GT ne compte pas (décision assumée)

Les 2 « spurious » restants sur le hero : **f11 = rebond du dribble pré-service**, **f44 = LE
SERVICE de Moutet** — physiquement réels, vérifiés frame par frame. La GT humaine n'annote ni le
rituel pré-service ni le service ; la GT n'est PAS modifiée (c'est le benchmark de l'humain),
c'est documenté ici. Un vrai marqueur de service = un plus produit.

## Web (`web/`) — l'offre post-extraction

Nouveaux blocs branchés sur des champs pipeline jusque-là JAMAIS affichés :
- **`KeyInsights`** (« Ce que la vidéo dit de ton jeu ») : phrases coach classées, ancrées frame
  (revers X km/h plus lent, fatigue −Y %, ratio fautes directes, biais de profondeur, cadence).
  Une règle ne tire que sur un signal fort — pas de remplissage.
- **`OutcomeBar`** : barre 100 % empilée des issues de rallyes + ratio « fautes directes /
  points décisifs ». Sémantique statut, icône+label+compte sur chaque segment.
- **`FatigueCard`** : le bloc `fatigue` réel (1er vs dernier tiers, drop %, pente/min,
  confiance d'échantillon honnête) — remplace le stub proxy.
- **`StrokeCompare`** : CD vs RV sur UN axe km/h (+ pointe, qualité en texte secondaire) avec
  le delta actionnable.
- **`SwapTimeline`** : les 19 side-swaps match-mode sur un rail cliquable, flag maillots
  ambigus.
- Couleurs dataviz par thème **validées** (bande de luminance, chroma, ΔE CVD, contraste 3:1)
  → tokens `--viz-*` dans `globals.css` ; timeline rebonds passée sur `depthHex` ;
  player-compare : attribution réelle frappe→rebond-suivant (fini le `i % 2`).
- **Fixture vitrine `acapulco-hero`** : stats réelles du hero + `homography_H` RÉEL (le tier
  `homography(high)` enfin démontré avec de vraies données) ; le parcours upload y atterrit.
- Pipeline : `_stats.json` émet désormais `homography_H` (3×3 image→mètres, convention
  minimap/projection.ts) — le front n'a plus à synthétiser de H.

## Vérifications

- 23/23 tests verts (planchers re-verrouillés inclus) ; felix bounce (kalman 0.800 / wasb 0.865)
  intact sur les tests ; runs felix end-to-end re-mesurés après le fix de bande (voir suite du CR).
- demo3 live ×3 (E ≡ F byte-identiques ; HERO2 = 17/17) + QA visuelle frame-par-frame du rendu.
- Web : `pnpm build` + `lint` + TypeScript verts ; rendu serveur des 5 fixtures vérifié par curl.
