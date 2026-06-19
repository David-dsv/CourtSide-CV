# CourtSide-CV — État de l'art (SOTA) & feuille de route techno — 2026-06-19

> **Objet.** Recherche internet exhaustive (103 agents, fan-out web multi-angles → fetch de 21 sources → vérification adverse 3-votes par affirmation → synthèse citée ; 24 claims confirmés / 1 réfuté) pour identifier les meilleures technos, modèles et méthodes permettant de faire passer CourtSide-CV à un niveau **state-of-the-art** sur le cas d'usage **tennis amateur filmé au téléphone → data exploitable**.
>
> **À lire avec** `AUDIT_2026-06-19.md` : ce document répond *« avec quoi construire »*, l'audit répond *« qu'est-ce qui est cassé »*. Les deux verrous mesurés de l'audit (rebonds F1≈0.29, échelle px→mètre fausse) trouvent ici leur réponse SOTA.

---

## 0. TL;DR — les 2 mouvements qui changent tout

1. **Remplacer la balle YOLO26+Kalman par un modèle heatmap-temporel de la famille TrackNet — `WASB`.** C'est la cause racine du verrou n°1 de l'audit : le tracking décroche → les rebonds s'effondrent. Un modèle heatmap est conçu *exactement* pour la balle petite/rapide/floue et ne « décroche » pas comme un détecteur bbox + Kalman.
2. **Construire une vraie homographie via keypoints de court (`TennisCourtDetector`, 14 points).** C'est le verrou n°2 : remplace l'échelle scalaire « 58 % de la hauteur » par une transformation projective px→mètre correcte → vitesse et profondeur enfin fiables.

Ces deux briques sont les **bottlenecks mesurés** du pipeline. Tout le reste (pose, stats, classification de coups) vient après et en dépend.

---

## 1. Détection & tracking de la balle — **passer en heatmap (TrackNet family)**

> **Verrou d'audit visé : n°1 (rebond cassé, cause = tracking qui décroche).**

L'approche actuelle (détecteur bbox YOLO + Kalman + interpolation) est structurellement fragile : dès que la détection manque quelques frames, le Kalman coast puis se perd sans récupérer. Le SOTA a changé de paradigme : **régression de heatmap sur empilement temporel de frames**, ce qui exploite le mouvement et ne dépend pas d'une bbox nette.

| Modèle | Quoi | Perf publiée | Maturité / licence | Adéquation amateur |
|---|---|---|---|---|
| **WASB** (BMVC 2023, NTT) | Heatmap-temporel léger, 3 composants (features HR + entraînement position-aware + inférence à cohérence temporelle), type HRNet | **Tennis F1=95.6 / Acc=91.8 / AP=94.2** (seuil 4 px), **1.5M params, 35 FPS**. Bat MonoTrack (F1=92.1) et TrackNetV2 (89.4). +7.8 à +16.8 % AP sur 6 méthodes / 5 sports | Code open-source [`nttcom/WASB-SBDT`](https://github.com/nttcom/WASB-SBDT). **Recommandé n°1** | ✅ Léger, rapide, multi-sport, plug-in |
| **TrackNetV3** | Heatmap 2 modules (prédiction avec fond estimé + rectification par inpainting des gaps) | Badminton **F1=98.56** (vs V2 97.03, YOLOv7 68 %) | [`qaz812345/TrackNetV3`](https://github.com/qaz812345/TrackNetV3). ⚠️ **Impl. badminton-only, sans tennis ni rebonds natifs → fine-tuning requis** | ⚠️ Demande adaptation tennis |
| **TrackNetV4** (ICASSP 2025) | Module **plug-and-play** : attention de mouvement par frame-differencing au-dessus d'un TrackNet existant | Améliore empiriquement V2 **et** V3 sur tennis + badminton (additif) | [arXiv 2409.14543](https://arxiv.org/pdf/2409.14543) | ✅ Renfort, pas un remplacement |
| **TOTNet** (2025) | Heatmap-temporel **3D conv, occlusion-aware** (loss pondérée visibilité + augmentation d'occlusion), **conçu pour l'analytics OFFLINE** | Tennis partiellement occulté **RMSE 63.41 vs WASB 105.73** ; bat TrackNetV2/WASB/monoTrack/TTNet sur 4 datasets | [arXiv 2508.09650](https://arxiv.org/pdf/2508.09650) | ✅✅ **Pile notre cas** (traitement différé, précision > vitesse) |

**Reco balle :**
- **Adopter `WASB`** comme nouveau tracker de balle (fine-tune sur tes données + dataset tennis ex-TrackNet). C'est le meilleur compromis perf/poids/maturité.
- **Garder le dataset/labels balle existants** : ils servent à fine-tuner WASB. Le YOLO26 fine-tuné n'est pas perdu (fallback / pré-filtre possible).
- **Évaluer TOTNet** si la précision sous occlusion (joueur devant la balle) reste un problème en traitement différé — c'est explicitement son terrain.
- ⚠️ *Veille :* TrackNetV5 (arXiv 2512.02789, déc. 2025) critique le frame-differencing de V4 — la famille bouge vite, refaire un point dans ~6 mois.

---

## 2. Détection de rebonds — **fit de 2 courbes + intersection**

> **Verrou d'audit visé : n°1 (F1≈0.29).** Préalable : une trajectoire balle dense (→ §1 d'abord).

| Méthode | Principe | Perf | Source |
|---|---|---|---|
| **Two-curve fit + intersection** (ICIG 2021) | Ajuster par moindres carrés le segment **descendant** et le segment **montant** de la trajectoire ; le rebond = leur **intersection** ; split choisi à **MSE minimale**. Monoculaire, **sans reconstruction 3D ni calibration** | **99.4 %** (338 cas normaux), 81.8 % près des lignes, **98.9 % global** | [arXiv 2107.09255](https://arxiv.org/pdf/2107.09255), impl. `joeyc408/TennisPoint` |

**Reco rebonds :**
- **Remplacer** ta détection « velocity-sign-change » par le **fit de 2 courbes + intersection**. C'est géométriquement plus robuste (un fit global encaisse le bruit là où un changement de signe local échoue) et ça se branche directement sur la trajectoire dense produite par WASB.
- ⚠️ **Nuance importante (vérifiée) :** le « 99 % » est de la précision **line-calling in/out** (caméra fixe 240 fps, un seul côté, dataset retiré) — **pas** un F1 de détection de *frame* de rebond directement comparable à ton 0.29. À traiter comme une **méthode prometteuse à valider**, pas comme un score garanti.
- **Continue d'utiliser ta vérité-terrain** (`felix.bounces.json` + `eval_bounces.py`) : c'est elle qui dira le vrai F1 de cette méthode chez toi. C'est ton thermomètre, garde-le.
- ⚠️ **Manque identifié :** il n'existe pas de dataset public de tennis avec **frames de rebond annotées** prêt à l'emploi → tu devras **étendre ta ground truth** (plusieurs clips amateurs annotés). C'est confirmé comme un prérequis non couvert par l'existant.

---

## 3. Court & homographie px→mètre — **keypoints, pas échelle scalaire**

> **Verrou d'audit visé : n°2 (échelle scalaire fausse sous perspective).**

| Brique | Quoi | Perf | Source / adéquation |
|---|---|---|---|
| **TennisCourtDetector** | **14 keypoints** de court via TrackNet modifié (1 image in, 15 canaux out = 14 pts + 1 centre) | Accuracy **0.961 / médiane 1.83 px** (avec refining ; base 0.933 / 2.83 px) | [`yastrebksv/TennisCourtDetector`](https://github.com/yastrebksv/TennisCourtDetector). **Brique à adopter** |
| **Court line detection amateur** | Pipeline robuste angles bas / court partiel : **filtrage couleur** du court + **détection filet YOLO-v5** pour cropper le côté proche + **suppression d'ombres** | **81–100 %** vs baseline Hough+homo **0–92.7 %** sur courts non-pros | [arXiv 2404.06977](https://arxiv.org/pdf/2404.06977). **Pile le cas amateur** |

**Reco court/homographie :**
- **Adopter `TennisCourtDetector`** pour obtenir 14 keypoints, puis calculer une **vraie homographie** `cv2.findHomography(image_pts, real_world_pts)` vers un plan métrique du court (dimensions ITF connues). Tu mesures alors vitesse et profondeur **en mètres réels**, pas via un scalaire faux.
- ⚠️ **Réfutation notable (vote 1-2) :** l'idée que TennisCourtDetector fournit l'homographie **clé-en-main** a été **rejetée** par la vérification. Il donne les **keypoints** ; **l'homographie reste à construire et valider toi-même**. Ça confirme que c'est bien LE chantier (pas un simple `pip install`).
- **Pour l'angle amateur / court partiel** (le fond du court hors champ → tous les keypoints ne sont pas visibles) : intégrer les techniques d'arXiv 2404.06977, et gérer l'**homographie avec keypoints manquants** (≥4 points visibles suffisent mathématiquement, mais la précision résiduelle est à mesurer — c'est une question ouverte).
- ⚠️ Métriques court self-reported sans benchmark tiers → **valider sur tes propres clips téléphone**.

---

## 4. Pose & tracking joueurs — **garder l'off-the-shelf, option RTMPose**

> **Pas un verrou.** Décision produit déjà actée : pose jamais réentraînée. La recherche le confirme.

- Ton **YOLOv8-pose off-the-shelf reste valable** — aucun besoin de réentraîner, cohérent avec `PROJET.md §2`.
- Si tu veux monter en qualité/vitesse sans réentraîner : **RTMPose** (MMPose) est la référence temps-réel ([`mmpose/projects/rtmpose`](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)), swap-in possible. Bénéfice marginal pour l'instant — **à différer**.

---

## 5. Pipelines tennis open-source — **squelette à réutiliser, pas SOTA**

Plusieurs systèmes complets vidéo→stats existent. À **piller pour la structure**, mais aucun n'est SOTA ni adapté tel quel à l'amateur :

| Repo / papier | Contenu | Verdict |
|---|---|---|
| **Hawk-Eye System** (arXiv 2511.04126, nov. 2025) | YOLOv8x joueurs + YOLOv5s balle + ResNet50 14 keypoints → homographie mini-court top-down | ⚠️ **Broadcast-tuned**, auto-déclaré : *« degrades with unusual camera positions or extreme angles »* = exactement le cas téléphone. Balle YOLO = viable mais **pas SOTA heatmap** |
| [`abdullahtarek/tennis_analysis`](https://github.com/abdullahtarek/tennis_analysis) | Pipeline éducatif YOLO + court + mini-map | Bon **squelette d'architecture** (vidéo→stats), à moderniser brique par brique |
| [`yastrebksv/TennisProject`](https://github.com/yastrebksv/TennisProject) | Projet tennis intégré du même auteur que TennisCourtDetector | Cohérent avec les briques §1/§3, à étudier en priorité |
| [`ArtLabss/tennis-tracking`](https://github.com/ArtLabss/tennis-tracking) | Tracking balle + court + heatmap rebonds | Référence d'intégration |

**Reco pipelines :** réutiliser l'**architecture** (vidéo → détections → homographie → mini-court → stats) de ces repos comme plan, mais brancher **TES** briques SOTA (WASB, TennisCourtDetector, fit-2-courbes). Ne pas adopter un pipeline broadcast en bloc.

---

## 6. Classification de coups & segmentation de rallyes — **zone à défricher**

> **Chaînon manquant data→insight de l'audit (§4).** Honnêteté : la recherche **n'a pas confirmé de SOTA** clair ici.

- Aucun claim vérifié n'a établi une méthode SOTA robuste pour coup droit/revers/service ni pour la segmentation automatique de points sur **angle amateur**. C'est une **lacune réelle**, pas un oubli.
- Pistes repérées (à creuser, non validées) : [`antoinekeller/tennis_shot_recognition`](https://github.com/antoinekeller/tennis_shot_recognition) (reconnaissance de coups via pose), dataset **TenniSet** (événements denses annotés), [arXiv 2104.09907] (action tennis). Datasets type **THETIS** (gestes tennis) à évaluer.
- **Reco :** commencer **géométrique** (position joueur + direction balle au contact, via pose + trajectoire) pour FH/BH, et segmenter les rallyes par **gaps temporels de la trajectoire balle** (silence = point fini). C'est cohérent avec le Chantier 5 de l'audit, et ça ne nécessite pas un modèle SOTA dédié pour un premier insight.

---

## 7. Datasets — ce qui existe, ce qui manque

| Besoin | Existe ? | Source |
|---|---|---|
| Trajectoire balle tennis | ✅ Dataset tennis ex-TrackNet (sert à WASB) | via WASB / TrackNet |
| Keypoints de court | ✅ Dataset TennisCourtDetector | `yastrebksv/TennisCourtDetector` |
| **Frames de rebond annotées (tennis)** | ❌ **N'existe pas prêt à l'emploi** | **À construire** (étendre ta GT `felix.bounces.json`) |
| Événements / coups annotés | ⚠️ TenniSet, THETIS (à évaluer) | semanticscholar / arXiv 2104.09907 |

**Le dataset le plus stratégique à constituer = ta vérité-terrain rebonds élargie** (plusieurs clips amateurs). C'est le prérequis n°1 déjà identifié dans l'audit, et la recherche confirme qu'aucun jeu public ne te l'évite.

---

## 8. Le chemin le plus court vers le SOTA — plan priorisé

Ordre de dépendance, aligné avec les chantiers de l'audit :

| Priorité | Action | Effort | Débloque | Verrou audit |
|---|---|---|---|---|
| **1** | **Migrer la balle vers WASB** (fine-tune sur tes labels + dataset tennis). Mesurer la densité de trajectoire vs l'actuel | **L** | Trajectoire dense, fiable → tout l'aval | n°1 |
| **2** | **TennisCourtDetector + vraie homographie** `cv2.findHomography` vers plan métrique ITF | **L** | Vitesse/profondeur **en mètres réels**, comparables | n°2 |
| **3** | **Rebonds = fit 2 courbes + intersection** sur la trajectoire WASB, mesuré contre ta GT (`eval_bounces.py`) | **M** | Rebonds fiables → profondeur/qualité/stats | n°1 |
| **4** | **Élargir la vérité-terrain rebonds** (5-10 clips amateurs annotés) | **M** | Mesure objective des priorités 1 & 3 | (prérequis) |
| **5** | **Robustesse amateur** : filtrage couleur + crop filet YOLO + homographie à keypoints partiels (arXiv 2404.06977) | **M→L** | Le système tient sur vraie vidéo téléphone | n°2 |
| **6** | **Segmentation rallyes (gaps balle) + attribution coup→joueur** (géométrique) | **M** | Stats **par joueur**, premières vraies insights | §4 audit |
| **7** | *(plus tard)* RTMPose, classification FH/BH ML, action recognition | **S→L** | Raffinement | — |

**Ce qu'on GARDE :** l'architecture deux-passes, la pose off-the-shelf, le harnais d'éval rebonds (`eval_bounces.py` — précieux), la couche affichage, le dataset balle.
**Ce qu'on REMPLACE :** tracker balle YOLO+Kalman → WASB ; échelle scalaire 58 % → homographie keypoints ; détection rebonds maison → fit 2 courbes.
**Ce qu'on AJOUTE :** keypoint court detector, robustesse amateur (color/net-crop/shadow), segmentation rallye + attribution joueur.

---

## 9. Limites & questions ouvertes (honnêteté méthodo)

- **Transfert de domaine :** beaucoup des meilleurs scores (TrackNetV3 98.6 %, une partie de TOTNet) viennent du **badminton/ping-pong**, pas du tennis. Les chiffres WASB tennis (95.6) sont sur le dataset ex-TrackNet, mais les comparaisons reposent en partie sur des ré-implémentations par les auteurs WASB.
- **Métrique rebonds :** le 99 % d'arXiv 2107.09255 = line-calling in/out, **pas** un F1 de frame de rebond — comparaison indicative, à re-mesurer chez toi.
- **Sources court/homographie :** TennisCourtDetector, 2404.06977, 2511.04126 = métriques self-reported, preprints à rigueur variable → **valider sur tes clips**.
- **Réfuté :** homographie clé-en-main de TennisCourtDetector → **rejeté (1-2)**, l'homographie est à construire.
- **Non couvert :** classification de coups & segmentation de points sur angle amateur — pas de SOTA confirmé, zone à défricher (§6).
- **Sensibilité temporelle :** recherche arrêtée mi-2026 ; la famille TrackNet évolue vite (TrackNetV5 déc. 2025).
- **Coût GPU offline :** le budget réel (temps/GPU) pour WASB + court + pose sur une vidéo complète en différé n'a pas été chiffré — à benchmarker avant industrialisation (Palier 2).

---

## Sources clés

**Balle :** [WASB repo](https://github.com/nttcom/WASB-SBDT) · [WASB BMVC paper](https://papers.bmvc2023.org/0310.pdf) · [TrackNetV3](https://github.com/qaz812345/TrackNetV3) · [TrackNetV4 arXiv](https://arxiv.org/pdf/2409.14543) · [TOTNet arXiv](https://arxiv.org/pdf/2508.09650)
**Rebonds :** [Two-curve fit arXiv 2107.09255](https://arxiv.org/pdf/2107.09255)
**Court/homographie :** [TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector) · [Amateur court lines arXiv 2404.06977](https://arxiv.org/pdf/2404.06977) · [Hawk-Eye arXiv 2511.04126](https://arxiv.org/pdf/2511.04126)
**Pipelines/pose :** [TennisProject](https://github.com/yastrebksv/TennisProject) · [tennis_analysis](https://github.com/abdullahtarek/tennis_analysis) · [tennis-tracking](https://github.com/ArtLabss/tennis-tracking) · [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)
**Coups/datasets :** [tennis_shot_recognition](https://github.com/antoinekeller/tennis_shot_recognition) · [TenniSet](https://www.semanticscholar.org/paper/4ee94572ae1d9c090fe81baa7236c7efbe1ca5b4) · [arXiv 2104.09907](https://arxiv.org/abs/2104.09907)
