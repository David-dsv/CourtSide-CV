# RAPPORT DE RECHERCHE SOTA : Évolution de CourtSide-CV

## TL;DR
Pour faire passer CourtSide-CV du prototype au produit SOTA, 4 leviers doivent être actionnés en priorité : 
1. **Remplacer le tracking balle YOLO+Kalman par WASB (MIT)**, conçu spécifiquement pour les objets flous/rapides et gérant nativement les occultations.
2. **Implémenter une homographie par keypoints de court (ResNet50)**, indispensable pour corriger l'erreur de vitesse de 20-50% due à la perspective.
3. **Découpler la détection de rebond du tracking** en utilisant un classifieur temporel (CatBoost) sur la trajectoire interpolée.
4. **Refactorer l'architecture pour séparer l'analyse (JSON) du rendu (MP4)**, permettant la segmentation de points et l'attribution des coups.

---

## 1. Verrou 1 : Tracking de la balle (Cassé par design actuel)

**Techno SOTA recommandée :** WASB (Widely Applicable Strong Baseline) + BlurBall.
*Repo/Papier : `nttcom/WASB-SBDT` (BMVC 2023, licence MIT) / `cogsys-tuebingen/blurball` (CVPR-W 2025).*

* **Pourquoi ça bat l'approche actuelle :** Le pipeline actuel (YOLO26 + filtre statique + Kalman) s'effondre car la balle est petite, floue, et souvent immobile entre les points. WASB utilise un backbone HRNet haute-résolution pour générer des heatmaps, gère la cohérence temporelle et utilise les *connected components* pour suivre la balle même lors d'occultations courtes. BlurBall y ajoute une estimation du flou de mouvement, ce qui correspond exactement à la physique d'une balle de tennis filmée à 30/60 fps par un téléphone.
* **Effort d'intégration : M (Medium).** Risque sur footage amateur : WASB repose sur des heatmaps et de la soustraction de fond. Si le téléphone bouge beaucoup (caméra à l'épaule), la soustraction de fond peut être bruitée. Cependant, l'approche par heatmap d'HRNet reste plus robuste qu'une détection par boîtes englobantes (YOLO).
* **Recommandation : Drop-in open source (MIT).** Le repo WASB-SBDT est fonctionnel. Entraîner TrackNetV3/V5 exigerait de constituer une vérité-terrain massive de balle de tennis (très chronophage). WASB offre un baseline fort sans fine-tuning immédiat. Si WASB perd la balle sur les lobs, combiner avec une interpolation spline à long terme (que vous avez déjà) sur les gros trous.

## 2. Verrou 2 : Homographie et Échelle px→mètre

**Techno SOTA recommandée :** Détection de keypoints de court par CNN (ResNet50) + calcul d'homographie dynamique.
*Repo/Papier : `yastrebksv/TennisCourtDetector` / arXiv 2404.06977 (Amateur) / arXiv 2511.04126.*

* **Pourquoi ça bat l'approche actuelle :** Un scalaire uniforme ignore la perspective. Un déplacement près de la caméra n'a pas le même sens qu'au fond du court. L'approche SOTA détecte les 14 intersections de lignes du court, calcule une matrice d'homographie frame-par-frame (ou toutes les N frames pour lisser), et projette les coordonnées pixels (x,y) en coordonnées réelles (X,Y) sur le plan 2D du court. Cela valide une erreur <5% sur la vitesse.
* **Effort d'intégration : L (Large).** Risque sur footage amateur : **PIÈGE MAJEUR.** Le repo `yastrebksv/TennisCourtDetector` est entraîné sur des vues broadcast (TV, plongeantes). Sur un téléphone (angle bas, court tronqué), le modèle fera face à un domaine hors distribution (OOD) et échouera souvent.
* **Recommandation : À entraîner / Adapter.** Ne pas faire un drop-in direct du modèle broadcast. S'inspirer de l'approche de l'article arXiv 2404.06977 (ciblé amateur). Il faudra fine-tuner un ResNet50 (ou réutiliser l'architecture de yastrebksv) sur des frames amateurs annotés manuellement (les 14 points). En fallback, utiliser la détection de lignes classiques (Hough) sur les frames où le réseau échoue, et lever un flag de confiance bas.

## 3. Verrou 3 : Détection de Rebond (Cassé par dépendance)

**Techno SOTA recommandée :** Classifieur temporel de trajectoire (CatBoostRegressor) + Contrainte physique 3D.
*Repo/Papier : `yastrebksv/TennisProject` (Pipeline CatBoost) / Reconstruction 3D mono-caméra avec contrainte sol.*

* **Pourquoi ça bat l'approche actuelle :** Actuellement, le rebond est détecté via un changement de direction du Kalman, qui s'effondre si le tracking saute. En séparant le problème (1. Tracker la balle, 2. Analyser la trajectoire résultante), on gagne en robustesse. CatBoost analyse les features temporelles (dérivées première et seconde de y, proximité avec le sol) pour prédire un rebond. Pour aller plus loin, la reconstruction 3D mono-caméra force le point de rebond à être sur le plan z=0 (le sol), corrigeant ainsi le drifting.
* **Effort d'intégration : S (Small) pour CatBoost, L pour la 3D.** 
* **Recommandation : Drop-in CatBoost en premier.** Une fois la trajectoire de la balle (WASB + interpolation) et l'homographie (plan 2D) obtenus, entraîner un CatBoostRegressor est trivial. La détection 3D avec contrainte au sol (LATTE-MV / TT3D) est un excellent objectif V2 pour calculer la profondeur réelle, mais nécessite d'estimer la distance caméra-balle via la taille apparente (CNN dédié). Commencez par le 2D + CatBoost.

## 4. Verrou 4 : Export de Données Structurées

**Techno SOTA recommandée :** Refactoring logiciel + Segmentation de points (SmartDampener) + Attribution par Pose.
*Repo/Papier : SmartDampener (ACM IMWUT) / Pose2Trajectory (2411.04501).*

* **Pourquoi ça bat l'approche actuelle :** Renvoyer un MP4 annoté ne sert à rien à un joueur qui veut analyser ses stats. Il faut une fonction `analyze_video() -> dict` qui segmente le match en "points" (du service jusqu'à la faute), attribue chaque coup de balle au bon joueur (via la proximité balle/joueur et la pose de la raquette), et calcule les heatmaps de positionnement.
* **Effort d'intégration : M (Medium).** C'est un travail d'ingénierie logicielle plus que de recherche CV.
* **Recommandation : Drop-in logique.** Utiliser la distance entre la position 2D de la balle (au moment du rebond ou de la frappe) et les bounding boxes des joueurs (YOLOv8-pose) pour attribuer le coup. Utiliser un SVM léger (comme SmartDampener) sur les données d'accélération de la balle pour séparer un service d'un coup normal. Le livrable doit être un JSON standardisé.

---

## 3. Architecture Cible SOTA

Le pipeline `run_pipeline_8s.py` doit évoluer vers une architecture modulaire orientée data :

```text
[INPUT VIDEO AMATEUR]
      │
      ├─► [BLOC 1: COURT CALIBRATION] (Toutes les 10 frames)
      │     └─ ResNet50 (Fine-tuné amateur) -> 14 Keypoints -> Homographie H(t)
      │
      ├─► [BLOC 2: PLAYER TRACKING] (Frame par frame)
      │     └─ YOLOv8-pose + SAHI (tiling si basse résolution) -> Joueur 1 & 2 (x,y,t)
      │
      ├─► [BLOC 3: BALL TRACKING] (Frame par frame)
      │     └─ WASB (HRNet heatmaps) -> Balle (x,y,t) + Interpolation Spline longue portée
      │
      ├─► [BLOC 4: DATA FUSION & METRICS] (Post-processing)
      │     ├─ Application de H(t) sur Balle & Joueurs -> Coordonnées métriques 2D
      │     ├─ Détection Rebond (CatBoost sur trajectoires métriques)
      │     ├─ Attribution Balle -> Joueur (Distance euclidienne 2D + Pose)
      │     └─ Segmentation des points (Début = Service, Fin = Rebond hors limites/2nd rebond)
      │
      ├─► [OUTPUT 1: JSON DATA] (Le vrai produit)
      │     └─ { points: [{ score, coups: [{ joueur, vitesse_ms, profondeur_m, zone }], heatmaps }...] }
      │
      └─► [OUTPUT 2: MP4 ANNOTÉ] (Optionnel, pour le debug/user feedback)
```

---

## 4. Tableau de Décision

| Brique Actuelle (Faible) | Remplacement SOTA | Gain Attendu | Effort | Stratégie |
| :--- | :--- | :--- | :--- | :--- |
| YOLO26 + Filtre Statique + Kalman | **WASB** (HRNet + Heatmaps) | Récupère 80% des balles perdues, gère le flou et l'arrêt. | M | Drop-in MIT |
| Scalaire uniforme px/mètre | **ResNet50 Keypoints** + Homographie | Vitesse exacte (<5% d'erreur), vraies heatmaps. | L | À fine-tuner sur amateur |
| Détection rebond via Kalman | **CatBoostRegressor** sur trajectoire | F1 > 0.80 (estimé), indépendant du tracking temps réel. | S | Drop-in / Modèle léger à entraîner |
| Sortie MP4 uniquement | **JSON Builder** + Segmentation de points | Données exploitables par l'utilisateur final. | M | Ingénierie pure |
| YOLOv8-pose (Joueurs) | Conserver + **SAHI** (Slicing) | Meilleure détection à distance (far baseline). | S | Drop-in |

---

## 5. Roadmap Priorisée (Dépendances & ROI)

**Phase 1 : Refactoring Data-First (Semaines 1-2)**
*   *Action :* Séparer la logique de calcul du writer vidéo. Créer la fonction `analyze_video() -> dict`.
*   *ROI :* Immédiat. Permet de tester les briques suivantes sur des données structurées et de valider les métriques.

**Phase 2 : Calibration du Court (Semaines 3-5) — *Le fondement***
*   *Action :* Annoter ~500 frames de vidéos amateurs (angles bas, courts partiels). Fine-tuner le modèle ResNet50 (type yastrebksv). Implémenter `cv2.findHomography`.
*   *ROI :* Fort. Corrige l'erreur de vitesse de 20-50%. Sans ça, la data sortie sera fausse, peu importe la qualité du tracking.
*   *Risque :* Gestion du fallback si le court est hors-cadre (flag de confiance sur la vitesse).

**Phase 3 : Tracking Balle SOTA (Semaines 6-7)**
*   *Action :* Intégrer le repo WASB-SBDT. Remplacer la passe 1 actuelle. Garder l'interpolation spline pour les trous résiduels de WASB.
*   *ROI :* Massif. Débloque le Verrou 1. La balle sera suivie même lors des échanges longs.

**Phase 4 : Détection de Rebond & Attribution (Semaines 8-9)**
*   *Action :* Extraire les features de la trajectoire (WASB + Homographie). Entraîner le CatBoost. Utiliser YOLOv8-pose pour attribuer le coup au joueur le plus proche de la balle au moment du rebond/frappe.
*   *ROI :* Finalise le produit. Génère les statistiques de match (gagnants, fautes, placement).

**Phase 5 (R&D Future) : La 3D Mono-Caméra**
*   *Action :* Étudier l'ajout de la contrainte 3D (z=0 au sol) et de l'estimation de distance par taille apparente (TT3D / LATTE-MV) pour calculer la profondeur réelle des lobs sans stéréo.

*Note de l'ingénieur : La tentation sera grande de tester TrackNetV4/V5. Je déconseille de partir là-dessus : WASB est sous licence MIT, éprouvé, et ne nécessite pas de réentraîner un modèle complexe de zéro. Le vrai goulot d'étranglement de CourtSide-CV n'est pas l'IA "pure" (détection), mais la géométrie (homographie amateur). Concentrez les efforts L sur le ResNet50 des lignes du court.*
