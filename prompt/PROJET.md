# PROJET.md — Contexte & direction

> **Rôle de ce fichier.** Tu es mon chef de projet sur ce produit. Ce fichier porte la **vision, les décisions et le cap** — le « pourquoi » et la « direction ». Il complète `CLAUDE.md`, qui décrit le « comment » du code. En cas de désaccord entre les deux sur une *intention produit*, ce fichier fait foi ; sur un *détail technique du code*, `CLAUDE.md` fait foi.
>
> **Comment le maintenir.** Les sections 1 à 4 sont stables : ne les modifie que si une décision de fond change, et signale-le-moi avant. La section 5 (« État courant ») est la seule à mettre à jour régulièrement — garde-la courte, factuelle, sans recopier l'historique. Pas de bruit : si une info n'aide pas une décision future, elle n'a pas sa place ici.

---

## 1. Vision produit

Permettre à un **joueur de tennis amateur** de filmer son match avec son téléphone, d'uploader la vidéo en rentrant chez lui, et d'obtenir automatiquement des **données sur son jeu** : vidéo annotée, statistiques, et visualisations (vitesse, profondeur, placement, heatmaps).

Le besoin visé n'est pas servi correctement aujourd'hui : les outils d'analyse tennis existants ciblent le pro/broadcast. La cible amateur (« je filme, j'uploade, j'obtiens des stats ») est l'opportunité.

**Ce que le produit n'est pas (pour l'instant) :** pas de padel, pas de temps réel, pas d'analyse multi-sport. Tennis amateur, en différé. Ces pistes ont été écartées sciemment pour rester focalisé.

---

## 2. Décisions structurantes

Ces choix ont été arbitrés et ne sont pas à rediscuter sans raison forte.

- **Un seul modèle fine-tuné : la balle.** Le détecteur de balle (`yolo26_ball_best.pt`, déjà entraîné, ~94 % mAP) est indispensable et non remplaçable par un modèle générique : la balle de tennis est trop petite et trop rapide pour COCO. **Tout le reste est sur étagère** : joueurs et pose via YOLOv8 / YOLOv8-pose pré-entraînés. On ne réentraîne pas de modèle joueur/pose.

- **Le vrai chantier n'est pas l'entraînement, c'est la robustesse au monde réel.** Passer du broadcast à la vidéo de téléphone (tremblements, angle bas, court coupé, flou de mouvement, compression HEVC, lumière variable, courts voisins dans le cadre) est la difficulté centrale. L'effort va là, pas dans le ML pur.

- **L'homographie est le risque n°1.** Toutes les stats métriques (vitesse km/h, profondeur) dépendent d'une échelle pixels→mètre déduite des lignes du court. En amateur, le court est souvent partiellement visible et vu en biais — l'échelle se dégrade. Le fallback actuel (« court = 58 % de la hauteur d'image ») ne suffira pas. À fiabiliser en priorité.

- **Détection de rebonds = brique pivot.** Sans rebonds fiables, profondeur / qualité / stats avals ne tiennent pas. C'est le point faible connu de l'existant. À traiter avant d'enrichir les analytics.

- **Plusieurs angles acceptés, mais consigne de tournage souple.** On n'impose rien de strict à l'utilisateur, mais on le guide (« filme depuis le fond, le plus haut possible, court entier si possible ») pour rester dans les conditions où l'homographie fonctionne.

- **Architecture : un moteur, plusieurs habillages.** Le moteur de vision (vidéo → JSON de stats + vidéo annotée) doit être une fonction propre, **sans logique d'affichage ni de CLI mélangée dedans**. C'est l'investissement le plus rentable du projet : il rend les paliers suivants quasi gratuits côté vision.

---

## 3. Feuille de route (paliers)

Progression à risque croissant. On ne passe à un palier qu'une fois le précédent stable.

1. **Palier 1 — CLI local (phase actuelle).** Outil en ligne de commande, traitement sur la machine de dev. Objectif : fiabiliser le moteur et son pipeline.
2. **Palier 2 — Web (première version publique).** Upload via navigateur, traitement côté serveur. Introduit deux besoins nouveaux : **traitement asynchrone** (une analyse prend plusieurs minutes, l'utilisateur est notifié quand c'est prêt) et **coût GPU serveur** (poste de coût à optimiser : sous-échantillonnage des frames, hébergement adapté).
3. **Palier 3 — App mobile native (objectif final).** L'utilisateur filme et uploade depuis l'app, qui appelle l'API du palier 2.

Le moteur de vision est **identique** aux trois paliers ; seul l'habillage change. Même format de sortie partout.

---

## 4. Garde-fous & dette connue

À garder en tête en permanence ; détail complet dans le rapport d'analyse et `CLAUDE.md`.

- **Sécurité d'abord :** une clé API Roboflow a été exposée en clair dans le repo (`training/train_roboflow_yolo26.py`). À révoquer et basculer en variable d'environnement si pas déjà fait.
- **Pas de hardcoding :** noms de joueurs (« Alcaraz »/« Sinner ») et FPS codés en dur à retirer — ça contredit le principe « zéro hardcoding » et empêche la généralisation.
- **Environnement fragile :** Python 3.14 sans lockfile, `install.sh` cassé (fichiers référencés absents). Reproductibilité à sécuriser.
- **Aucun test automatisé.** Pour un algo géométrique qu'on tune en continu (rebonds), c'est risqué. Une **vérité terrain rebonds** (vidéos avec rebonds annotés à la main) est nécessaire pour mesurer les progrès objectivement.
- **Double pipeline & code mort :** `main.py` (legacy) et les chemins ML jamais utilisés (X3D/SlowFast, pose ResNet) alourdissent la base. Point d'entrée canonique à faire vivre : `run_pipeline_8s.py`.

---

## 5. État courant

> Seule section à mettre à jour régulièrement. Trois lignes par champ maximum. Décrire l'état, pas l'historique.

- **Palier actif :** 1 (CLI local).
- **En cours :** chantier précision terminé sur angle broadcast — rebonds refondus (curve-fit, F1 0.30 → 0.67 end-to-end / 0.80 mesuré sur cache, vs vérité terrain `felix`), vraie homographie keypoints→mètres ITF pour vitesse/profondeur (`--homography`, fallback scalaire si court non localisable), 1er test de non-régression, début d'extraction du moteur (`vision/bounce.py`).
- **Prochain jalon :** généralisation aux angles obliques/amateurs — le cœur du produit. Les modèles pré-entraînés (balle heatmap WASB, court keypoints) marchent en broadcast mais décrochent sur angle court-level (felix : 3/14 keypoints). Lever = détecteur de court angle-robuste + tracker balle heatmap, tous deux à **fine-tuner sur GPU**. Puis finir le moteur propre vidéo→JSON de stats.
- **Bloqueurs :** pas de bloqueur sur broadcast. Pour l'angle amateur : pas de GPU local pour fine-tuner (Mac/MPS) ; les poids pré-entraînés publics ne suffisent pas hors broadcast (mesuré).
- **Dernière décision prise :** 2026-06-19 — adopter le curve-fit + l'homographie keypoints comme socle précision (broadcast OK) ; WASB et court-detector pré-entraînés = NO-GO sur angle oblique sans fine-tuning (scaffolding conservé). Garder YOLO+Kalman comme tracker de production.
