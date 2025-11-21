# Training YOLOv11 - Fine-tuning pour Tennis Analysis

Ce dossier contient tous les scripts nécessaires pour fine-tuner YOLOv11 sur vos propres datasets de tennis.

## 📁 Structure

```
training/
├── README.md                      # Ce fichier
├── train_yolov11.py              # Script principal d'entraînement
├── prepare_data.py               # Préparation et validation des datasets
├── convert_coco_to_yolo.py       # Conversion COCO → YOLO
├── dataset_config.yaml           # Configuration centralisée
│
├── dataset/                      # Vos datasets
│   ├── raquette/                # Dataset raquettes (format YOLO)
│   ├── tennis ball/             # Dataset balles (format YOLO)
│   ├── tennis_court_id/         # Dataset courts (format COCO)
│   └── tennis_court_id_yolo/    # Dataset courts converti (auto-généré)
│
├── runs/                         # Résultats d'entraînement (auto-généré)
│   ├── raquette_20241120_143022/
│   ├── tennis_ball_20241120_150315/
│   └── ...
│
└── models/                       # Modèles entraînés (auto-généré)
    ├── raquette_best.pt
    ├── tennis_ball_best.pt
    └── tennis_court_best.pt
```

## 🎯 Datasets Utilisés

### 1. Dataset Raquette
- **Format** : YOLO (natif)
- **Classes** : 1 (racket)
- **Description** : Images annotées de raquettes de tennis

### 2. Dataset Tennis Ball
- **Format** : YOLO (natif)
- **Classes** : 1 (tennis_ball)
- **Description** : Images annotées de balles de tennis

### 3. Dataset Tennis Court ID
- **Format** : COCO (converti automatiquement en YOLO)
- **Classes** : 1 (tennis_court)
- **Description** : Images annotées de courts de tennis

## 🚀 Utilisation Rapide

### Option 1 : Entraînement Automatique (Recommandé)

```bash
# Étape 1 : Préparer les données
python training/prepare_data.py

# Étape 2 : Entraîner tous les modèles
python training/train_yolov11.py

# Étape 3 : Vos modèles sont dans training/
# - raquette_best.pt
# - tennis_ball_best.pt
# - tennis_court_best.pt
```

### Option 2 : Entraînement Dataset par Dataset

```bash
# Entraîner seulement le modèle de raquettes
python training/train_yolov11.py --single-dataset raquette

# Entraîner seulement le modèle de balles
python training/train_yolov11.py --single-dataset tennis_ball

# Entraîner seulement le modèle de courts
python training/train_yolov11.py --single-dataset tennis_court
```

## ⚙️ Options Avancées

### Choisir le Modèle de Base

```bash
# YOLOv11 Nano (rapide, moins précis) - Par défaut
python training/train_yolov11.py --model yolo11n.pt

# YOLOv11 Small (bon compromis)
python training/train_yolov11.py --model yolo11s.pt

# YOLOv11 Medium (plus précis)
python training/train_yolov11.py --model yolo11m.pt

# YOLOv11 Large (très précis, lent)
python training/train_yolov11.py --model yolo11l.pt

# YOLOv11 XLarge (maximum précision, très lent)
python training/train_yolov11.py --model yolo11x.pt
```

### Ajuster les Hyperparamètres

```bash
# Plus d'époques pour meilleure convergence
python training/train_yolov11.py --epochs 200

# Batch size plus petit si mémoire limitée
python training/train_yolov11.py --batch-size 8

# Batch size plus grand si GPU puissant
python training/train_yolov11.py --batch-size 32

# Utiliser CPU au lieu de GPU
python training/train_yolov11.py --device cpu

# Forcer l'utilisation du GPU
python training/train_yolov11.py --device cuda

# Utiliser Apple Silicon (M1/M2/M3)
python training/train_yolov11.py --device mps
```

### Combinaisons Complètes

```bash
# Entraînement haute précision (GPU requis)
python training/train_yolov11.py \
    --model yolo11l.pt \
    --epochs 200 \
    --batch-size 32 \
    --device cuda

# Entraînement rapide pour tester (CPU OK)
python training/train_yolov11.py \
    --model yolo11n.pt \
    --epochs 50 \
    --batch-size 8 \
    --device cpu \
    --single-dataset tennis_ball
```

## 📊 Préparation des Données

Le script `prepare_data.py` effectue automatiquement :

1. ✅ Vérification de la structure des datasets
2. ✅ Conversion COCO → YOLO pour tennis_court_id
3. ✅ Validation des images et labels
4. ✅ Génération de statistiques
5. ✅ Création de visualisations
6. ✅ Rapport détaillé (dataset_report.yaml)

### Utilisation

```bash
# Préparation complète
python training/prepare_data.py

# Vérification seulement (sans conversion)
python training/prepare_data.py --check-only

# Avec visualisations
python training/prepare_data.py --visualize
```

### Résultats Générés

```
training/
├── dataset_report.yaml          # Rapport détaillé
├── visualizations/              # Graphiques
│   └── dataset_statistics.png
└── dataset/
    └── tennis_court_id_yolo/   # Dataset converti
```

## 🔄 Conversion COCO → YOLO

Si vous avez un dataset au format COCO, utilisez le convertisseur :

```bash
# Conversion automatique
python training/convert_coco_to_yolo.py \
    --input training/dataset/tennis_court_id \
    --output training/dataset/tennis_court_id_yolo

# Copier les images (recommandé)
python training/convert_coco_to_yolo.py \
    --input training/dataset/tennis_court_id \
    --output training/dataset/tennis_court_id_yolo \
    --copy-images
```

### Format COCO Attendu

```
tennis_court_id/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── _annotations.coco.json
├── valid/
│   ├── images/
│   └── _annotations.coco.json
└── test/
    ├── images/
    └── _annotations.coco.json
```

### Format YOLO Généré

```
tennis_court_id_yolo/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── labels/
│       ├── image1.txt
│       └── image2.txt
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── dataset.yaml
```

## 📈 Suivi de l'Entraînement

### Pendant l'Entraînement

Le script affiche en temps réel :
- Progression des époques
- Loss (box, cls, dfl)
- Métriques (Précision, Recall, mAP)
- Temps par époque

### Après l'Entraînement

Chaque run génère :

```
runs/raquette_20241120_143022/
├── weights/
│   ├── best.pt              # Meilleur modèle
│   ├── last.pt              # Dernier checkpoint
│   └── epoch_*.pt           # Checkpoints intermédiaires
├── results.csv              # Métriques par époque
├── results.png              # Graphiques d'entraînement
├── confusion_matrix.png     # Matrice de confusion
├── F1_curve.png            # Courbe F1
├── PR_curve.png            # Courbe Précision-Recall
└── val_batch*.jpg          # Prédictions de validation
```

### Visualiser les Résultats

```python
import pandas as pd
import matplotlib.pyplot as plt

# Charger les résultats
results = pd.read_csv('runs/raquette_20241120_143022/results.csv')

# Afficher les courbes
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(results['train/box_loss'], label='Train')
plt.plot(results['val/box_loss'], label='Val')
plt.title('Box Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(results['metrics/mAP50(B)'])
plt.title('mAP@50')

plt.show()
```

## 🎯 Utiliser les Modèles Entraînés

### Option 1 : Utilisation Directe

```python
from ultralytics import YOLO

# Charger le modèle
model = YOLO('training/raquette_best.pt')

# Prédire sur une image
results = model.predict('image.jpg', conf=0.5)

# Afficher
results[0].show()
```

### Option 2 : Intégration au Pipeline

Modifiez `config.yaml` :

```yaml
yolo:
  # Modèles personnalisés
  custom_models:
    racket: "training/raquette_best.pt"
    ball: "training/tennis_ball_best.pt"
    court: "training/tennis_court_best.pt"

  # Utiliser les modèles personnalisés
  use_custom_models: true
```

Puis :

```python
from main import TennisAnalysisPipeline

pipeline = TennisAnalysisPipeline("config.yaml")
results = pipeline.run(video_path="match.mp4")
```

## 📝 Configuration Centralisée

Le fichier `dataset_config.yaml` centralise tous les paramètres :

```yaml
datasets:
  raquette:
    num_classes: 1
    class_names: ["racket"]
    train_params:
      epochs: 100
      batch_size: 16
      imgsz: 640

  tennis_ball:
    num_classes: 1
    class_names: ["tennis_ball"]
    # ...

training:
  base_models:
    nano: "yolo11n.pt"
    small: "yolo11s.pt"
    medium: "yolo11m.pt"

  optimizer:
    type: "AdamW"
    lr0: 0.001
    # ...
```

### Modifier la Configuration

```yaml
# Augmenter les époques
datasets:
  raquette:
    train_params:
      epochs: 200  # Au lieu de 100

# Changer le learning rate
training:
  optimizer:
    lr0: 0.0005  # Au lieu de 0.001
```

## 💡 Conseils et Bonnes Pratiques

### 1. Première Utilisation

```bash
# Testez sur un seul dataset avec peu d'époques
python training/train_yolov11.py \
    --single-dataset tennis_ball \
    --epochs 10 \
    --batch-size 8
```

### 2. Dataset de Qualité

- ✅ Annotations précises
- ✅ Images variées (angles, éclairages)
- ✅ Ratio train/val/test : 80/10/10
- ✅ Minimum 100 images par classe

### 3. Hyperparamètres

| GPU | Modèle | Batch Size | Temps (100 epochs) |
|-----|--------|------------|-------------------|
| CPU | nano | 8 | ~2h |
| RTX 3060 | small | 16 | ~30min |
| RTX 3090 | medium | 32 | ~20min |
| A100 | large | 64 | ~15min |

### 4. Early Stopping

Le modèle s'arrête automatiquement si pas d'amélioration pendant 50 époques (paramètre `patience`).

### 5. Surapprentissage (Overfitting)

Signes :
- Train loss diminue mais val loss augmente
- mAP train > mAP val

Solutions :
- Augmenter le dataset
- Plus d'augmentation de données
- Modèle plus petit
- Plus de régularisation

## 🔧 Résolution de Problèmes

### Erreur : CUDA out of memory

```bash
# Solution 1 : Batch size plus petit
python training/train_yolov11.py --batch-size 4

# Solution 2 : Modèle plus petit
python training/train_yolov11.py --model yolo11n.pt

# Solution 3 : Utiliser CPU
python training/train_yolov11.py --device cpu
```

### Erreur : Dataset non trouvé

```bash
# Vérifiez la structure
python training/prepare_data.py --check-only

# Les datasets doivent être dans training/dataset/
ls -la training/dataset/
```

### Mauvaises Performances

```bash
# 1. Plus d'époques
python training/train_yolov11.py --epochs 200

# 2. Modèle plus grand
python training/train_yolov11.py --model yolo11m.pt

# 3. Vérifier les données
python training/prepare_data.py
```

### Conversion COCO Échoue

```bash
# Vérifier le format JSON
python -c "import json; json.load(open('dataset/tennis_court_id/train/_annotations.coco.json'))"

# Conversion manuelle avec logs
python training/convert_coco_to_yolo.py --input ... --output ...
```

## 📊 Métriques de Performance

### mAP (Mean Average Precision)

- **mAP@50** : Précision moyenne avec IoU > 0.5
- **mAP@50-95** : Précision moyenne sur IoU de 0.5 à 0.95

| Qualité | mAP@50 | mAP@50-95 |
|---------|--------|-----------|
| Excellent | > 0.90 | > 0.70 |
| Bon | 0.80-0.90 | 0.60-0.70 |
| Acceptable | 0.70-0.80 | 0.50-0.60 |
| Faible | < 0.70 | < 0.50 |

### Objectifs par Dataset

| Dataset | mAP@50 Cible | Difficulté |
|---------|--------------|------------|
| Raquette | > 0.85 | Moyenne |
| Tennis Ball | > 0.90 | Facile |
| Tennis Court | > 0.80 | Difficile |

## 🚀 Workflow Complet

### Étape 1 : Préparation

```bash
# Vérifier les datasets
python training/prepare_data.py

# Vérifier le rapport
cat training/dataset_report.yaml
```

### Étape 2 : Test Rapide

```bash
# Entraînement court pour valider
python training/train_yolov11.py \
    --single-dataset tennis_ball \
    --epochs 10
```

### Étape 3 : Entraînement Complet

```bash
# Tous les datasets
python training/train_yolov11.py --epochs 100
```

### Étape 4 : Évaluation

```bash
# Tester les modèles
python -c "
from ultralytics import YOLO
model = YOLO('training/raquette_best.pt')
results = model.val()
print(f'mAP@50: {results.box.map50:.3f}')
print(f'mAP@50-95: {results.box.map:.3f}')
"
```

### Étape 5 : Déploiement

```bash
# Copier dans le projet principal
cp training/raquette_best.pt models/
cp training/tennis_ball_best.pt models/
cp training/tennis_court_best.pt models/

# Mettre à jour config.yaml
# use_custom_models: true
```

## 📚 Ressources

- [Documentation YOLOv11](https://docs.ultralytics.com/)
- [Guide d'annotation](https://roboflow.com/annotate)
- [Dataset Tennis sur Roboflow](https://universe.roboflow.com/search?q=tennis)
- [Ultralytics HUB](https://hub.ultralytics.com/) - Entraînement cloud

## 🎓 Exemples de Commandes

```bash
# Exemple 1 : Entraînement rapide CPU
python training/train_yolov11.py --device cpu --epochs 50 --batch-size 8

# Exemple 2 : Entraînement haute qualité GPU
python training/train_yolov11.py --model yolo11l.pt --epochs 200 --batch-size 32 --device cuda

# Exemple 3 : Un seul dataset sur Apple Silicon
python training/train_yolov11.py --single-dataset raquette --device mps --epochs 100

# Exemple 4 : Tous les datasets avec modèle moyen
python training/train_yolov11.py --model yolo11m.pt --epochs 150 --batch-size 16
```

## 🎯 Prochaines Étapes

Après l'entraînement :

1. Évaluer les modèles sur vos propres vidéos
2. Ajuster les seuils de confiance dans `config.yaml`
3. Réentraîner avec plus de données si nécessaire
4. Partager vos résultats !

---

**Bon entraînement !** 🎾🚀

Pour toute question, consultez les logs générés ou le README principal du projet.