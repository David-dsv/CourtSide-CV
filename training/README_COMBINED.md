## 🎯 Dataset Combiné Multi-Classe

Guide pour créer et entraîner un **modèle YOLOv11 unique** qui détecte **10 classes** :
- 1 classe : **Raquette** (racket)
- 1 classe : **Balle** (tennis_ball)
- 8 classes : **Zones du court** (court, net, service boxes, etc.)

## 🚀 Processus en 2 Étapes

### Étape 1 : Combiner les Datasets

```bash
python training/combine_datasets.py \
    --racket-max 500 \
    --ball-max 500 \
    --court-max 200
```

**Ce que fait ce script :**
- ✅ Combine les 3 datasets sources
- ✅ Remape les IDs de classe (0-9 au lieu de 0, 0, 0-7)
- ✅ Crée la structure YOLO correcte
- ✅ Génère `data.yaml` avec les 10 classes
- ✅ Évite les collisions de noms de fichiers

**Résultat :**
```
training/dataset_combined/
├── data.yaml              # 10 classes
├── train/
│   ├── images/            # ~800-1000 images
│   └── labels/            # Labels remappés
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Étape 2 : Entraîner le Modèle Combiné

```bash
python training/train_combined.py --epochs 150 --batch-size 16
```

**Durée estimée :** ~45-60 minutes sur M4 Pro

## 📊 Classes du Modèle Combiné

| ID | Nom | Source |
|----|-----|--------|
| 0 | `racket` | Dataset Raquette |
| 1 | `tennis_ball` | Dataset Tennis Ball |
| 2 | `bottom-dead-zone` | Dataset Tennis Court |
| 3 | `court` | Dataset Tennis Court |
| 4 | `left-doubles-alley` | Dataset Tennis Court |
| 5 | `left-service-box` | Dataset Tennis Court |
| 6 | `net` | Dataset Tennis Court |
| 7 | `right-doubles-alley` | Dataset Tennis Court |
| 8 | `right-service-box` | Dataset Tennis Court |
| 9 | `top-dead-zone` | Dataset Tennis Court |

## ⚙️ Options Avancées

### Combiner avec Plus ou Moins d'Images

```bash
# Configuration minimale (rapide pour tester)
python training/combine_datasets.py \
    --racket-max 100 \
    --ball-max 100 \
    --court-max 50

# Configuration maximale (meilleure performance)
python training/combine_datasets.py \
    --racket-max 1000 \
    --ball-max 800 \
    --court-max 299
```

### Entraîner avec Modèle Plus Grand

```bash
# YOLOv11 Small (meilleure précision)
python training/train_combined.py \
    --model yolo11s.pt \
    --epochs 200 \
    --batch-size 16

# YOLOv11 Medium (encore meilleur)
python training/train_combined.py \
    --model yolo11m.pt \
    --epochs 200 \
    --batch-size 16
```

### Forcer CPU

```bash
python training/train_combined.py \
    --device cpu \
    --batch-size 8 \
    --epochs 100
```

## 💻 Utiliser le Modèle Combiné

### Python API

```python
from ultralytics import YOLO

# Charger le modèle combiné
model = YOLO('training/tennis_combined_best.pt')

# Prédire sur une image
results = model.predict('match.jpg', conf=0.3)

# Afficher
results[0].show()

# Obtenir les détections par classe
for box in results[0].boxes:
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    confidence = box.conf[0]
    print(f"{class_name}: {confidence:.2%}")
```

### Filtrer par Type d'Objet

```python
from ultralytics import YOLO

model = YOLO('training/tennis_combined_best.pt')
results = model.predict('match.jpg')

# Séparer par type
rackets = []
balls = []
court_zones = []

for box in results[0].boxes:
    class_id = int(box.cls[0])
    class_name = model.names[class_id]

    if class_name == 'racket':
        rackets.append(box)
    elif class_name == 'tennis_ball':
        balls.append(box)
    else:
        court_zones.append(box)

print(f"Raquettes: {len(rackets)}")
print(f"Balles: {len(balls)}")
print(f"Zones court: {len(court_zones)}")
```

### Analyse Avancée

```python
from ultralytics import YOLO
import cv2

model = YOLO('training/tennis_combined_best.pt')

# Prédire
results = model.predict('match.jpg')

# Analyser les détections
detections = {
    'rackets': [],
    'balls': [],
    'court': {}
}

for box in results[0].boxes:
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    x1, y1, x2, y2 = box.xyxy[0]
    conf = box.conf[0]

    if class_name == 'racket':
        detections['rackets'].append({
            'bbox': (x1, y1, x2, y2),
            'conf': float(conf)
        })
    elif class_name == 'tennis_ball':
        detections['balls'].append({
            'bbox': (x1, y1, x2, y2),
            'conf': float(conf)
        })
    else:  # Zone du court
        detections['court'][class_name] = {
            'bbox': (x1, y1, x2, y2),
            'conf': float(conf)
        }

# Afficher le résumé
print(f"Analyse du match:")
print(f"  Joueurs détectés: {len(detections['rackets'])}")
print(f"  Balles visibles: {len(detections['balls'])}")
print(f"  Zones court: {list(detections['court'].keys())}")
```

## 📈 Performances Attendues

### Avec Configuration Standard (500/500/200 images, 150 époques)

| Classe | mAP@50 Attendu |
|--------|----------------|
| Raquette | 65-75% |
| Balle | 65-75% |
| Zones Court | 55-70% |
| **Global** | **60-75%** |

### Avantages du Modèle Combiné

✅ **Un seul modèle** au lieu de 3
✅ **Inférence plus rapide** (1 passage au lieu de 3)
✅ **Cohérence** : Le modèle apprend les relations entre objets
✅ **Plus simple** à déployer et maintenir
✅ **Meilleure compréhension** du contexte (balle près raquette, etc.)

### Inconvénients Potentiels

⚠️ Performances légèrement inférieures par classe (vs modèles spécialisés)
⚠️ Plus difficile à débugger
⚠️ Plus long à entraîner

## 🔍 Vérifier le Dataset Combiné

Après la combinaison, vérifiez :

```bash
# Compter les images
find training/dataset_combined -name "*.jpg" -o -name "*.png" | wc -l

# Voir la structure
tree training/dataset_combined -L 2

# Vérifier les classes
cat training/dataset_combined/data.yaml
```

## 🛠️ Résolution de Problèmes

### Dataset combiné non trouvé

```bash
# Recréer le dataset
python training/combine_datasets.py
```

### Erreur de remapping des classes

Les IDs de classe sont automatiquement remappés :
- Dataset source : IDs locaux (0, 1, 2...)
- Dataset combiné : IDs globaux (0-9)

Le script gère cela automatiquement.

### Performances faibles sur une classe

Augmentez le nombre d'images pour cette classe :

```bash
python training/combine_datasets.py \
    --racket-max 800 \
    --ball-max 800 \
    --court-max 300
```

## 📊 Workflow Complet

```bash
# 1. Combiner les datasets
python training/combine_datasets.py \
    --racket-max 500 \
    --ball-max 500 \
    --court-max 200

# 2. Vérifier
cat training/dataset_combined/data.yaml

# 3. Entraîner (config standard)
python training/train_combined.py --epochs 150

# 4. Ou entraîner (config haute qualité)
python training/train_combined.py \
    --model yolo11s.pt \
    --epochs 200 \
    --batch-size 16

# 5. Tester
python -c "
from ultralytics import YOLO
model = YOLO('training/tennis_combined_best.pt')
results = model.predict('test_image.jpg')
results[0].show()
print(f'Classes: {model.names}')
"
```

## 🎯 Recommandation

Pour une **première utilisation**, commencez avec :

```bash
# Combinaison équilibrée
python training/combine_datasets.py \
    --racket-max 500 \
    --ball-max 500 \
    --court-max 200

# Entraînement standard
python training/train_combined.py \
    --model yolo11n.pt \
    --epochs 150 \
    --batch-size 16
```

Durée totale : ~60 minutes
Performances attendues : mAP@50 global de 60-70%

---

**Prêt à créer votre modèle multi-classe unique !** 🎾🚀