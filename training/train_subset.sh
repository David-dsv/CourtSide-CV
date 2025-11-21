#!/bin/bash
# Script de lancement rapide pour l'entraînement avec sous-ensembles
# Usage: bash training/train_subset.sh

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  ENTRAÎNEMENT YOLOV11 - SOUS-ENSEMBLES                         ║"
echo "║  - Raquette: 300 images                                        ║"
echo "║  - Tennis Ball: 200 images                                     ║"
echo "║  - Tennis Court: 100 images                                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Étape 1 : Convertir le dataset COCO si nécessaire
if [ ! -d "training/dataset/tennis_court_id_yolo" ]; then
    echo "📦 Étape 1/4 : Conversion COCO → YOLO pour tennis_court_id..."
    python training/convert_coco_to_yolo.py \
        --input training/dataset/tennis_court_id \
        --output training/dataset/tennis_court_id_yolo \
        --copy-images

    if [ $? -ne 0 ]; then
        echo "❌ Erreur lors de la conversion"
        exit 1
    fi
    echo "✓ Conversion terminée"
else
    echo "✓ Dataset tennis_court_id_yolo déjà converti"
fi

echo ""

# Étape 2 : Créer les sous-ensembles
echo "📊 Étape 2/4 : Création des sous-ensembles..."
python training/create_subset.py \
    --raquette 300 \
    --tennis-ball 200 \
    --tennis-court 100

if [ $? -ne 0 ]; then
    echo "❌ Erreur lors de la création des sous-ensembles"
    exit 1
fi

echo "✓ Sous-ensembles créés"
echo ""

# Étape 3 : Préparer et valider les données
echo "🔍 Étape 3/4 : Validation des sous-ensembles..."
python training/prepare_data.py --base-dir training/dataset_subset

if [ $? -ne 0 ]; then
    echo "⚠️  Avertissement : Erreurs détectées pendant la validation"
fi

echo ""

# Étape 4 : Lancer l'entraînement
echo "🚀 Étape 4/4 : Lancement de l'entraînement..."
echo ""
echo "Vous pouvez maintenant lancer l'entraînement avec:"
echo ""
echo "  # Entraîner tous les datasets:"
echo "  python training/train_yolov11_subset.py"
echo ""
echo "  # OU entraîner un seul dataset:"
echo "  python training/train_yolov11_subset.py --single-dataset raquette"
echo "  python training/train_yolov11_subset.py --single-dataset tennis_ball"
echo "  python training/train_yolov11_subset.py --single-dataset tennis_court"
echo ""
echo "  # Avec options personnalisées:"
echo "  python training/train_yolov11_subset.py --model yolo11s.pt --epochs 100 --batch-size 16"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Demander à l'utilisateur s'il veut lancer maintenant
read -p "Voulez-vous lancer l'entraînement maintenant ? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Lancement de l'entraînement..."
    python training/train_yolov11_subset.py
fi