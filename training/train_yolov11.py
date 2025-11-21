"""
Script d'entraînement YOLOv11 pour fine-tuning avec 3 datasets:
- Raquette (format YOLO)
- Tennis ball (format YOLO)
- Tennis court ID (format COCO - nécessite conversion)
"""

import os
import yaml
import torch
import argparse
from pathlib import Path
from ultralytics import YOLO
import logging
from datetime import datetime
import shutil

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YOLOv11Trainer:
    """Gestionnaire pour l'entraînement multi-dataset de YOLOv11"""

    def __init__(self, base_model='yolo11n.pt', device='auto'):
        """
        Initialise le trainer YOLOv11

        Args:
            base_model: Modèle de base YOLOv11 (n/s/m/l/x)
            device: Device pour l'entraînement ('cpu', 'cuda', 'mps', 'auto')
        """
        self.base_model = base_model
        self.device = self._setup_device(device)
        self.project_root = Path(__file__).parent.parent
        self.training_dir = Path(__file__).parent
        self.datasets_dir = self.training_dir / 'dataset'
        self.runs_dir = self.training_dir / 'runs'
        self.runs_dir.mkdir(exist_ok=True)

    def _setup_device(self, device):
        """Configure le device pour l'entraînement"""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return device

    def create_dataset_yaml(self, dataset_name, dataset_path, num_classes, class_names):
        """
        Crée un fichier YAML de configuration pour un dataset

        Args:
            dataset_name: Nom du dataset
            dataset_path: Chemin vers le dataset
            num_classes: Nombre de classes
            class_names: Liste des noms de classes
        """
        yaml_content = {
            'path': str(dataset_path),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': num_classes,
            'names': class_names
        }

        yaml_path = self.training_dir / f'{dataset_name}_dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        logger.info(f"Fichier YAML créé pour {dataset_name}: {yaml_path}")
        return yaml_path

    def train_single_model(self, dataset_name, dataset_yaml, epochs=100, imgsz=640,
                          batch_size=16, patience=50, save_period=10):
        """
        Entraîne un modèle sur un dataset unique

        Args:
            dataset_name: Nom du dataset
            dataset_yaml: Chemin vers le fichier YAML du dataset
            epochs: Nombre d'époques
            imgsz: Taille des images
            batch_size: Taille du batch
            patience: Patience pour l'early stopping
            save_period: Fréquence de sauvegarde du modèle
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"Début de l'entraînement pour: {dataset_name}")
        logger.info(f"{'='*50}\n")

        # Créer un nom unique pour cette session d'entraînement
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{dataset_name}_{timestamp}"

        # Charger le modèle
        model = YOLO(self.base_model)

        # Configuration d'entraînement
        train_args = {
            'data': str(dataset_yaml),
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'device': self.device,
            'project': str(self.runs_dir),
            'name': run_name,
            'patience': patience,
            'save_period': save_period,
            'exist_ok': False,
            'pretrained': True,
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'plots': True,
            'save': True,
            'cache': False,
            'verbose': True
        }

        try:
            # Lancer l'entraînement
            results = model.train(**train_args)

            # Sauvegarder le meilleur modèle
            best_model_path = self.runs_dir / run_name / 'weights' / 'best.pt'
            final_model_path = self.training_dir / f'{dataset_name}_best.pt'

            if best_model_path.exists():
                shutil.copy(best_model_path, final_model_path)
                logger.info(f"Modèle sauvegardé: {final_model_path}")

            # Évaluation sur le dataset de test
            logger.info(f"\nÉvaluation du modèle {dataset_name}...")
            metrics = model.val(data=str(dataset_yaml))

            # Afficher les métriques
            logger.info(f"\nMétriques pour {dataset_name}:")
            logger.info(f"mAP50: {metrics.box.map50:.4f}")
            logger.info(f"mAP50-95: {metrics.box.map:.4f}")

            return final_model_path, metrics

        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement de {dataset_name}: {e}")
            return None, None

    def prepare_datasets(self):
        """Prépare les configurations pour les trois datasets"""
        datasets_config = []

        # Configuration pour le dataset raquette
        raquette_path = self.datasets_dir / 'raquette'
        if raquette_path.exists():
            yaml_path = self.create_dataset_yaml(
                'raquette',
                raquette_path,
                num_classes=1,
                class_names=['racket']
            )
            datasets_config.append({
                'name': 'raquette',
                'yaml': yaml_path,
                'type': 'yolo'
            })
        else:
            logger.warning(f"Dataset raquette non trouvé à: {raquette_path}")

        # Configuration pour le dataset tennis ball
        tennis_ball_path = self.datasets_dir / 'tennis ball'
        if tennis_ball_path.exists():
            yaml_path = self.create_dataset_yaml(
                'tennis_ball',
                tennis_ball_path,
                num_classes=1,
                class_names=['tennis_ball']
            )
            datasets_config.append({
                'name': 'tennis_ball',
                'yaml': yaml_path,
                'type': 'yolo'
            })
        else:
            logger.warning(f"Dataset tennis ball non trouvé à: {tennis_ball_path}")

        # Configuration pour le dataset tennis court (nécessite conversion COCO -> YOLO)
        tennis_court_path = self.datasets_dir / 'tennis_court_id'
        if tennis_court_path.exists():
            logger.info("Dataset tennis court détecté (format COCO)")
            logger.info("Conversion COCO vers YOLO nécessaire - voir convert_coco_to_yolo.py")
            # Après conversion, ajouter la configuration
            tennis_court_yolo_path = self.datasets_dir / 'tennis_court_id_yolo'
            if tennis_court_yolo_path.exists():
                yaml_path = self.create_dataset_yaml(
                    'tennis_court',
                    tennis_court_yolo_path,
                    num_classes=1,  # À ajuster selon vos annotations
                    class_names=['tennis_court']
                )
                datasets_config.append({
                    'name': 'tennis_court',
                    'yaml': yaml_path,
                    'type': 'yolo'
                })
        else:
            logger.warning(f"Dataset tennis court non trouvé à: {tennis_court_path}")

        return datasets_config

    def train_all_models(self, epochs=100, batch_size=16):
        """Entraîne un modèle pour chaque dataset"""
        logger.info("\n" + "="*60)
        logger.info("DÉBUT DE L'ENTRAÎNEMENT MULTI-DATASET YOLOV11")
        logger.info("="*60)
        logger.info(f"Device: {self.device}")
        logger.info(f"Modèle de base: {self.base_model}")
        logger.info(f"Époques: {epochs}")
        logger.info(f"Batch size: {batch_size}")

        # Préparer les datasets
        datasets = self.prepare_datasets()

        if not datasets:
            logger.error("Aucun dataset trouvé!")
            return

        results = {}

        # Entraîner un modèle pour chaque dataset
        for dataset in datasets:
            model_path, metrics = self.train_single_model(
                dataset['name'],
                dataset['yaml'],
                epochs=epochs,
                batch_size=batch_size
            )

            results[dataset['name']] = {
                'model_path': model_path,
                'metrics': metrics
            }

        # Résumé final
        logger.info("\n" + "="*60)
        logger.info("RÉSUMÉ DE L'ENTRAÎNEMENT")
        logger.info("="*60)

        for dataset_name, result in results.items():
            if result['model_path']:
                logger.info(f"\n{dataset_name}:")
                logger.info(f"  - Modèle: {result['model_path']}")
                if result['metrics']:
                    logger.info(f"  - mAP50: {result['metrics'].box.map50:.4f}")
                    logger.info(f"  - mAP50-95: {result['metrics'].box.map:.4f}")
            else:
                logger.info(f"\n{dataset_name}: Échec de l'entraînement")

        return results

    def combine_models(self, model_paths, output_name='combined_model.pt'):
        """
        Combine plusieurs modèles YOLOv11 en un seul (optionnel)
        Note: Cette fonction est expérimentale
        """
        logger.info("\nCombination des modèles (expérimental)...")
        logger.warning("La combination de modèles avec différentes classes nécessite une architecture personnalisée")
        # Implémentation à développer selon vos besoins spécifiques
        pass


def main():
    parser = argparse.ArgumentParser(description='Entraînement YOLOv11 multi-dataset')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                       choices=['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt'],
                       help='Modèle YOLOv11 de base')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Nombre d\'époques')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Taille du batch')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'mps', 'auto'],
                       help='Device pour l\'entraînement')
    parser.add_argument('--single-dataset', type=str, default=None,
                       choices=['raquette', 'tennis_ball', 'tennis_court'],
                       help='Entraîner sur un seul dataset')

    args = parser.parse_args()

    # Créer le trainer
    trainer = YOLOv11Trainer(base_model=args.model, device=args.device)

    if args.single_dataset:
        # Entraîner sur un seul dataset
        datasets = trainer.prepare_datasets()
        dataset = next((d for d in datasets if d['name'] == args.single_dataset), None)

        if dataset:
            trainer.train_single_model(
                dataset['name'],
                dataset['yaml'],
                epochs=args.epochs,
                batch_size=args.batch_size
            )
        else:
            logger.error(f"Dataset {args.single_dataset} non trouvé!")
    else:
        # Entraîner sur tous les datasets
        trainer.train_all_models(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()