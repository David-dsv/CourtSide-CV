"""
Script d'entraînement YOLOv11 pour le dataset combiné multi-classe
Détecte : raquettes + balles + zones du court (10 classes au total)
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CombinedModelTrainer:
    """Entraîneur pour le modèle combiné multi-classe"""

    def __init__(self, base_model='yolo11n.pt', device='auto'):
        """
        Initialise le trainer

        Args:
            base_model: Modèle de base YOLOv11
            device: Device pour l'entraînement
        """
        self.base_model = base_model
        self.device = self._setup_device(device)
        self.training_dir = Path(__file__).parent
        self.dataset_dir = self.training_dir / 'dataset_combined'
        self.runs_dir = self.training_dir / 'runs_combined'
        self.runs_dir.mkdir(exist_ok=True)

    def _setup_device(self, device):
        """Configure le device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return device

    def check_dataset(self):
        """Vérifie que le dataset combiné existe"""
        yaml_path = self.dataset_dir / 'data.yaml'

        if not yaml_path.exists():
            logger.error(f"Dataset combiné non trouvé: {yaml_path}")
            logger.info("\nVeuillez d'abord créer le dataset combiné avec:")
            logger.info("  python training/combine_datasets.py")
            return None

        # Lire le YAML
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        logger.info(f"\nDataset combiné trouvé:")
        logger.info(f"  Chemin: {self.dataset_dir}")
        logger.info(f"  Classes: {data['nc']}")
        logger.info(f"  Noms: {data['names']}")

        return yaml_path

    def train(self, epochs=150, batch_size=16, imgsz=640, patience=50):
        """
        Entraîne le modèle combiné

        Args:
            epochs: Nombre d'époques
            batch_size: Taille du batch
            imgsz: Taille des images
            patience: Patience pour early stopping
        """
        # Vérifier le dataset
        yaml_path = self.check_dataset()
        if not yaml_path:
            return None

        logger.info("\n" + "="*60)
        logger.info("ENTRAÎNEMENT MODÈLE COMBINÉ YOLOV11")
        logger.info("="*60)
        logger.info(f"Modèle: {self.base_model}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Époques: {epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Image size: {imgsz}")

        # Nom du run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"combined_tennis_{timestamp}"

        # Charger le modèle
        model = YOLO(self.base_model)

        # Configuration d'entraînement
        train_args = {
            'data': str(yaml_path),
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'device': self.device,
            'project': str(self.runs_dir),
            'name': run_name,
            'patience': patience,
            'save_period': 10,
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
            logger.info(f"\n🚀 Début de l'entraînement...\n")
            results = model.train(**train_args)

            # Sauvegarder le meilleur modèle
            best_model_path = self.runs_dir / run_name / 'weights' / 'best.pt'
            final_model_path = self.training_dir / 'tennis_combined_best.pt'

            if best_model_path.exists():
                shutil.copy(best_model_path, final_model_path)
                logger.info(f"\n✓ Modèle sauvegardé: {final_model_path}")

            # Évaluation
            logger.info(f"\n📊 Évaluation du modèle...")
            metrics = model.val(data=str(yaml_path))

            # Afficher les métriques
            logger.info(f"\n" + "="*60)
            logger.info("MÉTRIQUES FINALES")
            logger.info("="*60)
            logger.info(f"mAP@50: {metrics.box.map50:.4f} ({metrics.box.map50*100:.2f}%)")
            logger.info(f"mAP@50-95: {metrics.box.map:.4f} ({metrics.box.map*100:.2f}%)")

            # Métriques par classe
            if hasattr(metrics.box, 'maps'):
                logger.info(f"\nMétriques par classe:")
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    class_names = data['names']

                for i, (map50, map50_95) in enumerate(zip(metrics.box.maps50, metrics.box.maps)):
                    if i < len(class_names):
                        logger.info(f"  {class_names[i]:20s} - mAP@50: {map50:.3f}, mAP@50-95: {map50_95:.3f}")

            return final_model_path, metrics

        except Exception as e:
            logger.error(f"\n❌ Erreur lors de l'entraînement: {e}")
            import traceback
            traceback.print_exc()
            return None, None


def main():
    parser = argparse.ArgumentParser(description='Entraîner le modèle YOLOv11 combiné')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                       choices=['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt'],
                       help='Modèle YOLOv11 de base')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Nombre d\'époques')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Taille du batch')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Taille des images')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'mps', 'auto'],
                       help='Device pour l\'entraînement')
    parser.add_argument('--patience', type=int, default=50,
                       help='Patience pour early stopping')

    args = parser.parse_args()

    # Créer le trainer
    trainer = CombinedModelTrainer(base_model=args.model, device=args.device)

    # Entraîner
    model_path, metrics = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        patience=args.patience
    )

    if model_path:
        logger.info("\n" + "="*60)
        logger.info("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        logger.info("="*60)
        logger.info(f"\nModèle final: {model_path}")
        logger.info(f"\nVous pouvez maintenant utiliser le modèle:")
        logger.info(f"  from ultralytics import YOLO")
        logger.info(f"  model = YOLO('{model_path}')")
        logger.info(f"  results = model.predict('image.jpg')")
    else:
        logger.error("\n❌ Entraînement échoué")


if __name__ == "__main__":
    main()