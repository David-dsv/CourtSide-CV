"""
Script d'entraînement YOLO26 spécialisé pour la détection de balle de tennis.

YOLO26 (Jan 2026) apporte :
- Inférence NMS-free (end-to-end) = latence réduite
- Suppression du module DFL = plus léger, meilleur sur edge
- ProgLoss + STAL = meilleure détection des petits objets (la balle!)
- Optimiseur MuSGD (hybride SGD + Muon) = convergence plus stable

Usage :
  # Entraîner sur le dataset tennis_ball seul
  python training/train_yolo26_ball.py --dataset ball

  # Entraîner sur le dataset combiné (10 classes)
  python training/train_yolo26_ball.py --dataset combined

  # Utiliser un modèle plus gros pour plus de précision
  python training/train_yolo26_ball.py --model yolo26m.pt --imgsz 1280

  # Reprendre un entraînement interrompu
  python training/train_yolo26_ball.py --resume training/runs_yolo26/last_run/weights/last.pt
"""

import os
import yaml
import torch
import argparse
import shutil
import logging
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_device(requested: str = "auto") -> str:
    """Sélectionne le meilleur device disponible."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def create_ball_dataset_yaml(dataset_path: Path, output_yaml: Path) -> Path:
    """
    Crée (ou vérifie) le data.yaml pour le dataset tennis_ball seul.
    Structure attendue :
      dataset_path/
        train/images/  train/labels/
        valid/images/  valid/labels/
        test/images/   test/labels/   (optionnel)
    """
    for split in ("train", "valid"):
        img_dir = dataset_path / split / "images"
        lbl_dir = dataset_path / split / "labels"
        if not img_dir.exists():
            raise FileNotFoundError(f"Dossier manquant : {img_dir}")
        if not lbl_dir.exists():
            raise FileNotFoundError(f"Dossier manquant : {lbl_dir}")

    content = {
        "path": str(dataset_path.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "nc": 1,
        "names": ["tennis_ball"],
    }

    # Ajouter test si disponible
    test_dir = dataset_path / "test" / "images"
    if test_dir.exists():
        content["test"] = "test/images"

    with open(output_yaml, "w") as f:
        yaml.dump(content, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Dataset YAML creé : {output_yaml}")
    return output_yaml


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class YOLO26BallTrainer:
    """
    Entraîneur YOLO26 optimisé pour la détection de balle de tennis.

    Particularités par rapport au script YOLOv11 existant :
    - Résolution plus haute par défaut (1280) pour les petits objets
    - Augmentations ajustées : pas de flip vertical, mosaic fort
    - Box loss boosté (la balle est minuscule, on veut des bbox précises)
    - Patience élevée car la convergence sur petits objets est lente
    - Support du mode end-to-end (NMS-free) de YOLO26
    """

    def __init__(
        self,
        base_model: str = "yolo26n.pt",
        device: str = "auto",
    ):
        self.base_model = base_model
        self.device = detect_device(device)
        self.training_dir = Path(__file__).parent
        self.runs_dir = self.training_dir / "runs_yolo26"
        self.runs_dir.mkdir(exist_ok=True)

        logger.info(f"YOLO26 Trainer initialisé : model={base_model}, device={self.device}")

    # ---- Dataset helpers --------------------------------------------------

    def get_ball_dataset_yaml(self) -> Path:
        """Retourne le YAML pour le dataset tennis_ball seul."""
        dataset_path = self.training_dir / "dataset" / "tennis ball"
        if not dataset_path.exists():
            # Essayer avec underscore
            dataset_path = self.training_dir / "dataset" / "tennis_ball"
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset tennis ball non trouvé dans {self.training_dir / 'dataset'}. "
                "Téléchargez-le depuis Roboflow et placez-le dans training/dataset/tennis ball/"
            )
        yaml_path = self.training_dir / "yolo26_ball_dataset.yaml"
        return create_ball_dataset_yaml(dataset_path, yaml_path)

    def get_combined_dataset_yaml(self) -> Path:
        """Retourne le YAML pour le dataset combiné (10 classes)."""
        yaml_path = self.training_dir / "dataset_combined" / "data.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Dataset combiné non trouvé : {yaml_path}. "
                "Lancez d'abord : python training/combine_datasets.py"
            )
        return yaml_path

    # ---- Training ---------------------------------------------------------

    def train(
        self,
        data_yaml: Path,
        epochs: int = 200,
        imgsz: int = 1280,
        batch_size: int = 8,
        patience: int = 80,
        resume: str = None,
    ):
        """
        Lance l'entraînement YOLO26.

        Args:
            data_yaml:   Chemin vers le data.yaml du dataset
            epochs:      Nombre d'époques max
            imgsz:       Taille des images (1280 recommandé pour petits objets)
            batch_size:  Taille du batch (réduire si OOM)
            patience:    Early stopping patience
            resume:      Chemin vers un checkpoint pour reprendre
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"yolo26_ball_{timestamp}"

        logger.info("=" * 60)
        logger.info("ENTRAÎNEMENT YOLO26 - DÉTECTION BALLE DE TENNIS")
        logger.info("=" * 60)
        logger.info(f"  Modèle de base : {self.base_model}")
        logger.info(f"  Device         : {self.device}")
        logger.info(f"  Dataset        : {data_yaml}")
        logger.info(f"  Époques        : {epochs}")
        logger.info(f"  Image size     : {imgsz}")
        logger.info(f"  Batch size     : {batch_size}")
        logger.info(f"  Patience       : {patience}")
        if resume:
            logger.info(f"  Resume from    : {resume}")
        logger.info("=" * 60)

        # Charger le modèle
        if resume:
            model = YOLO(resume)
            logger.info(f"Reprise de l'entraînement depuis {resume}")
        else:
            model = YOLO(self.base_model)
            logger.info(f"Modèle pré-entraîné chargé : {self.base_model}")

        # ------------------------------------------------------------------
        # Hyperparamètres optimisés pour la détection de petits objets
        # ------------------------------------------------------------------
        train_args = {
            "data": str(data_yaml),
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch_size,
            "device": self.device,
            "project": str(self.runs_dir),
            "name": run_name,
            "patience": patience,
            "save_period": 10,
            "exist_ok": False,
            "pretrained": True,

            # --- Optimiseur ---
            # YOLO26 introduit MuSGD mais AdamW reste un bon choix pour le
            # fine-tuning sur petits datasets. Essayez 'auto' pour laisser
            # YOLO26 choisir MuSGD si disponible.
            "optimizer": "AdamW",
            "lr0": 0.0005,         # LR plus bas pour fine-tuning
            "lrf": 0.01,           # LR final ratio
            "momentum": 0.937,
            "weight_decay": 0.001, # Régularisation un peu plus forte

            # --- Warmup ---
            "warmup_epochs": 5.0,  # Warmup plus long pour stabiliser
            "warmup_momentum": 0.5,
            "warmup_bias_lr": 0.01,

            # --- Pondération des losses ---
            # Box loss élevé : la balle est toute petite, on veut des bbox
            # très précises
            "box": 10.0,           # Défaut 7.5, on booste
            "cls": 0.5,            # Classification (1 classe = peu important)

            # --- Augmentations ---
            # La balle peut apparaître n'importe où et sous n'importe quel
            # éclairage, on augmente fortement
            "hsv_h": 0.02,         # Variation de teinte (balle jaune)
            "hsv_s": 0.8,          # Saturation forte
            "hsv_v": 0.5,          # Luminosité forte (ombre/soleil)
            "degrees": 0.0,        # Pas de rotation (la balle est ronde)
            "translate": 0.2,      # Translation plus forte
            "scale": 0.7,          # Scale plus agressif (balle loin vs proche)
            "shear": 0.0,
            "perspective": 0.001,  # Légère perspective
            "flipud": 0.0,         # Pas de flip vertical (gravité)
            "fliplr": 0.5,         # Flip horizontal OK
            "mosaic": 1.0,         # Mosaic à fond (plus d'exemples par batch)
            "mixup": 0.1,          # Un peu de mixup
            "copy_paste": 0.1,     # Copy-paste : duplique la balle aléatoirement

            # --- Divers ---
            "plots": True,
            "save": True,
            "cache": False,        # True si RAM suffisante
            "verbose": True,
            "amp": True,           # Mixed precision (accélère sur GPU)
            "close_mosaic": 20,    # Désactive mosaic les 20 dernières époques
                                   # pour affiner les prédictions
        }

        try:
            results = model.train(**train_args)

            # Sauvegarder le meilleur modèle
            best_path = self.runs_dir / run_name / "weights" / "best.pt"
            final_path = self.training_dir / "yolo26_ball_best.pt"

            if best_path.exists():
                shutil.copy(best_path, final_path)
                logger.info(f"Meilleur modèle copié vers : {final_path}")

            # Évaluation
            logger.info("\nÉvaluation sur le set de validation...")
            metrics = model.val(data=str(data_yaml))

            logger.info("=" * 60)
            logger.info("RÉSULTATS")
            logger.info("=" * 60)
            logger.info(f"  mAP@50     : {metrics.box.map50:.4f} ({metrics.box.map50*100:.1f}%)")
            logger.info(f"  mAP@50-95  : {metrics.box.map:.4f} ({metrics.box.map*100:.1f}%)")
            logger.info(f"  Precision  : {metrics.box.mp:.4f}")
            logger.info(f"  Recall     : {metrics.box.mr:.4f}")

            return final_path, metrics

        except Exception as e:
            logger.error(f"Erreur pendant l'entraînement : {e}")
            import traceback
            traceback.print_exc()
            return None, None

    # ---- Evaluation -------------------------------------------------------

    def evaluate(self, model_path: str, data_yaml: Path, imgsz: int = 1280):
        """Évalue un modèle entraîné."""
        logger.info(f"Évaluation de {model_path} sur {data_yaml}")
        model = YOLO(model_path)

        metrics = model.val(
            data=str(data_yaml),
            imgsz=imgsz,
            device=self.device,
            verbose=True,
        )

        logger.info("=" * 60)
        logger.info("MÉTRIQUES")
        logger.info("=" * 60)
        logger.info(f"  mAP@50     : {metrics.box.map50:.4f}")
        logger.info(f"  mAP@50-95  : {metrics.box.map:.4f}")
        logger.info(f"  Precision  : {metrics.box.mp:.4f}")
        logger.info(f"  Recall     : {metrics.box.mr:.4f}")

        return metrics

    # ---- Export -----------------------------------------------------------

    def export(self, model_path: str, format: str = "onnx", imgsz: int = 1280):
        """
        Exporte le modèle entraîné pour le déploiement.
        YOLO26 supporte l'export NMS-free natif.
        """
        logger.info(f"Export du modèle {model_path} en {format}")
        model = YOLO(model_path)
        model.export(format=format, imgsz=imgsz)
        logger.info(f"Export terminé : {format}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Entraîner YOLO26 pour la détection de balle de tennis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Entraîner sur le dataset balle seule (nano, rapide)
  python training/train_yolo26_ball.py --dataset ball --model yolo26n.pt

  # Entraîner sur le dataset combiné avec un modèle medium
  python training/train_yolo26_ball.py --dataset combined --model yolo26m.pt --imgsz 1280

  # Reprendre un entraînement
  python training/train_yolo26_ball.py --resume training/runs_yolo26/yolo26_ball_.../weights/last.pt

  # Évaluer un modèle existant
  python training/train_yolo26_ball.py --eval training/yolo26_ball_best.pt

  # Exporter en ONNX
  python training/train_yolo26_ball.py --export training/yolo26_ball_best.pt --format onnx
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo26s.pt",
        choices=["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"],
        help="Modèle YOLO26 de base (défaut: yolo26s.pt - bon compromis vitesse/précision)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ball",
        choices=["ball", "combined"],
        help="Dataset à utiliser : 'ball' (tennis_ball seul) ou 'combined' (10 classes)",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Nombre d'époques")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Taille des images (1280 recommandé pour petits objets, 640 si RAM limitée)",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Taille du batch")
    parser.add_argument("--patience", type=int, default=80, help="Patience early stopping")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device pour l'entraînement",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Chemin vers un checkpoint pour reprendre l'entraînement",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        help="Évaluer un modèle existant (chemin vers .pt)",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Exporter un modèle existant (chemin vers .pt)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "torchscript", "coreml", "tflite", "openvino"],
        help="Format d'export",
    )

    args = parser.parse_args()

    # Créer le trainer
    trainer = YOLO26BallTrainer(base_model=args.model, device=args.device)

    # Récupérer le dataset YAML
    if args.dataset == "ball":
        data_yaml = trainer.get_ball_dataset_yaml()
    else:
        data_yaml = trainer.get_combined_dataset_yaml()

    # Mode export
    if args.export:
        trainer.export(args.export, format=args.format, imgsz=args.imgsz)
        return

    # Mode évaluation
    if args.eval:
        trainer.evaluate(args.eval, data_yaml, imgsz=args.imgsz)
        return

    # Mode entraînement
    model_path, metrics = trainer.train(
        data_yaml=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        patience=args.patience,
        resume=args.resume,
    )

    if model_path:
        logger.info("\n" + "=" * 60)
        logger.info("ENTRAÎNEMENT TERMINÉ")
        logger.info("=" * 60)
        logger.info(f"\nModèle sauvegardé : {model_path}")
        logger.info("\nPour l'utiliser dans le pipeline :")
        logger.info(f"  1. Copier dans weights/ : cp {model_path} weights/yolo26_ball.pt")
        logger.info(f"  2. Modifier config.yaml : yolo_weights: weights/yolo26_ball.pt")
        logger.info(f"\nPour évaluer :")
        logger.info(f"  python training/train_yolo26_ball.py --eval {model_path}")
        logger.info(f"\nPour exporter (ONNX) :")
        logger.info(f"  python training/train_yolo26_ball.py --export {model_path}")
    else:
        logger.error("\nEntraînement échoué !")


if __name__ == "__main__":
    main()
