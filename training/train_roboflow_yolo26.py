"""
Télécharge le dataset Tennis Ball Detection depuis Roboflow (1000 images),
puis lance l'entraînement YOLO26 avec le dashboard live activé.

Usage :
  python training/train_roboflow_yolo26.py

  # Avec options
  python training/train_roboflow_yolo26.py --model yolo26m.pt --epochs 150 --batch-size 4

  # Dashboard seul (après un entraînement)
  python training/train_roboflow_yolo26.py --dashboard-only
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TRAINING_DIR = Path(__file__).parent
PROJECT_ROOT = TRAINING_DIR.parent


# ---------------------------------------------------------------------------
# 1. Téléchargement du dataset Roboflow
# ---------------------------------------------------------------------------

def download_dataset(target_dir: Path) -> Path:
    """
    Télécharge le dataset Tennis Ball Detection v3 depuis Roboflow
    au format yolo26 (YOLOv5/v8/v11/v26 compatible).
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        logger.info("Installation de roboflow...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])
        from roboflow import Roboflow

    logger.info("=" * 60)
    logger.info("TÉLÉCHARGEMENT DU DATASET ROBOFLOW")
    logger.info("=" * 60)

    rf = Roboflow(api_key="9Dho4KSn6P7slwKtjMN4")
    project = rf.workspace("nathan-4tlqa").project("tennis-ball-detection-vksde")
    version = project.version(3)

    # Télécharger au format yolo11v2 (compatible YOLO26)
    dataset = version.download("yolov11", location=str(target_dir))

    logger.info(f"Dataset téléchargé dans : {target_dir}")

    # Vérifier la structure
    for split in ("train", "valid"):
        img_dir = target_dir / split / "images"
        if img_dir.exists():
            n_images = len(list(img_dir.glob("*")))
            logger.info(f"  {split}: {n_images} images")
        else:
            logger.warning(f"  {split}/images non trouvé!")

    test_dir = target_dir / "test" / "images"
    if test_dir.exists():
        n_test = len(list(test_dir.glob("*")))
        logger.info(f"  test: {n_test} images")

    return target_dir


def limit_dataset(dataset_dir: Path, max_images: int = 1000):
    """
    Limite le dataset à max_images au total (réparti proportionnellement
    entre train/valid/test). Si le dataset a déjà <= max_images, ne fait rien.
    """
    import random
    import shutil

    splits = ["train", "valid", "test"]
    counts = {}
    total = 0

    for split in splits:
        img_dir = dataset_dir / split / "images"
        if img_dir.exists():
            images = list(img_dir.glob("*"))
            counts[split] = len(images)
            total += len(images)
        else:
            counts[split] = 0

    if total <= max_images:
        logger.info(f"Dataset déjà <= {max_images} images ({total} total). Pas de réduction.")
        return

    logger.info(f"Réduction du dataset de {total} à {max_images} images...")

    # Calculer les proportions
    for split in splits:
        if counts[split] == 0:
            continue

        target_count = int(max_images * (counts[split] / total))
        target_count = max(target_count, 1)  # Au moins 1 image

        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"

        images = sorted(img_dir.glob("*"))
        if len(images) <= target_count:
            continue

        # Sélectionner aléatoirement les images à garder
        random.seed(42)
        to_remove = random.sample(images, len(images) - target_count)

        for img_path in to_remove:
            # Supprimer l'image
            img_path.unlink()
            # Supprimer le label correspondant
            label_path = lbl_dir / (img_path.stem + ".txt")
            if label_path.exists():
                label_path.unlink()

        remaining = len(list(img_dir.glob("*")))
        logger.info(f"  {split}: {counts[split]} -> {remaining} images")


# ---------------------------------------------------------------------------
# 2. Vérification du dashboard
# ---------------------------------------------------------------------------

def check_dashboard():
    """Vérifie que le dashboard est disponible et fonctionnel."""
    dashboard_path = TRAINING_DIR / "dashboard.py"

    if not dashboard_path.exists():
        logger.warning("dashboard.py non trouvé!")
        return False

    # Vérifier les dépendances
    missing = []
    for pkg in ["matplotlib", "pandas", "numpy"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        logger.warning(f"Dépendances dashboard manquantes : {missing}")
        logger.info("Installation...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install"] + missing
        )

    logger.info("Dashboard activé et fonctionnel")
    logger.info(f"  Script : {dashboard_path}")
    logger.info("  Modes disponibles :")
    logger.info("    --live    : suivi en temps réel pendant l'entraînement")
    logger.info("    --static  : affichage des résultats finaux")
    logger.info("    --compare : comparaison de plusieurs runs")
    return True


# ---------------------------------------------------------------------------
# 3. Entraînement
# ---------------------------------------------------------------------------

def detect_device(requested: str = "auto") -> str:
    """Sélectionne le meilleur device disponible."""
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(dataset_dir: Path, args):
    """Lance l'entraînement YOLO avec les hyperparamètres optimisés pour la balle."""
    from ultralytics import YOLO
    from datetime import datetime
    import shutil

    # Utiliser le data.yaml du dataset Roboflow
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml non trouvé dans {dataset_dir}")

    device = detect_device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"yolo_ball_{timestamp}"
    runs_dir = TRAINING_DIR / "runs_yolo26"
    runs_dir.mkdir(exist_ok=True)

    logger.info("")
    logger.info("Pour suivre l'entraînement en temps réel, ouvrez un autre terminal :")
    logger.info(f"  python training/dashboard.py --live")
    logger.info("")

    # Charger le modèle
    if args.resume:
        model = YOLO(args.resume)
        logger.info(f"Reprise depuis {args.resume}")
    else:
        model = YOLO(args.model)
        logger.info(f"Modèle chargé : {args.model}")

    logger.info(f"Device : {device}")
    logger.info(f"Dataset : {yaml_path}")

    # Hyperparamètres optimisés pour petits objets (balle de tennis)
    train_args = {
        "data": str(yaml_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch_size,
        "device": device,
        "project": str(runs_dir),
        "name": run_name,
        "patience": args.patience,
        "save_period": 10,
        "exist_ok": False,
        "pretrained": True,
        "optimizer": "AdamW",
        "lr0": 0.0005,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.001,
        "warmup_epochs": 5.0,
        "warmup_momentum": 0.5,
        "warmup_bias_lr": 0.01,
        "box": 10.0,
        "cls": 0.5,
        "hsv_h": 0.02,
        "hsv_s": 0.8,
        "hsv_v": 0.5,
        "degrees": 0.0,
        "translate": 0.2,
        "scale": 0.7,
        "shear": 0.0,
        "perspective": 0.001,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.1,
        "plots": True,
        "save": True,
        "cache": False,
        "verbose": True,
        "amp": True,
        "close_mosaic": 10,
    }

    try:
        results = model.train(**train_args)

        best_path = runs_dir / run_name / "weights" / "best.pt"
        final_path = TRAINING_DIR / "yolo_ball_best.pt"

        if best_path.exists():
            shutil.copy(best_path, final_path)
            logger.info(f"Meilleur modèle copié vers : {final_path}")

        logger.info("\nÉvaluation sur le set de validation...")
        metrics = model.val(data=str(yaml_path))

        logger.info(f"  mAP@50     : {metrics.box.map50:.4f}")
        logger.info(f"  mAP@50-95  : {metrics.box.map:.4f}")
        logger.info(f"  Precision  : {metrics.box.mp:.4f}")
        logger.info(f"  Recall     : {metrics.box.mr:.4f}")

        return final_path, metrics

    except Exception as e:
        logger.error(f"Erreur pendant l'entraînement : {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download Roboflow dataset + Train YOLO26 tennis ball detector",
    )
    parser.add_argument("--model", type=str, default="yolo26s.pt",
                        help="Modèle YOLO26 de base (default: yolo26s.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Nombre d'époques")
    parser.add_argument("--imgsz", type=int, default=640, help="Taille des images")
    parser.add_argument("--batch-size", type=int, default=16, help="Taille du batch")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--resume", type=str, default=None,
                        help="Reprendre depuis un checkpoint")
    parser.add_argument("--max-images", type=int, default=1000,
                        help="Nombre max d'images (default: 1000)")
    parser.add_argument("--dashboard-only", action="store_true",
                        help="Lancer seulement le dashboard (pas de download/train)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Ne pas re-télécharger si le dataset existe déjà")

    args = parser.parse_args()

    # -- Dashboard only mode ------------------------------------------------
    if args.dashboard_only:
        check_dashboard()
        subprocess.run([sys.executable, str(TRAINING_DIR / "dashboard.py"), "--live"])
        return

    # -- Step 1 : Vérification du dashboard ---------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1/3 : VÉRIFICATION DU DASHBOARD")
    logger.info("=" * 60)
    dashboard_ok = check_dashboard()

    # -- Step 2 : Téléchargement du dataset ---------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2/3 : TÉLÉCHARGEMENT DU DATASET ROBOFLOW")
    logger.info("=" * 60)

    dataset_dir = TRAINING_DIR / "dataset" / "tennis_ball_roboflow"

    if args.skip_download and dataset_dir.exists():
        logger.info(f"Dataset existant trouvé : {dataset_dir} (skip download)")
    else:
        download_dataset(dataset_dir)

    # Limiter à max_images
    limit_dataset(dataset_dir, max_images=args.max_images)

    # -- Step 3 : Entraînement YOLO26 --------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3/3 : ENTRAÎNEMENT YOLO26")
    logger.info("=" * 60)

    model_path, metrics = train(dataset_dir, args)

    # -- Résumé final -------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE TERMINÉ")
    logger.info("=" * 60)

    if model_path:
        logger.info(f"  Modèle sauvegardé : {model_path}")
        if metrics:
            logger.info(f"  mAP@50     : {metrics.box.map50:.4f}")
            logger.info(f"  mAP@50-95  : {metrics.box.map:.4f}")

    if dashboard_ok:
        logger.info("")
        logger.info("Pour visualiser les résultats :")
        logger.info(f"  python training/dashboard.py --static")
        logger.info(f"  python training/dashboard.py --save results.png")


if __name__ == "__main__":
    main()