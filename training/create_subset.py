"""
Script pour créer des sous-ensembles de datasets avec un nombre limité d'images
Utile pour entraînement rapide ou tests
"""

import os
import shutil
import random
from pathlib import Path
import argparse
import logging
from tqdm import tqdm

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetSubsetCreator:
    """Crée des sous-ensembles de datasets avec un nombre limité d'images"""

    def __init__(self, base_dir='training/dataset'):
        self.base_dir = Path(base_dir)
        self.subset_dir = self.base_dir.parent / 'dataset_subset'

    def create_subset(self, dataset_name, dataset_path, max_images_per_split, output_name=None):
        """
        Crée un sous-ensemble d'un dataset

        Args:
            dataset_name: Nom du dataset
            dataset_path: Chemin vers le dataset complet
            max_images_per_split: Dict avec le nombre max d'images par split
                                 Ex: {'train': 240, 'valid': 40, 'test': 20}
            output_name: Nom du dataset de sortie (défaut: {dataset_name}_subset)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Création du sous-ensemble pour: {dataset_name}")
        logger.info(f"{'='*60}")

        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            logger.error(f"Dataset non trouvé: {dataset_path}")
            return False

        # Nom du dataset de sortie
        if output_name is None:
            output_name = f"{dataset_name}_subset"

        output_path = self.subset_dir / output_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Pour chaque split (train, valid, test)
        for split, max_images in max_images_per_split.items():
            logger.info(f"\nTraitement du split: {split} (max: {max_images} images)")

            # Chemins source
            src_images_dir = dataset_path / split / 'images'
            src_labels_dir = dataset_path / split / 'labels'

            if not src_images_dir.exists():
                logger.warning(f"Dossier images non trouvé: {src_images_dir}")
                continue

            # Chemins destination
            dst_images_dir = output_path / split / 'images'
            dst_labels_dir = output_path / split / 'labels'
            dst_images_dir.mkdir(parents=True, exist_ok=True)
            dst_labels_dir.mkdir(parents=True, exist_ok=True)

            # Lister toutes les images
            all_images = list(src_images_dir.glob('*.[jp][pn][g]'))
            total_available = len(all_images)

            # Sélectionner aléatoirement
            num_to_select = min(max_images, total_available)
            selected_images = random.sample(all_images, num_to_select)

            logger.info(f"  Images disponibles: {total_available}")
            logger.info(f"  Images sélectionnées: {num_to_select}")

            # Copier les images et labels sélectionnés
            for img_file in tqdm(selected_images, desc=f"Copie {split}"):
                # Copier l'image
                dst_img = dst_images_dir / img_file.name
                shutil.copy2(img_file, dst_img)

                # Copier le label correspondant si existe
                label_file = src_labels_dir / (img_file.stem + '.txt')
                if label_file.exists():
                    dst_label = dst_labels_dir / label_file.name
                    shutil.copy2(label_file, dst_label)

        # Copier le fichier dataset.yaml si existe
        src_yaml = dataset_path / 'dataset.yaml'
        if src_yaml.exists():
            dst_yaml = output_path / 'dataset.yaml'
            shutil.copy2(src_yaml, dst_yaml)

            # Mettre à jour le path dans le YAML
            import yaml
            with open(dst_yaml, 'r') as f:
                yaml_data = yaml.safe_load(f)

            yaml_data['path'] = str(output_path.absolute())

            with open(dst_yaml, 'w') as f:
                yaml.dump(yaml_data, f, default_flow_style=False)

            logger.info(f"\nFichier YAML créé: {dst_yaml}")

        logger.info(f"\n✓ Sous-ensemble créé: {output_path}")
        return output_path

    def create_all_subsets(self, limits):
        """
        Crée des sous-ensembles pour tous les datasets

        Args:
            limits: Dict avec les limites par dataset
                   Ex: {
                       'raquette': 300,
                       'tennis_ball': 200,
                       'tennis_court_id_yolo': 100
                   }
        """
        logger.info("\n" + "="*60)
        logger.info("CRÉATION DES SOUS-ENSEMBLES")
        logger.info("="*60)

        created_subsets = {}

        for dataset_name, total_images in limits.items():
            # Calculer la répartition (80% train, 10% valid, 10% test)
            train_images = int(total_images * 0.8)
            valid_images = int(total_images * 0.1)
            test_images = total_images - train_images - valid_images

            split_limits = {
                'train': train_images,
                'valid': valid_images,
                'test': test_images
            }

            logger.info(f"\n{dataset_name}:")
            logger.info(f"  Total: {total_images} images")
            logger.info(f"  Train: {train_images}, Valid: {valid_images}, Test: {test_images}")

            # Chemin du dataset source
            dataset_path = self.base_dir / dataset_name

            # Créer le sous-ensemble
            subset_path = self.create_subset(
                dataset_name,
                dataset_path,
                split_limits
            )

            if subset_path:
                created_subsets[dataset_name] = subset_path

        # Résumé final
        logger.info("\n" + "="*60)
        logger.info("RÉSUMÉ DES SOUS-ENSEMBLES CRÉÉS")
        logger.info("="*60)

        for dataset_name, subset_path in created_subsets.items():
            logger.info(f"\n{dataset_name}:")
            logger.info(f"  Chemin: {subset_path}")

            # Compter les images créées
            for split in ['train', 'valid', 'test']:
                images_dir = subset_path / split / 'images'
                if images_dir.exists():
                    num_images = len(list(images_dir.glob('*.[jp][pn][g]')))
                    logger.info(f"  {split}: {num_images} images")

        logger.info(f"\n✓ Tous les sous-ensembles sont dans: {self.subset_dir}")
        logger.info(f"\nVous pouvez maintenant lancer l'entraînement avec:")
        logger.info(f"  python training/train_yolov11_subset.py")

        return created_subsets


def main():
    parser = argparse.ArgumentParser(description='Créer des sous-ensembles de datasets')
    parser.add_argument('--base-dir', type=str, default='training/dataset',
                       help='Répertoire de base contenant les datasets')
    parser.add_argument('--raquette', type=int, default=300,
                       help='Nombre d\'images pour le dataset raquette')
    parser.add_argument('--tennis-ball', type=int, default=200,
                       help='Nombre d\'images pour le dataset tennis ball')
    parser.add_argument('--tennis-court', type=int, default=100,
                       help='Nombre d\'images pour le dataset tennis court')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed pour la sélection aléatoire')

    args = parser.parse_args()

    # Fixer le seed pour reproductibilité
    random.seed(args.seed)

    # Créer le gestionnaire
    creator = DatasetSubsetCreator(base_dir=args.base_dir)

    # Définir les limites
    limits = {
        'raquette': args.raquette,
        'tennis ball': args.tennis_ball,
        'tennis_court_id_yolo': args.tennis_court
    }

    # Créer les sous-ensembles
    creator.create_all_subsets(limits)


if __name__ == "__main__":
    main()