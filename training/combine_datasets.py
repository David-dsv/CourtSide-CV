"""
Script pour combiner 3 datasets YOLO en un seul dataset multi-classe
- Raquette (1 classe: racket)
- Tennis Ball (1 classe: tennis_ball)
- Tennis Court (8 classes: zones du court)
= Dataset combiné (10 classes au total)
"""

import os
import shutil
import yaml
from pathlib import Path
import random
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetCombiner:
    """Combine plusieurs datasets YOLO en un seul"""

    def __init__(self, output_dir='training/dataset_combined'):
        self.output_dir = Path(output_dir)
        self.datasets = []
        self.class_mapping = {}
        self.combined_classes = []

    def add_dataset(self, name, path, classes, max_images=None):
        """
        Ajoute un dataset à combiner

        Args:
            name: Nom du dataset
            path: Chemin vers le dataset
            classes: Liste des noms de classes
            max_images: Nombre max d'images à prendre (None = toutes)
        """
        dataset_path = Path(path)
        if not dataset_path.exists():
            logger.warning(f"Dataset non trouvé: {dataset_path}")
            return False

        self.datasets.append({
            'name': name,
            'path': dataset_path,
            'classes': classes,
            'max_images': max_images
        })

        logger.info(f"Dataset ajouté: {name} ({len(classes)} classes)")
        return True

    def build_class_mapping(self):
        """
        Construit le mapping des classes
        Chaque dataset a ses propres IDs de classe (0, 1, 2...)
        On doit les remapper vers les nouveaux IDs globaux
        """
        global_class_id = 0

        for dataset in self.datasets:
            dataset['class_offset'] = global_class_id
            dataset['class_mapping'] = {}

            for local_id, class_name in enumerate(dataset['classes']):
                # Ajouter à la liste globale si pas déjà présent
                if class_name not in self.combined_classes:
                    self.combined_classes.append(class_name)
                    new_id = global_class_id
                    global_class_id += 1
                else:
                    # Classe déjà existante, utiliser l'ID existant
                    new_id = self.combined_classes.index(class_name)

                dataset['class_mapping'][local_id] = new_id

        logger.info(f"\nMapping des classes créé:")
        for dataset in self.datasets:
            logger.info(f"  {dataset['name']}:")
            for local_id, global_id in dataset['class_mapping'].items():
                class_name = dataset['classes'][local_id]
                logger.info(f"    {local_id} ({class_name}) → {global_id}")

        logger.info(f"\nClasses combinées ({len(self.combined_classes)}):")
        for i, name in enumerate(self.combined_classes):
            logger.info(f"  {i}: {name}")

    def remap_label_file(self, label_path, class_mapping):
        """
        Remape les IDs de classe dans un fichier label

        Args:
            label_path: Chemin vers le fichier .txt
            class_mapping: Dict {old_id: new_id}

        Returns:
            Liste des lignes remappées
        """
        if not label_path.exists():
            return []

        new_lines = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    old_class_id = int(parts[0])
                    if old_class_id in class_mapping:
                        new_class_id = class_mapping[old_class_id]
                        new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
                        new_lines.append(new_line)

        return new_lines

    def copy_split(self, split='train'):
        """
        Copie un split (train/valid/test) de tous les datasets

        Args:
            split: 'train', 'valid' ou 'test'
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Traitement du split: {split}")
        logger.info(f"{'='*60}")

        # Créer les dossiers de sortie
        output_images = self.output_dir / split / 'images'
        output_labels = self.output_dir / split / 'labels'
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)

        total_images = 0

        for dataset in self.datasets:
            logger.info(f"\nDataset: {dataset['name']}")

            # Chemins source
            src_images = dataset['path'] / split / 'images'
            src_labels = dataset['path'] / split / 'labels'

            if not src_images.exists():
                logger.warning(f"  Dossier images non trouvé: {src_images}")
                continue

            # Lister les images
            image_files = list(src_images.glob('*.[jp][pn][g]'))

            # Limiter si demandé
            if dataset['max_images'] and split == 'train':
                # Calculer le ratio pour ce split
                total_in_dataset = len(list(dataset['path'].glob('**/*.[jp][pn][g]')))
                ratio = dataset['max_images'] / max(total_in_dataset, 1)
                max_for_split = int(len(image_files) * ratio)

                if max_for_split < len(image_files):
                    image_files = random.sample(image_files, max_for_split)
                    logger.info(f"  Limité à {max_for_split} images")

            logger.info(f"  Images à copier: {len(image_files)}")

            # Copier les images et remapper les labels
            for img_file in tqdm(image_files, desc=f"  Copie {dataset['name']}"):
                # Nouveau nom avec préfixe pour éviter les collisions
                new_name = f"{dataset['name']}_{img_file.name}"

                # Copier l'image
                dst_img = output_images / new_name
                shutil.copy2(img_file, dst_img)

                # Remapper et copier le label
                label_file = src_labels / (img_file.stem + '.txt')
                new_lines = self.remap_label_file(label_file, dataset['class_mapping'])

                if new_lines:
                    dst_label = output_labels / (Path(new_name).stem + '.txt')
                    with open(dst_label, 'w') as f:
                        f.writelines(new_lines)

                total_images += 1

        logger.info(f"\n✓ {split}: {total_images} images copiées")
        return total_images

    def create_yaml(self):
        """Crée le fichier data.yaml pour le dataset combiné"""
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': len(self.combined_classes),
            'names': self.combined_classes
        }

        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

        logger.info(f"\n✓ Fichier YAML créé: {yaml_path}")
        return yaml_path

    def combine(self):
        """Effectue la combinaison complète"""
        logger.info("\n" + "="*60)
        logger.info("COMBINAISON DES DATASETS")
        logger.info("="*60)

        if not self.datasets:
            logger.error("Aucun dataset ajouté!")
            return False

        # Construire le mapping des classes
        self.build_class_mapping()

        # Copier chaque split
        stats = {}
        for split in ['train', 'valid', 'test']:
            stats[split] = self.copy_split(split)

        # Créer le fichier YAML
        yaml_path = self.create_yaml()

        # Résumé final
        logger.info("\n" + "="*60)
        logger.info("RÉSUMÉ")
        logger.info("="*60)
        logger.info(f"Datasets sources: {len(self.datasets)}")
        for ds in self.datasets:
            logger.info(f"  - {ds['name']}: {len(ds['classes'])} classes")
        logger.info(f"\nDataset combiné:")
        logger.info(f"  Classes totales: {len(self.combined_classes)}")
        logger.info(f"  Train: {stats.get('train', 0)} images")
        logger.info(f"  Valid: {stats.get('valid', 0)} images")
        logger.info(f"  Test: {stats.get('test', 0)} images")
        logger.info(f"  Total: {sum(stats.values())} images")
        logger.info(f"\nDataset prêt: {self.output_dir}")
        logger.info(f"Config: {yaml_path}")

        return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Combiner plusieurs datasets YOLO')
    parser.add_argument('--output', type=str, default='training/dataset_combined',
                       help='Dossier de sortie')
    parser.add_argument('--racket-max', type=int, default=500,
                       help='Nombre max d\'images pour raquette')
    parser.add_argument('--ball-max', type=int, default=500,
                       help='Nombre max d\'images pour tennis ball')
    parser.add_argument('--court-max', type=int, default=200,
                       help='Nombre max d\'images pour tennis court')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed pour la sélection aléatoire')

    args = parser.parse_args()

    # Fixer le seed
    random.seed(args.seed)

    # Créer le combineur
    combiner = DatasetCombiner(output_dir=args.output)

    # Ajouter les datasets
    logger.info("Ajout des datasets...")

    # 1. Raquette (1 classe)
    combiner.add_dataset(
        name='racket',
        path='training/dataset/raquette',
        classes=['racket'],
        max_images=args.racket_max
    )

    # 2. Tennis Ball (1 classe)
    combiner.add_dataset(
        name='ball',
        path='training/dataset/tennis ball',
        classes=['tennis_ball'],
        max_images=args.ball_max
    )

    # 3. Tennis Court (8 classes)
    combiner.add_dataset(
        name='court',
        path='training/dataset/tennis court id yolov11',
        classes=[
            'bottom-dead-zone',
            'court',
            'left-doubles-alley',
            'left-service-box',
            'net',
            'right-doubles-alley',
            'right-service-box',
            'top-dead-zone'
        ],
        max_images=args.court_max
    )

    # Combiner
    success = combiner.combine()

    if success:
        logger.info("\n" + "="*60)
        logger.info("✅ COMBINAISON RÉUSSIE!")
        logger.info("="*60)
        logger.info("\nVous pouvez maintenant entraîner avec:")
        logger.info(f"  python training/train_combined.py")
    else:
        logger.error("\n❌ Échec de la combinaison")


if __name__ == "__main__":
    main()