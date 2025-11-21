"""
Script de préparation des données pour l'entraînement YOLOv11
- Vérifie l'intégrité des datasets
- Convertit le dataset COCO en YOLO si nécessaire
- Génère des statistiques sur les datasets
- Crée les fichiers de configuration YAML
"""

import os
import sys
import yaml
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetPreparer:
    """Classe pour préparer et valider les datasets"""

    def __init__(self, base_dir='training/dataset'):
        """
        Initialise le préparateur de données

        Args:
            base_dir: Répertoire de base contenant les datasets
        """
        self.base_dir = Path(base_dir)
        self.stats = defaultdict(dict)

    def check_dataset_structure(self, dataset_path, dataset_name):
        """
        Vérifie la structure d'un dataset YOLO

        Args:
            dataset_path: Chemin vers le dataset
            dataset_name: Nom du dataset

        Returns:
            Dictionnaire avec les statistiques du dataset
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"Vérification du dataset: {dataset_name}")
        logger.info(f"{'='*50}")

        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            logger.error(f"Dataset non trouvé: {dataset_path}")
            return None

        stats = {
            'name': dataset_name,
            'path': str(dataset_path),
            'valid': True,
            'splits': {}
        }

        required_splits = ['train', 'valid', 'test']
        for split in required_splits:
            split_stats = {
                'images': 0,
                'labels': 0,
                'missing_labels': [],
                'corrupted_images': [],
                'class_distribution': defaultdict(int),
                'bbox_stats': {
                    'count': 0,
                    'avg_per_image': 0,
                    'min_area': float('inf'),
                    'max_area': 0,
                    'avg_area': 0
                }
            }

            # Chemins des dossiers
            images_dir = dataset_path / split / 'images'
            labels_dir = dataset_path / split / 'labels'

            if not images_dir.exists():
                logger.warning(f"Dossier images manquant: {images_dir}")
                stats['valid'] = False
                continue

            if not labels_dir.exists():
                logger.warning(f"Dossier labels manquant: {labels_dir}")
                stats['valid'] = False
                continue

            # Compter les images
            image_files = list(images_dir.glob('*.[jp][pn][g]'))
            split_stats['images'] = len(image_files)

            # Vérifier les images et labels
            bbox_areas = []
            for img_file in tqdm(image_files, desc=f"Vérification {split}"):
                # Vérifier l'intégrité de l'image
                try:
                    img = cv2.imread(str(img_file))
                    if img is None:
                        split_stats['corrupted_images'].append(img_file.name)
                        continue
                    img_height, img_width = img.shape[:2]
                except:
                    split_stats['corrupted_images'].append(img_file.name)
                    continue

                # Vérifier le fichier label correspondant
                label_file = labels_dir / (img_file.stem + '.txt')
                if not label_file.exists():
                    split_stats['missing_labels'].append(img_file.name)
                else:
                    split_stats['labels'] += 1

                    # Analyser le contenu du label
                    try:
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    split_stats['class_distribution'][class_id] += 1

                                    # Calculer l'aire de la bbox
                                    x_center, y_center, width, height = map(float, parts[1:5])
                                    area = width * height
                                    bbox_areas.append(area)
                    except Exception as e:
                        logger.warning(f"Erreur lecture label {label_file}: {e}")

            # Calculer les statistiques des bboxes
            if bbox_areas:
                split_stats['bbox_stats']['count'] = len(bbox_areas)
                split_stats['bbox_stats']['avg_per_image'] = len(bbox_areas) / max(split_stats['images'], 1)
                split_stats['bbox_stats']['min_area'] = min(bbox_areas)
                split_stats['bbox_stats']['max_area'] = max(bbox_areas)
                split_stats['bbox_stats']['avg_area'] = np.mean(bbox_areas)

            stats['splits'][split] = split_stats

            # Afficher les résultats
            logger.info(f"\n{split.upper()}:")
            logger.info(f"  Images: {split_stats['images']}")
            logger.info(f"  Labels: {split_stats['labels']}")
            if split_stats['missing_labels']:
                logger.warning(f"  Labels manquants: {len(split_stats['missing_labels'])}")
            if split_stats['corrupted_images']:
                logger.warning(f"  Images corrompues: {len(split_stats['corrupted_images'])}")
            if split_stats['class_distribution']:
                logger.info(f"  Distribution des classes: {dict(split_stats['class_distribution'])}")
            if split_stats['bbox_stats']['count'] > 0:
                logger.info(f"  Bboxes: {split_stats['bbox_stats']['count']} total")
                logger.info(f"  Moyenne par image: {split_stats['bbox_stats']['avg_per_image']:.2f}")

        return stats

    def prepare_all_datasets(self):
        """Prépare tous les datasets"""
        logger.info("\n" + "="*60)
        logger.info("PRÉPARATION DES DATASETS POUR YOLOV11")
        logger.info("="*60)

        datasets_to_prepare = [
            ('raquette', 'raquette', 'yolo'),
            ('tennis ball', 'tennis_ball', 'yolo'),
            ('tennis_court_id', 'tennis_court', 'coco')
        ]

        for dataset_dir, dataset_name, format_type in datasets_to_prepare:
            dataset_path = self.base_dir / dataset_dir

            if format_type == 'coco':
                # Convertir COCO vers YOLO
                logger.info(f"\nConversion COCO -> YOLO pour {dataset_name}")
                output_path = self.base_dir / f"{dataset_dir}_yolo"

                if not output_path.exists():
                    logger.info(f"Lancement de la conversion...")
                    # Importer et utiliser le convertisseur
                    from convert_coco_to_yolo import COCOToYOLOConverter
                    converter = COCOToYOLOConverter(
                        coco_dir=dataset_path,
                        output_dir=output_path,
                        copy_images=True
                    )
                    success = converter.convert()
                    if success:
                        converter.verify_conversion()
                        dataset_path = output_path
                    else:
                        logger.error(f"Échec de la conversion pour {dataset_name}")
                        continue
                else:
                    logger.info(f"Dataset YOLO déjà converti: {output_path}")
                    dataset_path = output_path

            # Vérifier le dataset
            stats = self.check_dataset_structure(dataset_path, dataset_name)
            if stats:
                self.stats[dataset_name] = stats

        # Générer le rapport
        self.generate_report()

    def generate_report(self):
        """Génère un rapport détaillé sur les datasets"""
        logger.info("\n" + "="*60)
        logger.info("RAPPORT FINAL")
        logger.info("="*60)

        report = {
            'datasets': self.stats,
            'summary': {
                'total_datasets': len(self.stats),
                'valid_datasets': sum(1 for s in self.stats.values() if s['valid']),
                'total_images': 0,
                'total_annotations': 0
            }
        }

        # Calculer les totaux
        for dataset_name, stats in self.stats.items():
            if stats['valid']:
                for split_stats in stats['splits'].values():
                    report['summary']['total_images'] += split_stats['images']
                    report['summary']['total_annotations'] += split_stats['bbox_stats']['count']

        # Sauvegarder le rapport
        report_path = self.base_dir.parent / 'dataset_report.yaml'
        with open(report_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)

        logger.info(f"\nRésumé:")
        logger.info(f"  Datasets valides: {report['summary']['valid_datasets']}/{report['summary']['total_datasets']}")
        logger.info(f"  Total images: {report['summary']['total_images']}")
        logger.info(f"  Total annotations: {report['summary']['total_annotations']}")
        logger.info(f"\nRapport sauvegardé: {report_path}")

        # Créer les visualisations si matplotlib disponible
        try:
            self.create_visualizations()
        except ImportError:
            logger.info("matplotlib non disponible, visualisations ignorées")

    def create_visualizations(self):
        """Crée des visualisations des statistiques des datasets"""
        vis_dir = self.base_dir.parent / 'visualizations'
        vis_dir.mkdir(exist_ok=True)

        # Configuration du style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

        # 1. Distribution des images par dataset et split
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Statistiques des Datasets YOLOv11', fontsize=16)

        # Graphique 1: Nombre d'images par dataset
        ax = axes[0, 0]
        datasets = []
        train_counts = []
        valid_counts = []
        test_counts = []

        for name, stats in self.stats.items():
            if stats['valid']:
                datasets.append(name)
                train_counts.append(stats['splits'].get('train', {}).get('images', 0))
                valid_counts.append(stats['splits'].get('valid', {}).get('images', 0))
                test_counts.append(stats['splits'].get('test', {}).get('images', 0))

        x = np.arange(len(datasets))
        width = 0.25

        ax.bar(x - width, train_counts, width, label='Train', color='blue', alpha=0.8)
        ax.bar(x, valid_counts, width, label='Valid', color='green', alpha=0.8)
        ax.bar(x + width, test_counts, width, label='Test', color='red', alpha=0.8)

        ax.set_xlabel('Dataset')
        ax.set_ylabel('Nombre d\'images')
        ax.set_title('Distribution des images')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Graphique 2: Distribution des annotations
        ax = axes[0, 1]
        for name, stats in self.stats.items():
            if stats['valid']:
                class_dist = defaultdict(int)
                for split_stats in stats['splits'].values():
                    for class_id, count in split_stats['class_distribution'].items():
                        class_dist[class_id] += count

                if class_dist:
                    classes = list(class_dist.keys())
                    counts = list(class_dist.values())
                    ax.bar(classes, counts, label=name, alpha=0.7)

        ax.set_xlabel('Classe ID')
        ax.set_ylabel('Nombre d\'annotations')
        ax.set_title('Distribution des classes')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Graphique 3: Qualité des données
        ax = axes[1, 0]
        quality_data = []
        for name, stats in self.stats.items():
            if stats['valid']:
                total_missing = sum(len(s.get('missing_labels', [])) for s in stats['splits'].values())
                total_corrupted = sum(len(s.get('corrupted_images', [])) for s in stats['splits'].values())
                total_ok = sum(s.get('images', 0) for s in stats['splits'].values()) - total_corrupted
                quality_data.append({
                    'dataset': name,
                    'OK': total_ok,
                    'Labels manquants': total_missing,
                    'Images corrompues': total_corrupted
                })

        if quality_data:
            df = pd.DataFrame(quality_data)
            df.set_index('dataset').plot(kind='bar', stacked=True, ax=ax,
                                        color=['green', 'orange', 'red'], alpha=0.8)
            ax.set_xlabel('Dataset')
            ax.set_ylabel('Nombre de fichiers')
            ax.set_title('Qualité des données')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

        # Graphique 4: Statistiques des bboxes
        ax = axes[1, 1]
        bbox_stats = []
        for name, stats in self.stats.items():
            if stats['valid']:
                avg_bboxes = []
                for split_stats in stats['splits'].values():
                    if split_stats['bbox_stats']['count'] > 0:
                        avg_bboxes.append(split_stats['bbox_stats']['avg_per_image'])
                if avg_bboxes:
                    bbox_stats.append({
                        'dataset': name,
                        'avg': np.mean(avg_bboxes)
                    })

        if bbox_stats:
            df = pd.DataFrame(bbox_stats)
            ax.bar(df['dataset'], df['avg'], color='purple', alpha=0.7)
            ax.set_xlabel('Dataset')
            ax.set_ylabel('Moyenne de bboxes par image')
            ax.set_title('Densité des annotations')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        vis_path = vis_dir / 'dataset_statistics.png'
        plt.savefig(vis_path, dpi=100, bbox_inches='tight')
        logger.info(f"Visualisations sauvegardées: {vis_path}")
        plt.close()

    def create_combined_yaml(self):
        """Crée un fichier YAML pour entraîner sur tous les datasets combinés"""
        combined_path = self.base_dir.parent / 'combined_dataset.yaml'

        # Collecter toutes les classes
        all_classes = []
        for name, stats in self.stats.items():
            if stats['valid']:
                yaml_file = Path(stats['path']) / 'dataset.yaml'
                if yaml_file.exists():
                    with open(yaml_file, 'r') as f:
                        data = yaml.safe_load(f)
                        all_classes.extend(data.get('names', []))

        # Créer le YAML combiné
        combined_config = {
            'path': str(self.base_dir.absolute()),
            'datasets': list(self.stats.keys()),
            'train': [f"{name}/train/images" for name in self.stats.keys()],
            'val': [f"{name}/valid/images" for name in self.stats.keys()],
            'test': [f"{name}/test/images" for name in self.stats.keys()],
            'nc': len(set(all_classes)),
            'names': list(set(all_classes))
        }

        with open(combined_path, 'w') as f:
            yaml.dump(combined_config, f, default_flow_style=False)

        logger.info(f"Configuration combinée créée: {combined_path}")


def main():
    parser = argparse.ArgumentParser(description='Préparer les datasets pour YOLOv11')
    parser.add_argument('--base-dir', type=str, default='training/dataset',
                       help='Répertoire de base contenant les datasets')
    parser.add_argument('--check-only', action='store_true',
                       help='Vérifier seulement sans convertir')
    parser.add_argument('--visualize', action='store_true',
                       help='Créer des visualisations')

    args = parser.parse_args()

    # Créer le préparateur
    preparer = DatasetPreparer(base_dir=args.base_dir)

    # Préparer tous les datasets
    preparer.prepare_all_datasets()

    # Créer un YAML combiné (optionnel)
    preparer.create_combined_yaml()

    logger.info("\n" + "="*60)
    logger.info("PRÉPARATION TERMINÉE")
    logger.info("="*60)
    logger.info("Vous pouvez maintenant lancer l'entraînement avec:")
    logger.info("  python training/train_yolov11.py")
    logger.info("\nOu pour un dataset spécifique:")
    logger.info("  python training/train_yolov11.py --single-dataset raquette")


if __name__ == "__main__":
    # Importer pandas si disponible (pour les visualisations)
    try:
        import pandas as pd
    except ImportError:
        pass

    main()