"""
Script de conversion du format COCO vers le format YOLO
Spécifiquement conçu pour le dataset tennis_court_id
"""

import json
import os
import shutil
from pathlib import Path
import cv2
import logging
from tqdm import tqdm
import argparse

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class COCOToYOLOConverter:
    """Convertisseur de format COCO vers YOLO pour la détection d'objets"""

    def __init__(self, coco_dir, output_dir, copy_images=True):
        """
        Initialise le convertisseur

        Args:
            coco_dir: Chemin vers le dossier COCO
            output_dir: Chemin de sortie pour le format YOLO
            copy_images: Si True, copie les images; sinon, crée des liens symboliques
        """
        self.coco_dir = Path(coco_dir)
        self.output_dir = Path(output_dir)
        self.copy_images = copy_images

        # Créer la structure de sortie
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Mapping des catégories
        self.category_mapping = {}

    def load_coco_annotations(self, annotation_file):
        """
        Charge les annotations COCO

        Args:
            annotation_file: Chemin vers le fichier d'annotations COCO

        Returns:
            Dictionnaire des annotations COCO
        """
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        logger.info(f"Chargé {len(coco_data['images'])} images")
        logger.info(f"Chargé {len(coco_data['annotations'])} annotations")
        logger.info(f"Catégories: {[cat['name'] for cat in coco_data['categories']]}")

        return coco_data

    def create_category_mapping(self, categories):
        """
        Crée un mapping des catégories COCO vers des indices YOLO

        Args:
            categories: Liste des catégories COCO

        Returns:
            Dictionnaire de mapping {coco_id: yolo_idx}
        """
        self.category_mapping = {}
        self.category_names = []

        for idx, category in enumerate(categories):
            self.category_mapping[category['id']] = idx
            self.category_names.append(category['name'])

        logger.info(f"Mapping des catégories créé: {self.category_mapping}")
        return self.category_mapping

    def convert_bbox_to_yolo(self, bbox, img_width, img_height):
        """
        Convertit une bounding box COCO [x, y, width, height] vers YOLO [x_center, y_center, width, height]
        Les valeurs sont normalisées entre 0 et 1

        Args:
            bbox: Bounding box au format COCO [x, y, width, height]
            img_width: Largeur de l'image
            img_height: Hauteur de l'image

        Returns:
            Bounding box au format YOLO normalisé
        """
        x, y, width, height = bbox

        # Calculer le centre
        x_center = (x + width / 2) / img_width
        y_center = (y + height / 2) / img_height

        # Normaliser width et height
        width_norm = width / img_width
        height_norm = height / img_height

        return x_center, y_center, width_norm, height_norm

    def convert_segmentation_to_yolo(self, segmentation, img_width, img_height):
        """
        Convertit une segmentation COCO vers le format YOLO (pour YOLOv8 segment)

        Args:
            segmentation: Liste de points de segmentation
            img_width: Largeur de l'image
            img_height: Hauteur de l'image

        Returns:
            Points de segmentation normalisés
        """
        if not segmentation or len(segmentation) == 0:
            return None

        # Prendre le premier polygone
        polygon = segmentation[0]

        # Normaliser les coordonnées
        normalized_points = []
        for i in range(0, len(polygon), 2):
            x = polygon[i] / img_width
            y = polygon[i + 1] / img_height
            normalized_points.extend([x, y])

        return normalized_points

    def process_split(self, split_name, coco_data):
        """
        Traite un split spécifique (train, val, test)

        Args:
            split_name: Nom du split ('train', 'valid', 'test')
            coco_data: Données COCO pour ce split
        """
        logger.info(f"\nTraitement du split: {split_name}")

        # Créer les dossiers pour ce split
        images_dir = self.output_dir / split_name / 'images'
        labels_dir = self.output_dir / split_name / 'labels'
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Créer un dictionnaire image_id -> image_info
        images_dict = {img['id']: img for img in coco_data['images']}

        # Créer un dictionnaire image_id -> annotations
        annotations_dict = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_dict:
                annotations_dict[img_id] = []
            annotations_dict[img_id].append(ann)

        # Traiter chaque image
        for img_id, img_info in tqdm(images_dict.items(), desc=f"Conversion {split_name}"):
            # Nom du fichier
            img_filename = img_info['file_name']
            img_name_without_ext = Path(img_filename).stem

            # Chemin source de l'image
            src_img_path = self.coco_dir / split_name / 'images' / img_filename
            if not src_img_path.exists():
                # Essayer d'autres chemins possibles
                src_img_path = self.coco_dir / 'images' / img_filename
                if not src_img_path.exists():
                    logger.warning(f"Image non trouvée: {img_filename}")
                    continue

            # Chemin destination de l'image
            dst_img_path = images_dir / img_filename

            # Copier ou créer un lien symbolique
            if self.copy_images:
                if not dst_img_path.exists():
                    shutil.copy2(src_img_path, dst_img_path)
            else:
                if not dst_img_path.exists():
                    dst_img_path.symlink_to(src_img_path)

            # Obtenir les dimensions de l'image
            img_width = img_info['width']
            img_height = img_info['height']

            # Créer le fichier de labels YOLO
            label_file = labels_dir / f"{img_name_without_ext}.txt"

            # Obtenir les annotations pour cette image
            if img_id in annotations_dict:
                with open(label_file, 'w') as f:
                    for ann in annotations_dict[img_id]:
                        # Obtenir la classe YOLO
                        coco_cat_id = ann['category_id']
                        if coco_cat_id not in self.category_mapping:
                            logger.warning(f"Catégorie inconnue: {coco_cat_id}")
                            continue

                        yolo_class_id = self.category_mapping[coco_cat_id]

                        # Convertir la bounding box
                        if 'bbox' in ann:
                            bbox = ann['bbox']
                            x_center, y_center, width, height = self.convert_bbox_to_yolo(
                                bbox, img_width, img_height
                            )

                            # Écrire au format YOLO
                            f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                        # Si segmentation disponible (pour YOLOv8 segment)
                        if 'segmentation' in ann and ann['segmentation']:
                            seg_points = self.convert_segmentation_to_yolo(
                                ann['segmentation'], img_width, img_height
                            )
                            if seg_points:
                                # Format: class x1 y1 x2 y2 ... xn yn
                                seg_str = ' '.join([f"{p:.6f}" for p in seg_points])
                                # Décommentez pour utiliser la segmentation
                                # f.write(f"{yolo_class_id} {seg_str}\n")
            else:
                # Créer un fichier de labels vide si pas d'annotations
                open(label_file, 'w').close()

        logger.info(f"Conversion {split_name} terminée")

    def convert(self):
        """Effectue la conversion complète COCO vers YOLO"""
        logger.info(f"Début de la conversion COCO -> YOLO")
        logger.info(f"Source: {self.coco_dir}")
        logger.info(f"Destination: {self.output_dir}")

        # Chercher les fichiers d'annotations
        splits_to_process = []

        # Vérifier les différentes structures possibles
        for split in ['train', 'val', 'valid', 'test']:
            # Structure 1: annotations dans le dossier du split
            ann_path1 = self.coco_dir / split / '_annotations.coco.json'
            ann_path2 = self.coco_dir / split / 'annotations.json'

            # Structure 2: annotations dans un dossier séparé
            ann_path3 = self.coco_dir / 'annotations' / f'{split}.json'
            ann_path4 = self.coco_dir / 'annotations' / f'instances_{split}.json'

            # Structure 3: un seul fichier d'annotations
            ann_path5 = self.coco_dir / '_annotations.coco.json'

            for ann_path in [ann_path1, ann_path2, ann_path3, ann_path4]:
                if ann_path.exists():
                    splits_to_process.append((split, ann_path))
                    break

        # Si un seul fichier d'annotations pour tous les splits
        if not splits_to_process:
            ann_path = self.coco_dir / '_annotations.coco.json'
            if ann_path.exists():
                logger.info("Fichier d'annotations unique trouvé")
                coco_data = self.load_coco_annotations(ann_path)
                self.create_category_mapping(coco_data['categories'])

                # Diviser les données en train/val/test (80/10/10)
                total_images = len(coco_data['images'])
                train_end = int(total_images * 0.8)
                val_end = int(total_images * 0.9)

                # Créer les splits
                for split, start_idx, end_idx in [
                    ('train', 0, train_end),
                    ('valid', train_end, val_end),
                    ('test', val_end, total_images)
                ]:
                    split_data = {
                        'images': coco_data['images'][start_idx:end_idx],
                        'annotations': [],
                        'categories': coco_data['categories']
                    }

                    # Filtrer les annotations
                    img_ids = {img['id'] for img in split_data['images']}
                    split_data['annotations'] = [
                        ann for ann in coco_data['annotations']
                        if ann['image_id'] in img_ids
                    ]

                    self.process_split(split, split_data)
            else:
                logger.error("Aucun fichier d'annotations COCO trouvé!")
                return False
        else:
            # Traiter chaque split trouvé
            first_split = True
            for split_name, ann_path in splits_to_process:
                logger.info(f"Traitement de {split_name}: {ann_path}")
                coco_data = self.load_coco_annotations(ann_path)

                if first_split:
                    self.create_category_mapping(coco_data['categories'])
                    first_split = False

                # Mapper 'val' vers 'valid' pour YOLO
                output_split = 'valid' if split_name == 'val' else split_name
                self.process_split(output_split, coco_data)

        # Créer le fichier de configuration YOLO
        self.create_yolo_yaml()

        logger.info("\nConversion terminée avec succès!")
        return True

    def create_yolo_yaml(self):
        """Crée le fichier YAML de configuration pour YOLO"""
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': len(self.category_names),
            'names': self.category_names
        }

        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            import yaml
            yaml.dump(yaml_content, f, default_flow_style=False)

        logger.info(f"Fichier de configuration créé: {yaml_path}")

    def verify_conversion(self):
        """Vérifie la conversion en comptant les fichiers"""
        logger.info("\nVérification de la conversion:")

        for split in ['train', 'valid', 'test']:
            images_dir = self.output_dir / split / 'images'
            labels_dir = self.output_dir / split / 'labels'

            if images_dir.exists() and labels_dir.exists():
                n_images = len(list(images_dir.glob('*')))
                n_labels = len(list(labels_dir.glob('*.txt')))
                logger.info(f"{split}: {n_images} images, {n_labels} labels")
            else:
                logger.info(f"{split}: Non trouvé")


def main():
    parser = argparse.ArgumentParser(description='Convertir dataset COCO vers format YOLO')
    parser.add_argument('--input', type=str, default='training/dataset/tennis_court_id',
                       help='Chemin vers le dataset COCO')
    parser.add_argument('--output', type=str, default='training/dataset/tennis_court_id_yolo',
                       help='Chemin de sortie pour le dataset YOLO')
    parser.add_argument('--copy-images', action='store_true',
                       help='Copier les images au lieu de créer des liens symboliques')

    args = parser.parse_args()

    # Créer le convertisseur
    converter = COCOToYOLOConverter(
        coco_dir=args.input,
        output_dir=args.output,
        copy_images=args.copy_images
    )

    # Effectuer la conversion
    success = converter.convert()

    if success:
        # Vérifier la conversion
        converter.verify_conversion()
        logger.info(f"\nDataset YOLO créé dans: {args.output}")
        logger.info("Vous pouvez maintenant utiliser train_yolov11.py pour l'entraînement")
    else:
        logger.error("Échec de la conversion")


if __name__ == "__main__":
    main()