"""
Script pour valider et corriger les labels du dataset combiné
Détecte et corrige les problèmes courants :
- Coordonnées hors limites (< 0 ou > 1)
- Largeur/hauteur négatives ou nulles
- IDs de classe invalides
- Lignes mal formatées
"""

import os
from pathlib import Path
import logging
from tqdm import tqdm
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetValidator:
    """Valide et corrige les fichiers labels YOLO"""

    def __init__(self, dataset_dir, num_classes=10):
        self.dataset_dir = Path(dataset_dir)
        self.num_classes = num_classes
        self.issues = {
            'invalid_coords': 0,
            'invalid_class': 0,
            'invalid_format': 0,
            'empty_labels': 0,
            'fixed': 0,
            'removed': 0
        }

    def validate_bbox(self, parts):
        """
        Valide une bounding box YOLO

        Format: class_id x_center y_center width height
        Toutes les coordonnées doivent être entre 0 et 1
        """
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Vérifier l'ID de classe
            if class_id < 0 or class_id >= self.num_classes:
                return False, "invalid_class"

            # Vérifier les coordonnées
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                return False, "invalid_coords"

            if not (0 < width <= 1 and 0 < height <= 1):
                return False, "invalid_coords"

            # Vérifier que la bbox ne dépasse pas
            if (x_center - width/2) < 0 or (x_center + width/2) > 1:
                return False, "invalid_coords"

            if (y_center - height/2) < 0 or (y_center + height/2) > 1:
                return False, "invalid_coords"

            return True, None

        except (ValueError, IndexError):
            return False, "invalid_format"

    def fix_bbox(self, parts):
        """
        Tente de corriger une bounding box invalide
        """
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Clip les valeurs entre 0 et 1
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.001, min(1.0, width))
            height = max(0.001, min(1.0, height))

            # Ajuster pour que la bbox reste dans l'image
            x_center = max(width/2, min(1 - width/2, x_center))
            y_center = max(height/2, min(1 - height/2, y_center))

            return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

        except:
            return None

    def validate_label_file(self, label_path, fix=True):
        """
        Valide et corrige un fichier de labels

        Args:
            label_path: Chemin vers le fichier .txt
            fix: Si True, corrige les erreurs

        Returns:
            (is_valid, fixed_lines)
        """
        if not label_path.exists():
            return True, []

        with open(label_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            self.issues['empty_labels'] += 1
            return True, []

        fixed_lines = []
        file_has_issues = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                self.issues['invalid_format'] += 1
                file_has_issues = True
                continue

            # Valider la bbox
            is_valid, issue_type = self.validate_bbox(parts)

            if is_valid:
                fixed_lines.append(line + '\n')
            else:
                file_has_issues = True
                if issue_type:
                    self.issues[issue_type] += 1

                if fix:
                    # Tenter de corriger
                    fixed_line = self.fix_bbox(parts)
                    if fixed_line:
                        fixed_lines.append(fixed_line)
                        self.issues['fixed'] += 1

        return (not file_has_issues or fix), fixed_lines

    def process_split(self, split='train', fix=True):
        """
        Traite un split du dataset

        Args:
            split: 'train', 'valid' ou 'test'
            fix: Si True, corrige les fichiers
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Traitement du split: {split}")
        logger.info(f"{'='*60}")

        labels_dir = self.dataset_dir / split / 'labels'
        images_dir = self.dataset_dir / split / 'images'

        if not labels_dir.exists():
            logger.warning(f"Dossier labels non trouvé: {labels_dir}")
            return

        label_files = list(labels_dir.glob('*.txt'))
        logger.info(f"Fichiers de labels: {len(label_files)}")

        # Créer un backup
        if fix:
            backup_dir = self.dataset_dir / f'{split}_labels_backup'
            if not backup_dir.exists():
                shutil.copytree(labels_dir, backup_dir)
                logger.info(f"✓ Backup créé: {backup_dir}")

        files_to_remove = []

        for label_file in tqdm(label_files, desc=f"Validation {split}"):
            is_valid, fixed_lines = self.validate_label_file(label_file, fix=fix)

            if fix and not is_valid:
                if fixed_lines:
                    # Réécrire le fichier avec les corrections
                    with open(label_file, 'w') as f:
                        f.writelines(fixed_lines)
                else:
                    # Pas de labels valides, supprimer l'image et le label
                    files_to_remove.append(label_file)
                    self.issues['removed'] += 1

        # Supprimer les fichiers problématiques
        if fix and files_to_remove:
            logger.info(f"\nSuppression de {len(files_to_remove)} fichiers invalides...")
            for label_file in files_to_remove:
                # Supprimer le label
                label_file.unlink()

                # Supprimer l'image correspondante
                img_name = label_file.stem
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    img_file = images_dir / (img_name + ext)
                    if img_file.exists():
                        img_file.unlink()
                        break

        logger.info(f"\n✓ {split} traité")

    def validate_dataset(self, fix=True):
        """Valide le dataset complet"""
        logger.info("\n" + "="*60)
        logger.info("VALIDATION DU DATASET")
        logger.info("="*60)
        logger.info(f"Dataset: {self.dataset_dir}")
        logger.info(f"Classes: {self.num_classes}")
        logger.info(f"Mode: {'Correction' if fix else 'Vérification seule'}")

        # Traiter chaque split
        for split in ['train', 'valid', 'test']:
            self.process_split(split, fix=fix)

        # Résumé
        logger.info("\n" + "="*60)
        logger.info("RÉSUMÉ")
        logger.info("="*60)
        logger.info(f"Coordonnées invalides: {self.issues['invalid_coords']}")
        logger.info(f"Classes invalides: {self.issues['invalid_class']}")
        logger.info(f"Format invalide: {self.issues['invalid_format']}")
        logger.info(f"Labels vides: {self.issues['empty_labels']}")

        if fix:
            logger.info(f"\nCorrections:")
            logger.info(f"  Bboxes corrigées: {self.issues['fixed']}")
            logger.info(f"  Fichiers supprimés: {self.issues['removed']}")

        total_issues = (self.issues['invalid_coords'] +
                       self.issues['invalid_class'] +
                       self.issues['invalid_format'])

        if total_issues == 0:
            logger.info("\n✅ Dataset valide! Aucun problème détecté.")
        elif fix:
            logger.info(f"\n✅ Dataset corrigé! {total_issues} problèmes résolus.")
        else:
            logger.info(f"\n⚠️  {total_issues} problèmes détectés. Relancez avec --fix pour corriger.")

        return total_issues == 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Valider et corriger un dataset YOLO')
    parser.add_argument('--dataset', type=str, default='training/dataset_combined',
                       help='Chemin vers le dataset')
    parser.add_argument('--num-classes', type=int, default=10,
                       help='Nombre de classes')
    parser.add_argument('--fix', action='store_true',
                       help='Corriger automatiquement les problèmes')
    parser.add_argument('--check-only', action='store_true',
                       help='Vérifier seulement sans corriger')

    args = parser.parse_args()

    fix_mode = args.fix or not args.check_only

    # Créer le validateur
    validator = DatasetValidator(
        dataset_dir=args.dataset,
        num_classes=args.num_classes
    )

    # Valider le dataset
    is_valid = validator.validate_dataset(fix=fix_mode)

    if is_valid:
        logger.info("\n" + "="*60)
        logger.info("✅ DATASET PRÊT POUR L'ENTRAÎNEMENT")
        logger.info("="*60)
        logger.info("\nVous pouvez maintenant lancer:")
        logger.info("  python training/train_combined.py")
    elif not fix_mode:
        logger.info("\n" + "="*60)
        logger.info("⚠️  PROBLÈMES DÉTECTÉS")
        logger.info("="*60)
        logger.info("\nPour corriger automatiquement:")
        logger.info("  python training/validate_and_fix_dataset.py --fix")


if __name__ == "__main__":
    main()