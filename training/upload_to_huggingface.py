"""
Script pour uploader les modèles YOLOv11 vers Hugging Face Hub
"""

import os
from pathlib import Path
import shutil
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
import argparse


def prepare_model_repo(model_name, model_path, readme_path, output_dir='hf_repos'):
    """
    Prépare le dossier du repository pour upload

    Args:
        model_name: Nom du modèle (ex: "tennis-ball-yolov11")
        model_path: Chemin vers le fichier .pt
        readme_path: Chemin vers le README.md
        output_dir: Dossier de sortie temporaire
    """
    # Créer le dossier du repo
    repo_dir = Path(output_dir) / model_name
    repo_dir.mkdir(parents=True, exist_ok=True)

    # Copier le modèle
    model_dest = repo_dir / 'model.pt'
    shutil.copy2(model_path, model_dest)
    print(f"✓ Modèle copié: {model_dest}")

    # Copier le README
    readme_dest = repo_dir / 'README.md'
    shutil.copy2(readme_path, readme_dest)
    print(f"✓ README copié: {readme_dest}")

    # Créer un fichier de configuration
    config_content = f"""# Configuration
model_name: {model_name}
framework: ultralytics
architecture: yolov11n
task: object-detection
"""
    config_path = repo_dir / 'config.txt'
    with open(config_path, 'w') as f:
        f.write(config_content)
    print(f"✓ Config créée: {config_path}")

    # Créer un fichier requirements.txt
    requirements = """ultralytics>=8.0.0
torch>=2.0.0
opencv-python>=4.0.0
pillow>=9.0.0
numpy>=1.20.0
"""
    req_path = repo_dir / 'requirements.txt'
    with open(req_path, 'w') as f:
        f.write(requirements)
    print(f"✓ Requirements créé: {req_path}")

    # Créer un script d'exemple
    example_script = f"""#!/usr/bin/env python
# Example usage for {model_name}

from ultralytics import YOLO

# Load model from local file
model = YOLO('model.pt')

# Or download from Hugging Face (after upload)
# model = YOLO('hf://YOUR_USERNAME/{model_name}/model.pt')

# Predict on image
results = model.predict('image.jpg', conf=0.3)
results[0].show()

# Predict on video
results = model.predict('video.mp4', conf=0.3, save=True)
"""
    example_path = repo_dir / 'example.py'
    with open(example_path, 'w') as f:
        f.write(example_script)
    print(f"✓ Example créé: {example_path}")

    return repo_dir


def upload_to_hf(repo_dir, repo_name, username, token, private=False):
    """
    Upload le repository vers Hugging Face Hub

    Args:
        repo_dir: Dossier local du repository
        repo_name: Nom du repository sur HF
        username: Nom d'utilisateur Hugging Face
        token: Token d'authentification HF
        private: Si True, créer un repo privé
    """
    api = HfApi()

    # Créer le repository
    full_repo_name = f"{username}/{repo_name}"
    print(f"\n📦 Création du repository: {full_repo_name}")

    try:
        create_repo(
            repo_id=full_repo_name,
            token=token,
            private=private,
            exist_ok=True
        )
        print(f"✓ Repository créé/trouvé: https://huggingface.co/{full_repo_name}")
    except Exception as e:
        print(f"❌ Erreur création repo: {e}")
        return False

    # Upload les fichiers
    print(f"\n📤 Upload des fichiers...")
    try:
        api.upload_folder(
            folder_path=str(repo_dir),
            repo_id=full_repo_name,
            token=token,
            repo_type="model"
        )
        print(f"✓ Fichiers uploadés avec succès!")
        print(f"\n🎉 Modèle disponible sur: https://huggingface.co/{full_repo_name}")
        return True
    except Exception as e:
        print(f"❌ Erreur upload: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Upload YOLOv11 models to Hugging Face')
    parser.add_argument('--username', type=str, required=True,
                       help='Hugging Face username')
    parser.add_argument('--token', type=str, required=True,
                       help='Hugging Face token (get it from https://huggingface.co/settings/tokens)')
    parser.add_argument('--model', type=str, choices=['ball', 'racket', 'both', 'combined'], default='combined',
                       help='Which model to upload')
    parser.add_argument('--private', action='store_true',
                       help='Create private repository')

    args = parser.parse_args()

    training_dir = Path(__file__).parent

    models_to_upload = []

    if args.model in ['ball', 'both']:
        models_to_upload.append({
            'name': 'tennis-ball-yolov11',
            'model_path': training_dir / 'tennis_ball_subset_best.pt',
            'readme_path': training_dir / 'README_tennis_ball.md',
            'description': 'YOLOv11 Tennis Ball Detection'
        })

    if args.model in ['racket', 'both']:
        models_to_upload.append({
            'name': 'tennis-racket-yolov11',
            'model_path': training_dir / 'raquette_subset_best.pt',
            'readme_path': training_dir / 'README_raquette.md',
            'description': 'YOLOv11 Tennis Racket Detection'
        })

    if args.model == 'combined':
        models_to_upload.append({
            'name': 'CourtSide-Computer-Vision-v1',
            'model_path': training_dir / 'runs_combined' / 'combined_final' / 'weights' / 'best.pt',
            'readme_path': training_dir / 'README_MODEL.md',
            'description': 'YOLOv11 Complete Tennis Detection (10 classes)'
        })

    print("="*60)
    print("UPLOAD YOLOV11 MODELS TO HUGGING FACE")
    print("="*60)
    print(f"Username: {args.username}")
    print(f"Models: {args.model}")
    print(f"Private: {args.private}")
    print("="*60)

    for model_info in models_to_upload:
        print(f"\n{'='*60}")
        print(f"Processing: {model_info['name']}")
        print(f"{'='*60}")

        # Vérifier que les fichiers existent
        if not model_info['model_path'].exists():
            print(f"❌ Modèle non trouvé: {model_info['model_path']}")
            continue

        if not model_info['readme_path'].exists():
            print(f"❌ README non trouvé: {model_info['readme_path']}")
            continue

        # Préparer le repository
        repo_dir = prepare_model_repo(
            model_name=model_info['name'],
            model_path=model_info['model_path'],
            readme_path=model_info['readme_path']
        )

        # Upload vers Hugging Face
        success = upload_to_hf(
            repo_dir=repo_dir,
            repo_name=model_info['name'],
            username=args.username,
            token=args.token,
            private=args.private
        )

        if success:
            print(f"\n✅ {model_info['name']} uploadé avec succès!")
        else:
            print(f"\n❌ Échec upload {model_info['name']}")

    print("\n" + "="*60)
    print("UPLOAD TERMINÉ")
    print("="*60)
    print("\n📚 Vos modèles sont maintenant disponibles sur:")
    print(f"https://huggingface.co/{args.username}")


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════╗
║          UPLOAD YOLOV11 TO HUGGING FACE HUB                    ║
╚════════════════════════════════════════════════════════════════╝

Avant de commencer:

1. Créez un compte sur https://huggingface.co (si pas déjà fait)

2. Obtenez votre token d'accès:
   → https://huggingface.co/settings/tokens
   → Cliquez "New token"
   → Donnez un nom (ex: "yolov11-upload")
   → Sélectionnez "write" permissions
   → Copiez le token

3. Installez la dépendance:
   pip install huggingface_hub

4. Lancez l'upload:
   python training/upload_to_huggingface.py \\
       --username VOTRE_USERNAME \\
       --token VOTRE_TOKEN \\
       --model both

Options:
  --model ball      # Upload seulement tennis ball
  --model racket    # Upload seulement raquette
  --model both      # Upload les deux (défaut)
  --private         # Créer des repos privés

════════════════════════════════════════════════════════════════
""")

    main()