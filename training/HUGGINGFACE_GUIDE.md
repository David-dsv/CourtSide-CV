# Guide d'Upload vers Hugging Face Hub 🤗

Ce guide vous explique comment uploader vos modèles YOLOv11 sur Hugging Face.

## 📋 Prérequis

### 1. Créer un Compte Hugging Face

Si vous n'avez pas encore de compte :
1. Allez sur [huggingface.co](https://huggingface.co/join)
2. Créez un compte gratuit
3. Vérifiez votre email

### 2. Obtenir un Token d'Accès

1. Connectez-vous à [huggingface.co](https://huggingface.co)
2. Allez dans **Settings** → **Access Tokens**
   - Direct link: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Cliquez sur **"New token"**
4. Donnez un nom au token (ex: `yolov11-upload`)
5. Sélectionnez **"Write"** permissions
6. Copiez le token (gardez-le secret !)

### 3. Installer la Librairie

```bash
pip install huggingface_hub
```

## 🚀 Upload Simple

### Option 1 : Upload les Deux Modèles

```bash
python training/upload_to_huggingface.py \
    --username VOTRE_USERNAME \
    --token VOTRE_TOKEN \
    --model both
```

### Option 2 : Upload Seulement Tennis Ball

```bash
python training/upload_to_huggingface.py \
    --username VOTRE_USERNAME \
    --token VOTRE_TOKEN \
    --model ball
```

### Option 3 : Upload Seulement Raquette

```bash
python training/upload_to_huggingface.py \
    --username VOTRE_USERNAME \
    --token VOTRE_TOKEN \
    --model racket
```

### Option 4 : Créer des Repos Privés

```bash
python training/upload_to_huggingface.py \
    --username VOTRE_USERNAME \
    --token VOTRE_TOKEN \
    --model both \
    --private
```

## 📦 Ce Qui Sera Uploadé

Pour chaque modèle, un repository sera créé avec :

```
tennis-ball-yolov11/
├── README.md              # Model card complète
├── model.pt               # Modèle YOLOv11 (5.4 MB)
├── config.txt             # Configuration
├── requirements.txt       # Dépendances
└── example.py            # Script d'exemple
```

## 🔗 URLs des Repositories

Après l'upload, vos modèles seront disponibles à :

- **Tennis Ball**: `https://huggingface.co/VOTRE_USERNAME/tennis-ball-yolov11`
- **Raquette**: `https://huggingface.co/VOTRE_USERNAME/tennis-racket-yolov11`

## 💻 Utiliser les Modèles Depuis Hugging Face

### Télécharger et Utiliser

```python
from ultralytics import YOLO

# Charger depuis Hugging Face Hub
model = YOLO('hf://VOTRE_USERNAME/tennis-ball-yolov11/model.pt')

# Ou télécharger d'abord
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="VOTRE_USERNAME/tennis-ball-yolov11",
    filename="model.pt"
)
model = YOLO(model_path)

# Utiliser
results = model.predict('image.jpg', conf=0.3)
results[0].show()
```

### Depuis la CLI

```bash
# Télécharger le modèle
huggingface-cli download VOTRE_USERNAME/tennis-ball-yolov11 model.pt

# Utiliser avec YOLO
yolo detect predict model=model.pt source=image.jpg
```

## 🎨 Personnaliser les Model Cards

### Tennis Ball Model Card

Éditez `training/README_tennis_ball.md` avant l'upload :

```markdown
---
language: en
license: mit
tags:
  - yolo
  - tennis
  - object-detection
---

# Votre Titre Personnalisé

Votre description...
```

### Raquette Model Card

Éditez `training/README_raquette.md`

## 📊 Ajouter des Images Exemple

Pour rendre vos model cards plus attractives :

1. Créez un dossier `images/` dans votre repo
2. Ajoutez des images d'exemple
3. Éditez le README pour inclure les images :

```markdown
## Example Results

![Example 1](images/example1.jpg)
![Example 2](images/example2.jpg)
```

## 🔄 Mettre à Jour un Modèle

Si vous réentraînez et voulez mettre à jour :

```bash
# Réentraîner
python training/train_yolov11_subset.py --epochs 150

# Re-upload (écrase l'ancien)
python training/upload_to_huggingface.py \
    --username VOTRE_USERNAME \
    --token VOTRE_TOKEN \
    --model both
```

## 🔒 Sécurité du Token

### ⚠️ IMPORTANT : Ne Jamais Partager Votre Token !

**Méthode 1 : Variable d'Environnement (Recommandé)**

```bash
# Définir le token
export HF_TOKEN="votre_token_ici"

# Utiliser sans le token dans la commande
python training/upload_to_huggingface.py \
    --username VOTRE_USERNAME \
    --token $HF_TOKEN \
    --model both
```

**Méthode 2 : Fichier .env**

```bash
# Créer .env
echo "HF_TOKEN=votre_token_ici" > .env

# Ajouter au .gitignore
echo ".env" >> .gitignore
```

Puis dans le script :
```python
from dotenv import load_dotenv
load_dotenv()
token = os.getenv("HF_TOKEN")
```

**Méthode 3 : Login Hugging Face CLI**

```bash
# Login une seule fois
huggingface-cli login

# Ensuite le token est sauvegardé
python training/upload_to_huggingface.py \
    --username VOTRE_USERNAME \
    --model both
```

## 📝 Checklist Avant Upload

- [ ] Les modèles sont entraînés et testés
- [ ] Les Model Cards sont à jour (README_*.md)
- [ ] Vous avez un compte Hugging Face
- [ ] Vous avez obtenu un token avec permissions "write"
- [ ] `huggingface_hub` est installé
- [ ] Vous avez choisi si public ou privé

## 🎯 Bonnes Pratiques

### Noms de Repository

- ✅ `tennis-ball-yolov11` (clair et descriptif)
- ✅ `tennis-racket-detector` (explicite)
- ❌ `model1` (trop vague)
- ❌ `my-yolo` (pas assez descriptif)

### Tags

Ajoutez des tags pertinents dans le YAML frontmatter :

```yaml
tags:
  - yolo
  - yolov11
  - tennis
  - sports
  - object-detection
  - computer-vision
```

### Description

Écrivez une description claire :
- Ce que fait le modèle
- Sur quoi il a été entraîné
- Performances attendues
- Cas d'usage

## 🔍 Vérifier l'Upload

Après l'upload, vérifiez :

1. **Model Card** : Le README s'affiche correctement
2. **Fichiers** : Tous les fichiers sont présents
3. **Téléchargement** : Vous pouvez télécharger le modèle
4. **Test** : Le modèle fonctionne depuis HF Hub

```python
# Test rapide
from ultralytics import YOLO
model = YOLO('hf://VOTRE_USERNAME/tennis-ball-yolov11/model.pt')
print("✓ Modèle chargé avec succès!")
```

## 📈 Statistiques et Likes

- Les utilisateurs peuvent liker votre modèle ⭐
- Vous pouvez voir les téléchargements 📊
- Les modèles populaires apparaissent en trending 🔥

## 🤝 Partager Vos Modèles

Une fois uploadés, partagez-les :

```markdown
# Sur README.md principal
Check out my models:
- [Tennis Ball Detection](https://huggingface.co/USERNAME/tennis-ball-yolov11)
- [Tennis Racket Detection](https://huggingface.co/USERNAME/tennis-racket-yolov11)
```

## 🐛 Résolution de Problèmes

### Erreur : "Invalid token"
- Vérifiez que votre token a les permissions "write"
- Régénérez le token si nécessaire

### Erreur : "Repository not found"
- Vérifiez votre username
- Créez d'abord le repo sur HF (ou laissez le script le faire)

### Erreur : "File too large"
- Les modèles YOLOv11n font ~5MB, donc OK
- Pour modèles >5GB, utilisez Git LFS

### Erreur : "huggingface_hub not found"
```bash
pip install --upgrade huggingface_hub
```

## 📚 Resources

- [Hugging Face Docs](https://huggingface.co/docs)
- [Model Cards Guide](https://huggingface.co/docs/hub/model-cards)
- [Ultralytics Hub](https://hub.ultralytics.com/)

## ✅ Exemple Complet

```bash
# 1. Installation
pip install huggingface_hub

# 2. Login (une seule fois)
huggingface-cli login

# 3. Upload
python training/upload_to_huggingface.py \
    --username mon_username \
    --token hf_XXX \
    --model both

# 4. Test
python -c "
from ultralytics import YOLO
model = YOLO('hf://mon_username/tennis-ball-yolov11/model.pt')
print('✓ Success!')
"
```

## 🎉 Vous Êtes Prêt !

Vos modèles seront maintenant :
- ✅ Accessibles mondialement
- ✅ Faciles à télécharger et utiliser
- ✅ Bien documentés
- ✅ Partageables avec la communauté

---

**Questions ?** Consultez la [documentation Hugging Face](https://huggingface.co/docs) ou ouvrez une issue sur GitHub.