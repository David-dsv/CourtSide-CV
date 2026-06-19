Voici le plan d'intégration concret et chiffré pour remplacer le tracking actuel par WASB dans `run_pipeline_8s.py`.

# PLAN D'INTÉGRATION WASB -> CourtSide-CV

## 1. Décision d'architecture : WASB remplace QUOI exactement ?

WASB remplace **la totalité de la Pass 1** (détection, filtre statique, tracking Kalman). WASB est un tracker par détection-heatmap end-to-end : il ne fait pas que détecter, il assure la cohérence temporelle.

*   **Lignes 602-624 (Pass 1a - YOLO) : SUPPRIMÉES.** YOLO26 est trop bruité pour la balle et génère le bug du filtre statique.
*   **Lignes 627-654 (Step 1b - Filtre statique) : SUPPRIMÉES.** (Voir point 2).
*   **Lignes 656-739 (Step 1c - Kalman) : SUPPRIMÉES.** La logique de cohérence temporelle de WASB (spatio-temporelle via HRNet) remplace le Kalman.
*   **Lignes 741-748 (Post-process - Spline) : SUPPRIMÉES.** WASB gère nativement les occultations de quelques frames. Pour les trous longs, on s'appuiera sur le lissage final.
*   **Ce qui reste :** La lecture vidéo (`VideoReader.iter_frames()`), et l'aval (`estimate_px_per_meter`, `detect_bounces_from_trajectory`, Pass 2).

## 2. Le filtre statique (1b) : Suppression ou garde-fou ?

**DÉCISION : Suppression totale et immédiate.**

**Justification :** Le bug n°1 vient de l'heuristique de quantification (grille 20px) qui détruit les trajectoires de service. WASB intègre dans son architecture même le rejet des faux positifs statiques : il utilise la différence de frames (motion) couplée aux heatmaps. Un logo de court génère une heatmap stable mais sans dynamique temporelle, WASB l'ignore naturellement. Rajouter le filtre statique par-dessus WASB casserait les vraies balles arrêtées (ex: balle qui roule sur le filet) sans aucun bénéfice.

## 3. Pseudocode d'intégration (Nouvelle Pass 1)

WASB attend des séquences de frames (MIMO). On utilise un buffer glissant.

```python
# Remplace les lignes 602 à 748 de run_pipeline_8s.py
import torch
from wasb_model import WASB  # Wrapper du repo nttcom/WASB-SBDT
import numpy as np
from scipy.signal import savgol_filter

# Init WASB
device = args.device # 'cuda', 'mps', ou 'cpu'
wasb = WASB(pretrained=True, device=device)
wasb.eval()

WINDOW_SIZE = 32  # WASB attend ~32 frames pour le contexte temporel
STRIDE = 16        # Chevauchement pour éviter les coupures de trajectoire

raw_ball_centers = [None] * total_frames
buffer = []
frame_indices = []

for idx, frame in enumerate(video_reader.iter_frames()):
    buffer.append(frame)
    frame_indices.append(idx)
    
    if len(buffer) == WINDOW_SIZE:
        # Inference par batch
        with torch.no_grad():
            # WASB retourne une liste de coords [(x,y), None, (x,y), ...]
            detections = wasb.track_sequence(buffer) 
        
        # On ne garde que les prédictions des frames du milieu (pour avoir le contexte avant/après)
        # Évite les effets de bord
        start = WINDOW_SIZE // 2 - STRIDE // 2
        end = WINDOW_SIZE // 2 + STRIDE // 2
        
        for i in range(start, end):
            real_idx = frame_indices[i]
            if raw_ball_centers[real_idx] is None:
                raw_ball_centers[real_idx] = detections[i]
        
        # Garder les dernières STRIDE frames pour le prochain batch
        buffer = buffer[-STRIDE:]
        frame_indices = frame_indices[-STRIDE:]

# Traitement du dernier buffer restant (si < WINDOW_SIZE, on pad ou on réduit la fenêtre)
if buffer:
    with torch.no_grad():
        detections = wasb.track_sequence(buffer)
    for i, det in enumerate(detections):
        if raw_ball_centers[frame_indices[i]] is None:
            raw_ball_centers[frame_indices[i]] = det

# Calcul des vitesses (voir point 4)
ball_speeds_px = compute_speeds_savgol(raw_ball_centers, fps)
all_ball_centers = raw_ball_centers # Pas de spline post-process, WASB gère l'interpolation courte
```

## 4. Recalcul de `ball_speeds_px` proprement

WASB ne sort pas de vitesse. La différence finie brute ($v_t = P_t - P_{t-1}$) est trop bruitée et casse `detect_bounces_from_trajectory` (qui cherche des pics de décélération). On utilise un **lissage Savitzky-Golay** sur les coordonnées, puis on dérive.

**Formule précise (implémentation Scipy) :**
1. Extraire les $X$ et $Y$ des `raw_ball_centers` (interpolation linéaire des `None` sur de max 5 frames pour la continuité mathématique).
2. Appliquer un filtre Savitzky-Golay d'ordre 2, fenêtre 7 (couvre ~0.1s à 60fps).
3. La vitesse est la norme de la dérivée première.

```python
def compute_speeds_savgol(centers, fps):
    n = len(centers)
    x = np.array([c[0] if c else np.nan for c in centers])
    y = np.array([c[1] if c else np.nan for c in centers])
    
    # Interp linéaire des courts trous (< 5 frames)
    x = pd.Series(x).interpolate(limit=5).bfill().ffill().values
    y = pd.Series(y).interpolate(limit=5).bfill().ffill().values
    
    # Savgol sur positions (dérivée=1 pour avoir la vitesse directe)
    # window_length doit être impair et < n
    win = min(7, n if n%2!=0 else n-1) 
    vx = savgol_filter(x, win, 2, deriv=1, delta=1.0/fps)
    vy = savgol_filter(y, win, 2, deriv=1, delta=1.0/fps)
    
    speeds = np.hypot(vx, vy)
    speeds[np.isnan(x)] = 0.0 # Là où on n'a jamais eu de balle
    return speeds.tolist()
```

## 5. Faut-il garder le Kalman ?

**NON.** Le supprimer complètement.
*   **Pourquoi :** WASB intègre déjà un mécanisme de cohérence temporelle (spatio-temporel MIMO). Empiler un Kalman par-dessus WASB créerait un conflit de tracking (deux prédicteurs temporels qui se battent). 
*   Le seul rôle qu'on pourrait vouloir garder au Kalman serait le "coast" (prédiction sur 30 frames de trou), mais WASB est conçu pour gérer l'occultation courte. Pour les trous de 700+ frames (balle hors champ), il ne faut **pas** inventer de positions : `None` est la bonne réponse, l'interpolation spline actuelle (qui invente la trajectoire sur 700 frames) est physiquement absurde et pollue `detect_bounces`.

## 6. WASB drop-in vs fine-tune

*   **Drop-in :** Les poids pré-entraînés WASB (sur Tennis et autres sports) sont très robustes. Ils gèrent mieux le flou de mouvement que YOLO.
*   **Risque OOD (Out Of Distribution) sur téléphone amateur :** **MODÉRÉ**. Le risque principal n'est pas la balle, mais le **mouvement de caméra**. WASB utilise la soustraction de fond/différence temporelle. Si le téléphone bouge beaucoup, le fond bouge, générant des faux positifs.
*   **Plan B (Si F1 rebond baisse) :** Il faudra fine-tuner le backbone HRNet de WASB. On prend 5 matchs amateurs (env. 2000 frames annotées manuellement via CVAT) et on ré-entraîne WASB pendant 10 époques avec un LR à $1e^{-4}$ en gelant les couches de bas niveau.

## 7. Gestion mémoire/perf et MPS

WASB nécessite de charger $N$ frames en tenseur `(N, C, H, W)`. À 1080p, 32 frames en FP32 = ~250MB de VRAM. C'est gérable, mais pour le streaming :

1.  **Intégration VideoReader :** Ne pas tout charger. Le pseudocode du point 3 utilise un buffer glissant de taille `WINDOW_SIZE` (32). On ne garde en RAM que 32 frames décodées à la fois.
2.  **Redimensionnement :** WASB attend souvent du 720p. Redimensionner les frames avant inference, puis remapper les coordonnées $(x,y)$ dans l'espace original via un simple ratio.
3.  **Faisabilité MPS (Apple Silicon) :** 
    *   **Faisable mais instable.** PyTorch MPS supporte les convolutions 3D/2D de HRNet.
    *   **Piège :** Le cache MPS ne se vide pas tout seul. Il faut ajouter `torch.mps.empty_cache()` à la fin de chaque inference de fenêtre.
    *   **Perf estimée :** CPU = ~2 fps ; MPS = ~8-10 fps ; CUDA = ~25 fps. Sur un clip de 8s (240 frames à 30fps), MPS prendra ~30s. Acceptable.

## 8. Plan de validation (F1 rebond)

L'objectif est de prouver que WASB améliore le F1 sur `felix.bounces.json` (16 rebonds).

1.  **Baseline (Run 0) :** Lancer le `run_pipeline_8s.py` actuel sur le clip de Félix. Enregistrer le F1 rebond, le nombre de faux positifs, et la longueur moyenne des tracks de balle. (Attendu : F1 ~0.4 à cause du filtre statique).
2.  **Run 1 (WASB Brut) :** Intégrer WASB sans toucher au reste. Lancer l'évaluateur.
    *   *Critère de succès :* F1 > 0.6. Si oui, WASB a corrigé le bug du filtre statique.
3.  **Run 2 (WASB + Savgol Speeds) :** Activer le calcul de vitesse par Savitzky-Golay (point 4).
    *   *Critère de succès :* F1 rebond > F1 du Run 1. (Si le F1 chute, c'est que le bruit de vitesse de WASB trompe `detect_bounces_from_trajectory`).
4.  **Analyse des échecs :** Si des rebonds manquent, vérifier si la balle est détectée par WASB juste avant le rebond. Si non -> OOD, fine-tune nécessaire.

## 9. Risques, pièges et ordre d'implémentation

**Risques spécifiques :**
*   **Échelle de coordonnées :** WASB risque de sortir des coords en 720p. Ne pas oublier de les multiplier par `(frame_width / wasb_width, frame_height / wasb_height)` avant de les mettre dans `raw_ball_centers`.
*   **Effet de bord MIMO :** Les prédictions de WASB sont moins fiables sur les premières et dernières frames d'un batch. D'où l'importance du chevauchement (`STRIDE = 16`) dans le pseudocode, où l'on ne conserve que le centre de la fenêtre.

**Ordre d'implémentation (du plus petit risque au plus intégré) :**

*   **Étape 1 (Isolation) :** Créer un script `test_wasb.py` indépendant. Charger 8s de vidéo de Félix, faire tourner WASB, et dumper les centres de balle dans un JSON. Valider visuellement que WASB trouve la balle dans la zone de service (là où l'ancien pipeline échouait).
*   **Étape 2 (Vitesse) :** Implémenter et tester `compute_speeds_savgol` sur le JSON de l'étape 1. Vérifier que les vitesses (en px/frame) sont cohérentes (entre 0 et 40 px/frame).
*   **Étape 3 (Branche pipeline) :** Remplacer les lignes 602-748 dans une copie de `run_pipeline_8s.py` par le pseudocode de la section 3. Lancer le pipeline complet.
*   **Étape 4 (Évaluation) :** Lancer le F1 evaluator sur la sortie du pipeline. Comparer à la baseline. Si F1 < Baseline, ne pas merger, analyser les faux négatifs de rebonds.
