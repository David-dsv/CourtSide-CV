#!/usr/bin/env python3
"""
Script de correction pour le bug IndexError dans tracker.py
Exécutez ce script depuis le répertoire tennis_analysis
"""

import shutil
from pathlib import Path

def fix_tracker():
    """Applique le correctif au tracker.py"""
    
    tracker_file = Path("models/tracker.py")
    
    if not tracker_file.exists():
        print("❌ Erreur: models/tracker.py non trouvé!")
        print("Assurez-vous d'être dans le répertoire tennis_analysis")
        return False
    
    # Faire une sauvegarde
    backup_file = Path("models/tracker_original.py")
    shutil.copy(tracker_file, backup_file)
    print(f"✅ Sauvegarde créée: {backup_file}")
    
    # Lire le fichier
    with open(tracker_file, 'r') as f:
        lines = f.readlines()
    
    # Trouver et corriger la ligne problématique (autour de la ligne 236)
    modified = False
    for i in range(len(lines)):
        # Chercher la ligne problématique
        if 'pos = self.trackers[t].predict()[0]' in lines[i]:
            # Remplacer cette ligne et les suivantes
            indent = len(lines[i]) - len(lines[i].lstrip())
            
            # Nouvelle version corrigée
            new_lines = [
                lines[i].replace('[0]', ''),  # Enlever [0] de predict()
                ' ' * indent + '\n',
                ' ' * indent + '# FIX: Ensure pos is properly formatted as array\n',
                ' ' * indent + '# Handle the case where predict() might return different formats\n',
                ' ' * indent + 'if isinstance(pos, (list, tuple)):\n',
                ' ' * (indent + 4) + 'pos = pos[0] if len(pos) > 0 else pos\n',
                ' ' * indent + '\n',
                ' ' * indent + '# Convert to numpy array and flatten\n',
                ' ' * indent + 'pos = np.array(pos).flatten()\n',
                ' ' * indent + '\n',
                ' ' * indent + '# Check if we have a valid prediction with at least 4 values\n',
                ' ' * indent + 'if pos.size < 4:\n',
                ' ' * (indent + 4) + 'to_del.append(t)\n',
                ' ' * (indent + 4) + 'continue\n',
                ' ' * indent + '\n',
                ' ' * indent + '# Check for NaN values\n',
                ' ' * indent + 'if np.any(np.isnan(pos[:4])):\n',
                ' ' * (indent + 4) + 'to_del.append(t)\n',
                ' ' * (indent + 4) + 'continue\n',
                ' ' * indent + '\n',
                ' ' * indent + '# Store the tracker prediction\n'
            ]
            
            # Remplacer les lignes
            lines[i:i+1] = new_lines
            
            # Aussi chercher et modifier la ligne suivante trks[t] = ...
            for j in range(i+len(new_lines), min(i+len(new_lines)+5, len(lines))):
                if 'trks[t] = [pos[0], pos[1], pos[2], pos[3], 0]' in lines[j]:
                    # Cette ligne est maintenant correcte, pas besoin de la modifier
                    break
                    
            modified = True
            print("✅ Correction principale appliquée")
            break
    
    if not modified:
        print("⚠️ Pattern exact non trouvé, recherche alternative...")
        # Chercher juste la ligne problématique
        for i in range(len(lines)):
            if 'trks[t] = [pos[0], pos[1], pos[2], pos[3], 0]' in lines[i]:
                indent = len(lines[i]) - len(lines[i].lstrip())
                
                # Insérer les vérifications avant cette ligne
                check_lines = [
                    ' ' * indent + '# FIX: Handle potential scalar or invalid pos\n',
                    ' ' * indent + 'if not isinstance(pos, np.ndarray):\n',
                    ' ' * (indent + 4) + 'pos = np.array(pos)\n',
                    ' ' * indent + 'pos = pos.flatten()\n',
                    ' ' * indent + 'if pos.size < 4:\n',
                    ' ' * (indent + 4) + 'to_del.append(t)\n',
                    ' ' * (indent + 4) + 'continue\n',
                    ' ' * indent + '\n'
                ]
                
                lines[i:i] = check_lines
                modified = True
                print("✅ Correction alternative appliquée")
                break
    
    # Corriger aussi la méthode predict() dans KalmanBoxTracker
    for i in range(len(lines)):
        if 'def predict(self) -> np.ndarray:' in lines[i]:
            # Chercher le return dans cette méthode
            for j in range(i, min(i+20, len(lines))):
                if 'return self.history[-1]' in lines[j]:
                    # Remplacer pour retourner directement le bbox
                    lines[j] = lines[j].replace('self.history[-1]', 'bbox')
                    
                    # Ajouter la variable bbox avant
                    for k in range(j-1, i, -1):
                        if 'self.history.append(' in lines[k]:
                            indent = len(lines[k]) - len(lines[k].lstrip())
                            # Modifier pour stocker bbox d'abord
                            lines[k] = ' ' * indent + 'bbox = self._convert_x_to_bbox(self.kf.x)\n'
                            lines.insert(k+1, ' ' * indent + 'self.history.append(bbox)\n')
                            print("✅ Méthode predict() corrigée")
                            break
                    break
            break
    
    # Corriger get_state() pour toujours retourner un array 1D
    for i in range(len(lines)):
        if 'def get_state(self) -> np.ndarray:' in lines[i]:
            for j in range(i, min(i+10, len(lines))):
                if 'return' in lines[j] and '_convert_x_to_bbox' in lines[j]:
                    indent = len(lines[j]) - len(lines[j].lstrip())
                    new_lines = [
                        ' ' * indent + 'bbox = self._convert_x_to_bbox(self.kf.x)\n',
                        ' ' * indent + '# Ensure bbox is always returned as 1D array\n',
                        ' ' * indent + 'return bbox.flatten()\n'
                    ]
                    lines[j:j+1] = new_lines
                    print("✅ Méthode get_state() corrigée")
                    break
            break
    
    # Corriger aussi dans la méthode update() de OCSort où on utilise get_state()
    for i in range(len(lines)):
        if 'd = trk.get_state()[0]' in lines[i]:
            indent = len(lines[i]) - len(lines[i].lstrip())
            new_lines = [
                ' ' * indent + '# Get the current state\n',
                ' ' * indent + 'd = trk.get_state()\n',
                ' ' * indent + '\n',
                ' ' * indent + '# Ensure d is a 1D numpy array\n',
                ' ' * indent + 'd = np.array(d).flatten()\n'
            ]
            lines[i:i+1] = new_lines
            print("✅ Utilisation de get_state() corrigée")
            break
    
    # Ajouter une vérification avant l'utilisation de d
    for i in range(len(lines)):
        if 'track_data = np.concatenate([' in lines[i]:
            # Chercher en arrière pour ajouter une vérification
            for j in range(i-1, max(i-10, 0), -1):
                if 'if (trk.time_since_update < 1)' in lines[j]:
                    indent = len(lines[j]) - len(lines[j].lstrip())
                    # Ajouter une vérification de taille après le if
                    check_line = ' ' * (indent + 4) + '# Ensure d has at least 4 elements before using it\n'
                    size_check = ' ' * (indent + 4) + 'if d.size >= 4:\n'
                    
                    # Trouver où insérer
                    lines.insert(j+1, check_line)
                    lines.insert(j+2, size_check)
                    
                    # Indenter tout le bloc track_data
                    for k in range(j+3, i+5):
                        if k < len(lines) and not lines[k].strip().startswith('#'):
                            lines[k] = '    ' + lines[k]
                    
                    print("✅ Vérification de taille ajoutée")
                    break
            break
    
    # Corriger l'indentation de d.flatten() pour utiliser uniquement les 4 premiers éléments
    for i in range(len(lines)):
        if 'd.flatten(),' in lines[i]:
            lines[i] = lines[i].replace('d.flatten(),', 'd[:4],  # Only use first 4 elements (x1, y1, x2, y2)')
            print("✅ Utilisation de d corrigée")
            break
    
    # Écrire le fichier corrigé
    with open(tracker_file, 'w') as f:
        f.writelines(lines)
    
    print("\n✅ Toutes les corrections ont été appliquées!")
    print(f"📁 Fichier original sauvegardé: {backup_file}")
    print(f"📁 Fichier corrigé: {tracker_file}")
    
    return True

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Vérifier qu'on est dans le bon répertoire
    if not Path("main.py").exists() or not Path("models").exists():
        print("❌ Erreur: Ce script doit être exécuté depuis le répertoire tennis_analysis")
        print("   où se trouvent main.py et le dossier models/")
        sys.exit(1)
    
    success = fix_tracker()
    
    if success:
        print("\n🎾 Le bug a été corrigé! Vous pouvez maintenant relancer:")
        print('   python main.py --video "votre_video.mp4" --start-time 120 --end-time 220 --output-dir data/output')
    else:
        print("\n❌ Erreur lors de la correction. Utilisez le fichier tracker_corrected.py fourni.")