# PROMPT — S4 : homographie court AUTOMATIQUE oblique (Backend R&D+impl)

> **Session AUTONOME ULTRACODE.** Lance en `/ultracode`.
> ⛔ **NE POSE JAMAIS DE QUESTION. Décide, documente, livre.**
> ✅ **RECHERCHE WEB ENCOURAGÉE** : cherche les meilleures approches 2024-2026 de
> détection automatique de court de tennis / homographie (TennisCourtDetector,
> line+RANSAC, vanishing points, RF-DETR keypoint, court segmentation, PnP)…
> Évalue licence (le bloqueur SaaS connu = yastrebksv sans LICENSE → éviter),
> faisabilité no-GPU, et implémente la voie classique recommandée.

## Le problème (le bloqueur SaaS) + le R&D déjà fait
La calibration court métrique marche en SEMI-AUTO (clic 4 coins). L'AUTOMATIQUE (sans clic) manque sur angles obliques/amateurs. **Un doc R&D existe déjà** (à lire et IMPLÉMENTER) : `docs/research/auto-homography-oblique.md` (branche `origin/docs/auto-homography-brainstorm` — `git show origin/docs/auto-homography-brainstorm:docs/research/auto-homography-oblique.md`). Il recommande la voie **classique no-GPU** : masque lignes blanches → HoughLinesP → RANSAC vanishing points → 4 coins → `cv2.findHomography` → warp-back self-score gate. Implémente-la derrière `--auto-court` (défaut OFF).

## Le thermomètre (à CONSTRUIRE — pas de GT court encore)
- Crée un scorer de **re-projection** : projette le court modèle ITF via l'homographie candidate, mesure l'alignement aux lignes détectées (le warp-back self-score du doc). C'est la métrique de confiance ET de gating.
- Clips de test (mesurés dans le doc R&D) : felix (grazing, 1 famille de lignes → doit ABSTENIR proprement), tennis broadcast (keypoint gagne déjà), et le clip Djokovic court-level si présent (`*Djokovic*`). Mesure le taux de solve + la qualité sur chacun.
- ⚠️ Sur felix (grazing), la bonne réponse est **abstenir** (honnête), pas forcer une homographie fausse.

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-s4homog feat/s4-auto-homography
```
Branche `feat/s4-auto-homography` (depuis `feat/accuracy-overhaul`). Édite SOUS le worktree. **Fichiers : `models/court_detector.py` + `models/court_calibrator.py` + un nouveau `models/court_auto.py` (la voie classique) + le flag dans `run_pipeline_8s.py` (zone court, disjointe des autres sessions).** NE touche PAS `vision/`, `models/yolo_detector.py`, `models/wasb_ball_tracker.py`.

## Pistes (du doc R&D + web)
1. **Voie classique no-GPU** (priorité) : line mask → HoughLinesP → 2 familles d'orientation → RANSAC VP → correspondance 4 coins ITF → findHomography → self-score. Le cœur dur = associer chaque ligne détectée à sa ligne ITF nommée (le doc le signale non résolu — résous-le par mesure).
2. **Gate de confiance** : warp-back self-score → high/low/abstain. Ne JAMAIS émettre une homographie fausse (felix grazing = abstain).
3. **Licence** : éviter yastrebksv (no LICENSE). Si un modèle keypoint licence-clean (Apache) existe (recherche web), évalue-le comme option GPU-gated, mais livre d'abord le classique no-GPU.

## Contraintes dures
- `--auto-court` **défaut OFF** ; ON = la voie classique. OFF = comportement actuel inchangé (semi-auto/keypoint). Ne régresse RIEN (tous les tests verts ; le speed/depth sur broadcast garde sa précision).
- **Zéro hardcoding** (dimensions ITF = vraies constantes documentées, pas du tuning vidéo) ; **zéro triche**. Licence vérifiée.

## Boucle + livrable
Itère SANS plafond. CR `docs/research/s4-auto-homography-CR.md` : journal (taux de solve + qualité re-projection par clip, dont l'abstention felix), technos web évaluées (+licence), ce qui marche, limites honnêtes (grazing, association de lignes), idée réutilisable. Commit + push. Chiffres mesurés.
