# PROMPT — S5 : forehand/backhand fiable (Backend R&D+impl)

> **Session AUTONOME ULTRACODE.** Lance en `/ultracode`.
> ⛔ **NE POSE JAMAIS DE QUESTION. Décide, documente, livre.**
> ✅ **RECHERCHE WEB ENCOURAGÉE** : cherche les méthodes 2024-2026 de classification
> de coups de tennis depuis la pose (forehand/backhand/serve, swing classification,
> pose-based stroke recognition, handedness inference)… Évalue + implémente.

## Le problème + le R&D déjà fait
La classification forehand/backhand (`vision/shots.py` `classify_forehand_backhand`) est rudimentaire (2/8 sur la GT). **Un doc R&D existe** (à lire et IMPLÉMENTER) : `docs/research/forehand-backhand.md` (branche `origin/docs/fhb-brainstorm` — `git show origin/docs/fhb-brainstorm:docs/research/forehand-backhand.md`). Il a MESURÉ : supprimer la correction de "facing" → **2/8 → 7/8** (quick-win), + un composite (ball_dx au contact + override de swing + abstention + handedness) → **8/8, confusion 0/0**. Implémente le composite.

## Le thermomètre (GT + scorer — mesure LIVE, zéro triche)
- GT : `tests/fixtures/shots/tennis_demo3.shots.json` (8 frappes avec label `type` forehand/backhand).
- Scorer : à construire (le doc le spécifie) — apparie frappe prédite↔GT (±N frames), puis **accuracy du label FH/BH** sur les appariées + confusion FH↔BH. Mesure sur le LIVE `_stats.json` (pas un cache).
- Baseline actuel : 2/8.

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-s5fhb feat/s5-forehand-backhand
```
Branche `feat/s5-forehand-backhand` (depuis `feat/accuracy-overhaul`). Édite SOUS le worktree. **Fichiers : `vision/shots.py` (classify_forehand_backhand + detect_hits si besoin) + un nouveau scorer `tools/event_eval/score_fhb.py` + un test.** NE touche PAS `vision/events.py`, `vision/pose.py`, `models/`.

## Pistes (du doc R&D + web)
1. **Quick-win mesuré** : supprimer la correction de "facing" (le doc dit qu'elle inverse la décision 6/8 fois) → 2/8→7/8. Vérifie-le par la mesure.
2. **Composite** : `ball_dx` au contact (signal primaire) + override de swing sur désaccord net + bande d'abstention près de l'axe du corps + prior de handedness (mode du côté forehand des coups d'un joueur). → 8/8 attendu.
3. **Far player** : maintenant que la far-pose est dense (84.9%), le `ball_dx`/poignet far devient exploitable → le FH/BH far devient mesurable. Exploite-le.
4. **Techno externe** (recherche web) : un classifieur de coups pose-based léger, si licence-clean et mieux mesuré.

## Contraintes dures
- **Cible** : FH/BH accuracy 2/8 → ≥7/8 (idéalement 8/8), confusion FH↔BH minimale, sur le LIVE.
- **NE régresse PAS** : `test_shots_regression` (si présent), `test_event_confusion_regression`, et les autres — verts.
- **Zéro hardcoding** (seuils = bande angulaire/géométrie, pas pixels demo3 ; le doc note des seuils felix-provisionnels → généralise) ; **zéro triche** ; anti-surapprentissage (8 frappes = peu — ne sur-tune pas).

## Boucle + livrable
Itère SANS plafond. CR `docs/research/s5-forehand-backhand-CR.md` : journal (accuracy FH/BH + confusion à chaque étape, sur le LIVE), le quick-win + composite, technos web évaluées, limites (8 frappes, handedness non validé), idée réutilisable. Commit + push. Chiffres LIVE mesurés.
