# PROMPT — S3 : génération de candidats d'événements (le plafond du recall) (Backend R&D+impl)

> **Session AUTONOME ULTRACODE.** Lance en `/ultracode`.
> ⛔ **NE POSE JAMAIS DE QUESTION. Décide, documente, livre.**
> ✅ **RECHERCHE WEB ENCOURAGÉE** : cherche les méthodes 2024-2026 de détection
> d'événements de tennis (bounce/hit detection, TrackNet+event, change-point
> detection, trajectory segmentation, physics-based apex/impact detection)…
> Évalue + implémente ce qui mesure mieux la GENERATION de candidats.

## Le problème (mesuré, GT humaine)
La classification rebond/frappe (`vision/events.py` `classify_events`) est BONNE (firewall 0/0), mais le **recall est plafonné par la GÉNÉRATION DE CANDIDATS** : un événement absent du pool ne peut pas être classé. Mesuré sur demo3 : bounce recall ~5/9 en live (les rebonds far-court apex + cold-start n'ont pas de candidat). Ton job = **faire entrer les vrais événements dans le pool** (`detect_turning_points` + `detect_sharp_turns` + `detect_wrist_hits`), sans inonder de faux positifs.

## Le thermomètre (GT + scorer — mesure LIVE, zéro triche)
- GT : `tests/fixtures/bounces/tennis_demo3.bounces.json` (9) + `tests/fixtures/shots/tennis_demo3.shots.json` (8).
- Cache scorer : `python tools/event_eval/run_demo3.py` → matrice confusion + F1.
- **LIVE scorer (la vérité)** : `python run_pipeline_8s.py tennis.mp4 -s 73 -d 13 --device cpu --dump-bounces /tmp/e.json` puis `python tools/event_eval/score_stats.py <_stats.json>`. ⚠️ **Le cache MENT** (4× prouvé : la piste balle live diffère) — mesure le recall+confusion sur **≥2 runs LIVE frais**, pas le cache.
- Le pool de candidats : combien des 17 GT (9 bounces + 8 shots) ont AU MOINS un candidat dans ±8f ? C'est ta métrique #1 (le plafond).

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-s3events feat/s3-event-recall
```
Branche `feat/s3-event-recall` (depuis `feat/accuracy-overhaul`). Édite SOUS le worktree. **Fichiers : `vision/events.py` (générateurs de candidats) + éventuellement `vision/bounce.py` (detect_bounces_robust).** NE touche PAS `models/`, `vision/pose.py`, `vision/shots.py` (detect_hits — sauf si tu améliores un générateur de candidats hit, alors documente).

## Pistes (mesurer, pas supposer)
1. **Le pool plafonne à combien ?** Mesure d'abord : sur 17 GT, combien ont un candidat ? Les manquants = trou de génération (pas de classification). detect_sharp_turns (angle vitesse ≥70°) récupérait les apex far — élargir/affiner sa fenêtre/seuil sur le plateau connu (60-75°).
2. **Cold-start** : les événements du début sont ratés (la balle démarre froid). Une fois S1 (balle) plus dense, re-mesure — mais toi, assure-toi que le générateur ne rate pas un événement présent dans la trajectoire.
3. **Techno externe** (recherche web) : un détecteur d'impact/bounce basé physique ou ML léger qui complète les turning-points.
4. **Sans casser le firewall 0/0** : tout nouveau candidat passe par l'arbitrage `classify_events` (firewall `hit_like ⇏ bounce`). Vérifie que la confusion reste 0/0 EN LIVE sur ≥2 runs.

## Contraintes dures
- **Cible** : pool ≥ 15/17 → idéalement 17/17, recall live qui monte, confusion **0/0 robuste sur ≥2 runs LIVE** (pas un run chanceux).
- **NE régresse PAS** : `test_event_confusion_regression`, `test_sharp_turns`, `test_bounce_regression`, `test_bounce_wasb_regression`, `test_vx_veto`, `test_far_coverage` — verts.
- **Zéro hardcoding / zéro triche** ; pas de seuil tuné sur demo3 ; mesure LIVE.

## Boucle + livrable
Itère SANS plafond. CR `docs/research/s3-event-recall-CR.md` : journal (pool recall + F1 + confusion sur ≥2 runs LIVE à chaque étape), technos web évaluées, gain, preuve firewall 0/0 robuste + felix intact, idée réutilisable. Commit + push. Chiffres LIVE.
