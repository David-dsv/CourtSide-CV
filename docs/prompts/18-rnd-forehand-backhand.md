# PROMPT — R&D : classification forehand/backhand fiable (recherche, PAS de code) — SESSION 5/5

> **Session AUTONOME R&D — RECHERCHE SEULEMENT.** Lance en `/ultracode`.
> ⛔ **NE POSE JAMAIS DE QUESTION à l'utilisateur. Aucun outil de clarification.**
> Décide seule, documente, ne bloque jamais. Une question = run raté.
> **AUCUN code de production** — seulement un doc `.md`. Probes jetables `/tmp` OK
> (supprimées), chiffres mesurés dans le doc.

## Contexte
Repo CourtSide-CV. Lire `CLAUDE.md`, `PROJET.md`, `docs/pm.md`, et le code : `vision/shots.py` (`detect_hits`, `classify_forehand_backhand` — le détecteur géométrique rudimentaire actuel), `vision/pose.py` (poses COCO 17 : épaules 5/6, coudes 7/8, poignets 9/10, hanches 11/12).

## Le problème
La classification **forehand/backhand** est rudimentaire (géométrique, peu fiable). Pour viser le SOTA, il faut une méthode robuste depuis la pose : côté du poignet vs épaule/hanche, orientation du corps, trajectoire du swing, **handedness** (droitier/gaucher), et le cas du **petit joueur far** (squelette bruité).

> Note liée : une session sœur travaille à fiabiliser la sélection du joueur far (`vision/pose.py`) — ce qui débloquera les **poignets far**. Ton doc doit supposer que les poignets far deviendront disponibles et en tirer parti.

## Le benchmark existe (l'exploiter)
`tests/fixtures/shots/tennis_demo3.shots.json` : **8 frappes GT avec label `type` (forehand/backhand)**. C'est une vraie GT pour mesurer l'accuracy FH/BH. Spécifie (sans coder) le harnais : matcher frappe prédite↔GT (tol ±N frames), puis accuracy du label sur les frappes appariées.

## Objectif — TROUVER une méthode FH/BH robuste
Produire `docs/research/forehand-backhand.md` :
1. **Inventaire des signaux pose** : poignet actif vs centre du corps, côté (gauche/droite de la ligne épaules-hanches), rotation du tronc, position du poignet relative à l'épaule au contact. Lesquels sont fiables near vs far ?
2. **Handedness** : comment l'inférer (la plupart des coups d'un joueur révèlent sa main dominante) sans la supposer ?
3. **3-5 approches** (pure géométrie pose améliorée ; règle au contact ; fenêtre temporelle du swing ; éventuellement un petit classifieur) avec pseudo-maths + robustesse near/far + cas durs (revers à deux mains, slice, contact occulté).
4. **Plan de validation** contre la GT 8-frappes (accuracy, et la confusion FH↔BH).
5. Recommandation classée + découpage en chantiers pour de futures sessions de code.

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-fhb docs/fhb-brainstorm
```
Branche `docs/fhb-brainstorm` (depuis `feat/accuracy-overhaul`). Édite SOUS `/Users/vuong/Documents/CourtSide-CV-fhb/...`. Écris **uniquement** `docs/research/forehand-backhand.md`.

## Livrable — CR (dans le doc + résumé)
Le doc + résumé 5 lignes (approche recommandée + pourquoi + effort + risques). Commit + push. **Aucune ligne de code de prod.**
