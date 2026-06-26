# PROMPT — R&D : homographie oblique AUTOMATIQUE (recherche, PAS de code) — SESSION 4/5

> **Session AUTONOME R&D — RECHERCHE SEULEMENT.** Lance en `/ultracode`.
> ⛔ **NE POSE JAMAIS DE QUESTION à l'utilisateur. Aucun outil de clarification.**
> Décide seule, documente, ne bloque jamais. Une question = run raté.
> **AUCUN code de production** — seulement un doc `.md`. Probes jetables `/tmp` OK
> (supprimées), avec leurs chiffres mesurés dans le doc.

## Contexte
Repo CourtSide-CV. Lire `CLAUDE.md`, `PROJET.md`, `docs/pm.md`, et les mémoires `courtside-sota-2026-06`, `courtside-felix-grazing-angle`, `courtside-product-vision-2026-06` (la licence yastrebksv = bloqueur SaaS).

## Le problème (le grand bloqueur SaaS)
La calibration court métrique marche en **semi-auto** (`models/court_calibrator.py` + `tools/calibrate_court.py` : 4 coins cliqués → homographie). Mais l'**AUTOMATIQUE** (sans clic) manque : le modèle court-keypoint (`models/court_detector.py`) fire sur trop peu de points sur les angles amateurs/bas (felix : 3/14). C'est ce qui empêche vitesse/profondeur métriques **sans intervention** — donc le produit SaaS amateur « film sur trépied, zéro clic ».

## Objectif — TROUVER comment auto-calibrer sans clic
Produire `docs/research/auto-homography-oblique.md` qui évalue concrètement :
1. **Voie no-GPU (interim)** : détecteur de lignes classique (HoughLinesP) + **RANSAC** pour fitter les lignes de court, avec **warp-back self-scoring** (re-projeter le court modèle ITF via l'homographie candidate et scorer l'alignement aux lignes détectées) pour (a) auto-seeder les 4 coins et (b) gater la confiance. Quels taux de réussite plausibles, quels cas d'échec (grazing felix) ?
2. **Voie GPU-gated (pilote)** : RF-DETR keypoint (Apache-2.0, skeletons arbitraires) fine-tuné sur un dataset de courts obliques labellisés. Coût (cloud GPU + data + smoke-test MPS) ?
3. **Hybride** : classique pour seeder + raffiner, modèle pour les cas durs.
4. **Licence** : éviter yastrebksv (no LICENSE, bloqueur SaaS) — quelles alternatives Apache/MIT ?

## Forme du doc
Diagnostic (file:line du chemin actuel) ; inventaire des signaux (lignes, coins, net posts, zones) ; 3-5 approches avec pseudo-maths + probes mesurées quand pas cher (ex. compter les lignes HoughLinesP exploitables sur felix/tennis) ; recommandation classée (no-GPU d'abord) + découpage en chantiers pour de futures sessions de code ; plan de validation (une GT de coins court + métrique de re-projection) ; risques (grazing, licence, généralisation).

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-homog docs/auto-homography-brainstorm
```
Branche `docs/auto-homography-brainstorm` (depuis `feat/accuracy-overhaul`). Édite SOUS `/Users/vuong/Documents/CourtSide-CV-homog/...`. Écris **uniquement** `docs/research/auto-homography-oblique.md`.

## Livrable — CR (dans le doc + résumé)
Le doc + un résumé 5 lignes (approche recommandée + pourquoi + effort estimé + risques ouverts). Commit + push. **Aucune ligne de code de prod.**
