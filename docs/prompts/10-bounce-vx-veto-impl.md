# PROMPT — Impl : veto vx-flip rebond≠frappe (A3⊕A4 + CHANTIER 0) (Backend)

> **Session d'IMPLÉMENTATION.** Effort recommandé : **ultracode** (touche la
> couche de DÉTECTION — un bug ici dégrade la F1 rebond ; mesure obligatoire,
> ré-itération jusqu'à preuve chiffrée). **UNE seule session** : CHANTIER 0 + 1
> + 2 touchent `run_pipeline_8s.py` ET `vision/bounce.py`, fichiers couplés.

## Contexte — LIRE D'ABORD le doc de recherche
La conception est **déjà faite et mesurée** :
**`docs/research/bounce-vs-shot-direction.md`** (sur `feat/accuracy-overhaul`).
Lire **en entier** avant de coder. Lire aussi `CLAUDE.md`, `PROJET.md`, `docs/pm.md`.

**Résumé de ce qui est prouvé (reproduit sur la production, pas théorique) :**
- Le bug est en DÉTECTION : `detect_bounces_robust` émet de vraies frappes comme rebonds (`f820`/`f1609` = réversions horizontales ~180°).
- Physique mesurée aux 16 rebonds GT felix : **vx-signe PRÉSERVÉ 15/15**, vy-FLIP 15/15 (1 occlus `f424`). Une frappe INVERSE vx ; un rebond le préserve.
- Veto vx-flip one-directional → **bounce F1 0.865→0.914 en natif 1080p, 0 vrai rebond perdu** (tue `f820`/`f1609`).

## Décision PM (ARRÊTÉE) — implémenter A3 ⊕ A4
**Veto vx-flip ONE-DIRECTIONAL + deadband RELATIF (échelle-libre) + CONFIRM exige aussi vy-flip.** *One-directional* = on ne REJETTE un candidat rebond que si vx s'inverse *de façon mesurable et confiante* ; **sinon on S'ABSTIENT (on garde le candidat)** → le veto **ne peut qu'ajouter de la précision, jamais retirer un vrai rebond**. C'est l'invariant de sécurité central.

## Worktree (OBLIGATOIRE — voir docs/pm.md)
```
git worktree add ../CourtSide-CV-vxveto feat/bounce-vx-veto
```
Branche : `feat/bounce-vx-veto` (depuis `feat/accuracy-overhaul`). Commit + push après chaque chantier.
⚠️ **Mémoire [[courtside-edit-targets-main-tree]]** : dans un worktree, Edit/Write vers le chemin canonique tape le MAIN tree. **Édite toujours sous `/Users/vuong/Documents/CourtSide-CV-vxveto/...`.**

## L'invariant BLOQUANT (le plus important)
Le veto lit la direction sur les **centres tracker BRUTS pré-spline AVEC un masque `is_real`** — **JAMAIS** sur `all_ball_centers`/`re_smoothed` (post-spline, qui dégrade le signal 15/15→14/15 et lit à travers les trous comblés). **Tant que le masque réel/interpolé n'est pas threadé, ne rien merger.**

### 🎯 Simplification CLÉ (vérifiée sur source — ne pas sur-architecturer)
`raw_ball_centers` **EXISTE DÉJÀ** et est pré-spline :
- WASB : `run_pipeline_8s.py:860` (`raw_ball_centers = tracker.track_ball(...)`).
- Kalman : `run_pipeline_8s.py:1022` (`raw_ball_centers.append(ball_center)`).
- Le spline est `run_pipeline_8s.py:1026` (`all_ball_centers = smooth_ball_trajectory(raw_ball_centers, ...)`).
- ⚠️ **Piège** : à `run_pipeline_8s.py:1101`, `detect_bounces_robust` **réassigne `all_ball_centers`** (= son `re_smoothed`). Le veto doit utiliser **`raw_ball_centers`** (capturé AVANT toute réassignation), pas `all_ball_centers`.

Donc **CHANTIER 0 ≈ utiliser la variable qui existe** : `is_real[i] = (raw_ball_centers[i] is not None)`. Pas besoin de reconstruire la sortie tracker.

## CHANTIERS (dans l'ordre — chacun commit+push+mesure)

### CHANTIER 0 — masque pré-spline (pré-requis)
- Construire `is_real = [c is not None for c in raw_ball_centers]` au point où `raw_ball_centers` est complet (après l.1022 / l.867), AVANT le spline l.1026.
- Garder `raw_ball_centers` disponible au site d'appel rebond (l.1100). Il l'est déjà — juste ne pas l'écraser.

### CHANTIER 1 — helper vitesse signée + veto (cœur A3/A4)
Dans `vision/bounce.py`, nouvelle fonction PURE testable, ex. :
```
def vx_flip_veto(bounce_events, raw_centers, is_real, fps, frame_width):
    # pour chaque (frame, x, y) candidat :
    half        = round(fps * 0.12)                          # 0.08-0.12 s = 15/15 stable ; 0.20 s dégrade
    speed_scale = median(|x[i+1]-x[i]| sur paires consécutives RÉELLES dans [f-half..f+half])
    eps         = max(0.15 * speed_scale, half * 1e-3 * (60/fps))   # deadband RELATIF (échelle-libre)
    # ⚠️ 0.15 et 1e-3 = felix-provisionnels ; documenter, à re-dériver cross-clip (CHANTIER 5).
    vx_pre  = polyfit pente x(t) sur ≤half points RÉELS (is_real) dans [f-half..f]   (≥2 requis)
    vx_post = polyfit pente x(t) sur ≤half points RÉELS dans [f..f+half]             (≥2 requis)
    si vx_pre/vx_post None             -> ABSTAIN (garde, tag "unconfirmed")   # f424
    sinon si |vx_pre|≤eps ou |vx_post|≤eps -> ABSTAIN (horizontal quasi-nul)   # drop-shot lent
    sinon si sign(vx_pre)==sign(vx_post)   -> CONFIRM (garde, tag "metric")
    sinon                                  -> REJECT (frappe)                  # f820/f1609
    # greffe (b) : au CONFIRM, exiger aussi vy-flip ; sinon -> garde mais tag "unconfirmed"
    #   (NE retire PAS → ne touche pas la recall). vy-flip = sign(vy_pre)>0 et sign(vy_post)<0 (repères image).
```
- Le veto **filtre la liste** de `detect_bounces_robust` (post-filtre), il ne ré-écrit pas le détecteur.
- **ABSTAIN dès qu'une fenêtre traverse un gap interpolé** > `round(fps*0.05)` (ne lis jamais à travers un trou).
- Câbler dans `run_pipeline_8s.py` **juste après** la ligne 1101 (chemin WASB uniquement) en passant `raw_ball_centers` + `is_real`.

### CHANTIER 3 — chemin Kalman = SKIP explicite
`detect_bounces_from_trajectory` (l.1104) n'expose pas de trajectoire+masque → **NE PAS appliquer le veto** sur ce chemin (vy-flip recall Kalman = 62-73 % → risque de descendre sous 0.72). Documenter en commentaire comme dette. (Le veto ne tourne QUE sur le chemin WASB.)

### CHANTIER 2 — tags de confiance UI (greffe légère)
Propager `direction_state ∈ {metric, unconfirmed}` sur chaque bounce dans `_stats.json` (présentation seule, aucun risque détection). Ne PAS modifier l'overlay/le front (autres scopes) — juste émettre le champ.

### (DIFFÉRÉ — ne PAS faire) CHANTIER 4 (arbitre hit↔bounce) — bloqué sur GT. CHANTIER 5 cross-clip — hors scope ici.

## Validation + ré-itération (jusqu'à PREUVE chiffrée)
- `python -m py_compile vision/bounce.py run_pipeline_8s.py`.
- **Test unitaire** `tests/test_vx_veto.py` (assert, sans vidéo/GPU) : trajectoires synthétiques — vx-préservé+vy-flip → CONFIRM ; vx-flip → REJECT ; épars (<2 pts) → ABSTAIN ; gap interpolé → ABSTAIN.
- **Régression rebond (NE PAS CASSER)** : `python tests/test_bounce_regression.py` doit rester **≥ 0.72** (chemin Kalman, inchangé par construction).
- **Preuve du gain** sur le cache WASB felix (`tests/fixtures/cache/felix_wasb_centers.json`) : reproduire `detect_bounces_robust` → veto → score vs `tests/fixtures/bounces/felix.bounces.json` (matcher greedy 1-1, tol `round(0.15*fps)=9`). **Cible : F1 monte (~0.865→0.914 en natif 1080p), `f820`/`f1609` tués, 0 TP perdu.** ⚠️ **Mesure en 1080p natif** (la liste de candidats dépend de la résolution — voir doc §7).
- Ré-itère jusqu'à : gain prouvé chiffré + 0 TP perdu + régression Kalman intacte.

## CR à la fin (OBLIGATOIRE)
- Le résultat CHIFFRÉ : F1 rebond avant/après veto sur felix WASB natif (+ FP tués, TP perdus = 0 attendu).
- Confirmation que le veto lit `raw_ball_centers` pré-spline + `is_real` (jamais le spline), et qu'il S'ABSTIENT proprement (épars/occlus/gap).
- Confirmation régression Kalman ≥ 0.72 intacte + le chemin Kalman SKIP le veto.
- Le champ `direction_state` ajouté au `_stats.json`.
- Honnête : angles morts NON traités (down-the-line vx-préservé, net-cord-avant), gain dense-seulement, constantes 0.15/1e-3 felix-provisionnelles à re-dériver.
- py_compile + test_vx_veto + test_bounce_regression : résultats exacts.
Pousse tout sur `feat/bounce-vx-veto`. Touche UNIQUEMENT `vision/bounce.py`, `run_pipeline_8s.py`, `tests/test_vx_veto.py`.
