# PROMPT — Brainstorm : distinguer REBOND et FRAPPE par la direction (recherche, PAS de code)

> **Session de RÉFLEXION pure.** Effort recommandé : **xhigh**.
> ⚠️ **Cette session NE CODE PAS.** Elle produit UNIQUEMENT un document de
> solutions (analyse + propositions chiffrées + recommandation). Aucune édition
> de `vision/`, `run_pipeline_8s.py`, etc. Le livrable est un `.md` de synthèse.

## Contexte
Repo CourtSide-CV. Lire `CLAUDE.md`, `PROJET.md`, `docs/pm.md`, puis le code de
détection (lecture seule) :
- `vision/bounce.py` — `detect_bounces_from_trajectory` (~l.112) + `detect_bounces_robust` (~l.484).
- `vision/shots.py` — `detect_hits` (~l.200), `_wrist_speed_series` (~l.128), `_fwhm_at` (~l.184).
- `run_pipeline_8s.py` — appels : bounce (~l.1098), shot (~l.1273), Kalman vx/vy (~l.1017).
- `vision/player_track.py` — `TwoPlayerTracker` (centroïdes/pieds P1/P2, net divider).

## Le problème (diagnostiqué, validé par l'utilisateur)
À l'usage, la vidéo annotée **confond encore frappe et rebond**. Le fix #1 (déjà
mergé) ne traitait que **l'affichage** (formes orthogonales). Le vrai problème
est **en amont, dans la DÉTECTION**.

**État réel du code (reconnaissance faite) :**
- **Rebond** = maximum local de `y` (image) + fit balistique descente/montée +
  garde d'énergie `|pente_montée| ≤ 1.8 × |pente_descente|` (un hit injecte de
  l'énergie → rebond trop raide → rejeté). Source = **trajectoire de la balle**.
- **Frappe** = pic de **vitesse du poignet** (pose), filtré par FWHM (4-8 frames
  = swing vs 1-2 = jitter) + proximité balle↔joueur. Source = **squelette**.
- Les deux signaux sont **orthogonaux** ; seule protection croisée = une garde
  temporelle `±0.10 s` (`bounce_guard`) qui supprime un hit trop proche d'un
  rebond.
- **AUCUNE logique de changement de direction** n'existe. Les vecteurs vitesse
  `(vx, vy)` du Kalman sont calculés puis **jetés** (seule la norme = vitesse
  km/h est gardée, `run_pipeline_8s.py:1017`). L'ancienne règle `vy` down→up a
  été abandonnée (lissée par le spline, fragile sur broadcast).

**L'intuition de l'utilisateur (à exploiter) :**
> « Tu ne peux pas définir un rebond comme un tir si globalement la direction ne
> change pas. »

Physique : **un REBOND est une réflexion VERTICALE qui PRÉSERVE le sens du
déplacement horizontal** (la balle continue globalement dans la même direction
X) ; **une FRAPPE est une DÉVIATION/INVERSION** de la trajectoire (le sens
horizontal s'inverse ou dévie fortement — le joueur renvoie la balle). Le code
actuel n'a aucune notion de « le sens de déplacement a-t-il changé ? ».

## Objectif de la session — TROUVER des solutions, pas les coder
Produire `docs/research/bounce-vs-shot-direction.md` (créer le dossier
`docs/research/`) qui répond à : **comment utiliser le changement (ou la
préservation) de la direction de déplacement de la balle pour distinguer
fiablement rebond et frappe, et corriger/compléter la détection existante ?**

## Axes à explorer (au minimum)
1. **Signal de direction** : comment définir « la direction globale » de la balle
   avant/après un événement de façon robuste au bruit et au lissage spline ?
   Pistes à évaluer (avec leurs limites) :
   - angle entre vecteurs vitesse moyens pré/post (dot-product / `atan2`),
     fenêtres de N frames de chaque côté ;
   - signe de `vx` (composante horizontale) avant vs après : un rebond préserve
     `sign(vx)`, une frappe l'inverse souvent ;
   - réflexion attendue : un rebond reflète `vy` (inverse le vertical) en gardant
     `vx` ; une frappe change les DEUX ;
   - sur trajectoire éparse (YOLO+Kalman) vs dense (WASB) : que reste-t-il
     exploitable ? (le spline tue les réversions — faut-il mesurer la direction
     AVANT lissage, sur les détections brutes ?)
2. **Fusion des deux détecteurs** : aujourd'hui bounce (trajectoire) et shot
   (poignet) sont indépendants. Faut-il :
   - une **étape d'arbitrage** qui, pour chaque événement candidat, décide
     bounce/shot/aucun via la direction + la proximité joueur + le filet ?
   - une **table de désambiguïsation** quand les deux fireraient au même endroit ?
   - garder les deux signaux mais ajouter la direction comme **gate/tie-breaker** ?
3. **Géométrie de court** : exploiter le filet (`net_y`), les baselines, la
   position des joueurs (`TwoPlayerTracker`) et l'homographie quand dispo :
   - une frappe arrive **près d'un joueur** ; un rebond arrive **au sol, loin
     des deux** → la proximité joueur est-elle un meilleur discriminant que la
     direction, ou complémentaire ?
   - en mètres (homographie) : un rebond a `y≈0` au sol projeté ; une frappe est
     à hauteur de raquette. Exploitable ?
4. **Cas durs** : volée (frappe SANS rebond préalable), amorti (rebond très
   mou), balle qui touche le filet, smash (frappe quasi verticale), passages
   où la balle est occultée au contact (broadcast). Pour chacun : la solution
   proposée tient-elle ?
5. **Mesurabilité** : comment VALIDER objectivement une future implémentation ?
   Il existe déjà `tests/fixtures/bounces/felix.bounces.json` + l'évaluateur
   `tools/bounce_eval/`. Proposer la GT et la métrique pour les FRAPPES (et la
   confusion bounce↔shot) — sans la coder, juste spécifier le harnais.

## Contraintes de design (rappel CLAUDE.md)
- **Zéro hardcoding** : tout seuil = ratio de fps / dimensions / géométrie
  détectée. Pas de magie tirée d'une vidéo.
- **Généralisation > précision** : robuste sur n'importe quelle vidéo amateur /
  broadcast, surface, angle.
- Respecter ce qui marche déjà (bounce F1≈0.80 sur felix — ne pas casser).

## Forme du livrable `docs/research/bounce-vs-shot-direction.md`
1. **Diagnostic** : reformuler le vrai problème (détection, pas affichage) avec
   les `file:line` du code actuel.
2. **Inventaire des signaux disponibles** au moment de la détection (vx/vy bruts,
   trajectoire lissée vs brute, poses, filet, homographie) + lesquels sont
   fiables sur broadcast vs amateur.
3. **3 à 5 approches** concrètes, chacune avec : principe, pseudo-formule
   (pas de code exécutable — du pseudocode/maths au plus), robustesse attendue,
   coût, cas durs gérés/non gérés.
4. **Recommandation classée** (du PM-ready au spéculatif) + le découpage en
   chantiers implémentables (que des *futures* sessions coderont).
5. **Plan de validation** (GT frappes + métrique confusion) à mettre en place.
6. **Risques / ce qui pourrait ne pas marcher**.

## CR à la fin (OBLIGATOIRE)
- Lien vers `docs/research/bounce-vs-shot-direction.md`.
- Résumé en 5 lignes : la (les) approche(s) recommandée(s) + pourquoi.
- Ce qui reste incertain et demanderait un prototype pour trancher.
- **Rappel : aucune ligne de code de production écrite.** Si la session a fait
  des mesures exploratoires (ex. tracer des angles sur la trajectoire d'un
  clip), elles vont dans des scripts jetables `/tmp` NON committés, et leurs
  résultats CHIFFRÉS dans le doc.

> Pas de worktree nécessaire (aucune écriture de code de prod) — la session écrit
> seulement `docs/research/bounce-vs-shot-direction.md` (+ commit/push de ce
> seul fichier sur une branche `docs/bounce-vs-shot-brainstorm`).
