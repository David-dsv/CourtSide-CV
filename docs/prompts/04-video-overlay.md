# PROMPT — Overlay vidéo : shot≠bounce + HUD par-échange + radar 1 rebond (Backend)

> **Session S1 — la SEULE à toucher `run_pipeline_8s.py`.** Effort recommandé : **ultracode** (3 corrections couplées dans la boucle Pass 2, validation visuelle obligatoire, c'est le cœur du rendu).

## Contexte
Repo CourtSide-CV. Lire `CLAUDE.md`, `PROJET.md`, `docs/pm.md`, puis :
- `run_pipeline_8s.py` **Pass 2** uniquement (boucle d'annotation, lignes ~1080–1253).
- Les primitives que tu APPELLES (ne les modifie pas — d'autres sessions s'en occupent) :
  - `vision/fx.py` : `shockwave`, `glow_marker`, `draw_text`, `speed_gauge`, `stat_card`.
  - `vision/minimap.py` : `draw_minimap(...)` (appel actuel à `run_pipeline_8s.py:1213`).

Tu possèdes **`run_pipeline_8s.py` en exclusivité**. Aucune autre session ne le modifie.

## Les 3 problèmes (diagnostiqués)

### #1 — On confond shot et bounce dans la vidéo
- Bounces dessinés via `fx.shockwave` (anneau, couleur=depth red/cyan/green) + labels DEPTH/km-h/Q — `run_pipeline_8s.py:1137–1161`.
- Shots dessinés via `fx.glow_marker` (point, couleur=stroke cyan/orange) + tag FOREHAND/BACKHAND — `run_pipeline_8s.py:1164–1187`.
- **Problème user** : à l'usage on ne distingue PAS un rebond d'une frappe. Le code les croit distincts mais visuellement ça se confond (même fenêtre 0.5 s, deux glows colorés proches, labels qui se chevauchent). **Forme et sémantique doivent être sans ambiguïté.**

### #3 — Le stat card (bas-gauche) ne se réinitialise pas par échange + pas de tag d'échange
- HUD construit à `run_pipeline_8s.py:1222–1248`. `card_lines` = compteurs **cumulés depuis le début du clip** (`n_fh`, `n_bh`, `n_rally` = tous les shots/bounces `<= frame_idx`). Jamais remis à zéro.
- **Problème user** : les stats doivent **se réinitialiser à chaque nouvel échange (rally)**, et le **tag de l'échange en cours doit être écrit dans ce tableau** (ex. « ÉCHANGE 3 »).

### #4 — Le court radar affiche tous les rebonds ; il ne doit afficher QUE le dernier
- Pass 2 passe `bounces_so_far` = **tous** les rebonds `<= frame_idx` à `draw_minimap` (`run_pipeline_8s.py:1197–1198`).
- **Problème user** : sur le radar, **un seul rebond** doit être visible — celui qui vient d'être fait. Dès qu'un nouveau rebond arrive, il remplace l'ancien.

## Worktree (OBLIGATOIRE — voir docs/pm.md)
```
git worktree add ../CourtSide-CV-overlay feat/video-overlay
```
Branche : `feat/video-overlay` (depuis `feat/accuracy-overhaul`). Commit + push après chaque étape.

## Découpage des rallies (réutilise l'existant)
Les rallies sont déjà segmentés à `run_pipeline_8s.py:1286–1321` (gap > 2.5 s entre shots → nouveau rally ; chaque rally a `start_frame`, `end_frame`, `shot_frames`, `bounce_frames`, `n_shots`, et `outcome` via `vision/rally_outcome.py`). **MAIS** cette segmentation est calculée APRÈS la boucle Pass 2. Pour un HUD live par-échange, tu dois connaître, à chaque `frame_idx`, **dans quel rally on est**. Construis une structure `rally_at_frame[frame_idx] -> rally_index` (ou réutilise/avance la segmentation avant la boucle de rendu). Numérote les échanges à partir de 1.

## Objectif — overlay sans ambiguïté + HUD par-échange + radar 1 rebond

### #1 — Distinguer shot et bounce visuellement
1. Choisis deux langages visuels **orthogonaux et lisibles** :
   - **Bounce** = impact au sol → garde l'anneau `shockwave` (depth-color) mais ajoute une **icône/forme « impact »** non confondable (ex. cible/croix au centre) et garde le label depth.
   - **Shot** = frappe joueur → forme distincte (ex. arc/raquette ou anneau pointillé) + tag stroke. Couleur de stroke conservée mais forme ≠ bounce.
2. Évite la collision de labels : si un shot et un bounce coïncident temporellement/spatialement, décale les textes pour qu'ils ne se superposent pas.
3. Optionnel mais recommandé : une mini-légende discrète (un coin) « ● rebond / ◗ frappe » pour lever toute ambiguïté.
   Si tu as besoin d'une **nouvelle primitive de forme**, NE l'écris PAS dans `fx.py` (S4) — code-la localement OU demande-la en contrat à S4 (voir §Contrats). Préfère réutiliser `glow_marker`/`shockwave`/`draw_text` avec des paramètres différents.

### #3 — HUD par-échange + tag
1. Remplace les compteurs cumulés par des compteurs **du rally courant** : `n_fh`, `n_bh`, nombre de frappes, nombre de rebonds **depuis `rally.start_frame`** du rally en cours. Reset net au changement de rally.
2. Ajoute une **ligne/titre « ÉCHANGE N »** (N = index 1-based du rally courant) bien visible dans le card. Quand l'`outcome` du rally est connu (winner/forced_error/…), affiche-le aussi.
3. `Peak` : peak du rally courant (reset par échange) — pas le peak global.
4. Le rendu passe par `fx.stat_card(...)`. Si tu veux un **variant per-rally** plus joli (titre échange + outcome), c'est S4 qui le fournit (contrat ci-dessous) ; en attendant, fais-le avec `stat_card` + `draw_text` existants.

### #4 — Radar : un seul rebond
1. Au lieu de `bounces_so_far` (tous), passe à `draw_minimap` **uniquement le dernier rebond survenu** `<= frame_idx` (le plus récent par frame). Si aucun rebond encore, passe une liste vide.
2. S3 ajoute à `draw_minimap` un mode « single/latest bounce » (clean marker, pas de heatmap cumulée). Contrat ci-dessous. En attendant S3, passe déjà une liste à 1 élément — l'ancien `draw_minimap` dessinera juste ce point (et la heatmap dégénère à 1 point, acceptable en transition).

## Contrats d'API (ce que tu attends des autres sessions)
Documente ces attentes en haut de ton diff (commentaire) pour que l'intégration les vérifie :
- **S3 (minimap)** : `draw_minimap(..., bounces_img=<liste 0 ou 1 rebond>, single_bounce=True)` → dessine au plus un marqueur propre, sans heatmap cumulative. Si `single_bounce` absent, comportement legacy (liste = ce qu'on passe).
- **S4 (fx)** : `fx.rally_card(frame, x, y, *, rally_index, lines, outcome=None, width=...)` → card glassmorphism avec en-tête « ÉCHANGE N » + ligne outcome. Si non dispo, fallback sur `fx.stat_card` + un `draw_text` titre.
Code **défensivement** : `try`/`hasattr` pour utiliser le variant S3/S4 s'il existe, sinon fallback. Comme ça ta branche compile et tourne **seule**, et profite des autres à l'intégration.

## Validation + ré-itération (jusqu'à parfait)
- `python -m py_compile run_pipeline_8s.py`.
- **Run réel court** sur un segment qui contient ≥ 2 rallies, ex. :
  `python run_pipeline_8s.py tennis.mp4 -s 0 -d 30 --device cpu --match-mode -o /tmp/s1_overlay.mp4`
  (mps si la RAM le permet). Ouvre la vidéo : (a) rebond vs frappe **immédiatement distinguables** ; (b) le card affiche « ÉCHANGE N », se **remet à zéro** à chaque nouvel échange ; (c) le radar ne montre **qu'un seul rebond** à la fois.
- Ré-itère le rendu jusqu'à ce que les 3 points soient **visuellement parfaits**. Documente les frames-clés que tu as inspectées.

## CR à la fin (OBLIGATOIRE)
- Comment shot et bounce sont désormais distingués (formes/couleurs/labels) — avec captures ou n° de frames inspectées.
- Comment le HUD calcule le rally courant + reset + tag ÉCHANGE N (+ outcome).
- Comment le radar reçoit 1 seul rebond (+ quel contrat S3 tu utilises/attends).
- Les contrats EXACTS attendus de S3 (`single_bounce`) et S4 (`rally_card`) — signatures.
- Honnête : ce qui reste imparfait (collisions de labels résiduelles, rallies mal segmentés sur clips denses — gap 2.5 s connu).
- Résultats py_compile + run réel (chemin de la vidéo de test).
Pousse tout sur `feat/video-overlay`. Ne touche QUE `run_pipeline_8s.py`.
