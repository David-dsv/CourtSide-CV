# PROMPT — FX : carte HUD par-échange + badge « ÉCHANGE N » (Backend dessin)

> **Session S4.** Effort recommandé : **high** (primitive de dessin isolée, pure cv2/PIL, testable seule ; pas de pipeline lourd).

## Contexte
Repo CourtSide-CV. Lire `CLAUDE.md`, `PROJET.md`, `docs/pm.md`, puis :
- `vision/fx.py` — tu le possèdes **en exclusivité**. C'est la lib de FX (cv2+numpy+PIL, pas de GPU). Primitives existantes utiles : `draw_text` (~332), `stat_card` (~533), `speed_gauge` (~461), `glow_marker` (~656), `shockwave` (~406).
- Le consommateur (NE le modifie pas, c'est S1) : `run_pipeline_8s.py:1248` appelle `fx.stat_card(out, margin, card_y, card_lines, title="Match", width=card_w)`.

## Le problème (décision user, ARRÊTÉE)
Le tableau bas-gauche doit afficher les stats **de l'échange en cours** (reset à chaque rally) et **écrire le tag de l'échange** (« ÉCHANGE N »). S1 calcule les valeurs et le numéro de rally ; **toi tu fournis la primitive de rendu** dédiée, plus jolie qu'un `stat_card` générique : en-tête « ÉCHANGE N », corps de stats, et (optionnel) ligne d'outcome (winner/forced_error/…).

## Worktree (OBLIGATOIRE — voir docs/pm.md)
```
git worktree add ../CourtSide-CV-fx feat/fx-rally-card
```
Branche : `feat/fx-rally-card` (depuis `feat/accuracy-overhaul`). Commit + push après chaque étape.

## Objectif — nouvelle primitive `fx.rally_card`
Ajouter **une nouvelle fonction** à `vision/fx.py` (sans modifier les signatures existantes — non-régression totale pour les autres sessions qui utilisent déjà `stat_card`/`speed_gauge`).

## Contrat d'API (à respecter EXACTEMENT — S1 code contre ça)
```python
def rally_card(frame, x, y, *, rally_index, lines, outcome=None,
               width=None, title="ÉCHANGE"):
    """
    Carte glassmorphism bas-gauche pour les stats de l'ÉCHANGE COURANT.
      - en-tête : f"{title} {rally_index}"  (ex. "ÉCHANGE 3")
      - lines : list[(label, value)]  (ex. [("Frappes","4"),("Rebonds","3"),("Peak","98 km/h")])
      - outcome : str | None  → si fourni, une ligne/pastille colorée
        (winner=vert, forced_error=orange, unforced_error=rouge, neutral=gris).
      - width : largeur en px (défaut: dérivée comme stat_card).
    Retourne la frame annotée (même convention que stat_card).
    """
```
- Réutilise le style glassmorphism de `stat_card` (fond translucide, coins arrondis, `draw_text` pour le texte net). Reste **cohérent visuellement** avec le reste du HUD.
- **N'altère pas** `stat_card` ni les autres fonctions : ajout pur.

## Étapes
1. Implémente `rally_card` en réutilisant les helpers existants (`draw_text`, le panneau translucide de `stat_card` — factorise un helper privé partagé si utile, **sans** changer la signature publique de `stat_card`).
2. En-tête bien lisible « ÉCHANGE N » (plus gros / accentué que les lignes de stats).
3. Mapping couleur outcome documenté (constante en haut du bloc).
4. Gère les cas : `lines` vide, `outcome=None`, `rally_index` grand (largeur stable).

## Validation + ré-itération (jusqu'à parfait)
- `python -m py_compile vision/fx.py`.
- **Test de rendu isolé** (`/tmp/test_rally_card.py`, ne pas committer) : frame noire 1920×1080, appelle `rally_card` avec rally_index=1/3/12, lines variées, outcome ∈ {None, "winner", "forced_error", "unforced_error"}. Sauve les PNG. Vérifie à l'œil : en-tête correct, reset visuel clair, pastille outcome de la bonne couleur, pas de débordement.
- Vérifie la **non-régression** : `stat_card`/`speed_gauge` inchangés (un rendu rapide de contrôle).
- Ré-itère jusqu'à un rendu net et premium.

## CR à la fin (OBLIGATOIRE)
- La signature finale de `fx.rally_card` (+ confirmation : aucune signature existante modifiée).
- Le mapping couleur outcome.
- Comment S1 doit l'appeler (exemple d'appel exact), pour remplacer `stat_card` dans le HUD bas-gauche.
- Joins/décris les PNG de test.
- Honnête : limites (texte très long, beaucoup de lignes).
- Résultats py_compile + test de rendu.
Pousse tout sur `feat/fx-rally-card`. NE touche QUE `vision/fx.py`.
