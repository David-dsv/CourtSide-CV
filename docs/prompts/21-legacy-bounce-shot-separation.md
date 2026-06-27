# PROMPT — Confusion rebond/frappe S3 : chemin LEGACY par défaut (Backend) — SESSION 3/3

> **Session AUTONOME ULTRACODE.** Lance en `/ultracode`.
> ⛔ **NE POSE JAMAIS DE QUESTION à l'utilisateur. Aucun outil de clarification.**
> Décide seule, documente, ne bloque jamais. Une question = run raté.

## Contexte — pourquoi cette session
Repo CourtSide-CV. Lire `CLAUDE.md`, `PROJET.md`, `docs/pm.md`, et les mémoires `courtside-bounce-vs-shot-direction`, `courtside-bounce-broadcast-multifactor`, `courtside-far-player-selection-not-detection`.

**Le chemin par DÉFAUT (legacy, sans `--event-methodo`) est ce que l'utilisateur voit dans la vidéo**, et il confond massivement rebonds et frappes : mesuré sur demo3 régénéré = **2 bounces / 13 shots** prédits vs GT **9 bounces / 8 shots** (7 rebonds ratés, 5 frappes en trop). Deux autres sessions travaillent sur la méthodo `--event-methodo` ; **toi, tu améliores le chemin LEGACY** (le défaut), de façon indépendante et complémentaire.

Le chemin legacy = `detect_bounces_robust` + `vx_flip_veto` (rebonds, `vision/bounce.py`) ET `detect_hits` (frappes par pic de poignet, `vision/shots.py`), arbitrés seulement par une garde temporelle `bounce_guard`. C'est cette séparation faible qui produit la confusion.

## CE QUI A CHANGÉ (un levier nouveau)
La pose du joueur far vient de passer **0.9% → 84.9%** (far-select mergé). `detect_hits` peut maintenant voir le poignet far → moins de frappes far ratées ET la **proximité balle↔poignet** devient un discriminant fiable des deux côtés (avant, le far était aveugle).

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-legacysep feat/legacy-bounce-shot-sep
```
Branche `feat/legacy-bounce-shot-sep` (depuis `feat/accuracy-overhaul`). Commit/push par itération.
⚠️ `courtside-edit-targets-main-tree` : édite SOUS `/Users/vuong/Documents/CourtSide-CV-legacysep/...`. Fichiers : `vision/bounce.py` (rebonds + veto) ET `vision/shots.py` (frappes + proximité). **NE touche PAS `vision/events.py`** (c'est la méthodo des autres sessions).

## Le diagnostic à exploiter
La vraie séparation physique (mémoire `courtside-bounce-vs-shot-direction`, déjà mesurée et mergée) :
- **REBOND** : préserve le sens horizontal (vx-signe) + flip vy ; loin des deux joueurs au sol.
- **FRAPPE** : inverse vx ; PRÈS d'un joueur (le poignet, maintenant visible far ET near).
Le `vx_flip_veto` actuel ne fait que REJETER des FP de frappe émis comme rebonds. Il ne récupère PAS les rebonds ratés (7/9 sur ce clip !) ni n'arbitre proprement. **Le vrai problème legacy = le détecteur de rebonds rate 7/9 rebonds** (cf. `courtside-ball-density-apex-bounce` : les apex far n'ont pas d'extremum y).

## Pistes (mesure d'abord, ne code pas aveuglément)
1. **Rebonds ratés (7/9)** : pourquoi `detect_bounces_robust` n'en trouve que 2 ? Densité balle ? Apex sans extremum (la solution `detect_sharp_turns` existe dans `events.py` — peux-tu en porter le PRINCIPE de l'angle de virage dans le chemin legacy de `bounce.py`, SANS dépendre de events.py) ?
2. **Frappes en trop (5)** : `detect_hits` sur-déclenche. Avec la pose far dense, la proximité balle↔poignet rejette-t-elle les faux contacts ?
3. **Arbitrage** : remplacer le `bounce_guard` aveugle par un arbitrage direction+proximité (un event près d'un joueur avec vx-flip = frappe ; loin + vx-préservé = rebond).

## Le benchmark + l'évaluateur
- GT : `tests/fixtures/bounces/tennis_demo3.bounces.json` + `tests/fixtures/shots/tennis_demo3.shots.json`.
- Mesure end-to-end : `python run_pipeline_8s.py tennis.mp4 -s 73 -d 13 --device cpu --dump-bounces /tmp/leg.json` puis score le `_stats.json` (bounces + shots) vs GT via `tools/event_eval/`. **Mesure le LIVE, pas un cache.**

## Contraintes dures
- **Cible** : sur le `_stats.json` legacy de PROD, bounce recall passe de 2/9 à nettement mieux, et les frappes en trop chutent — la confusion VISIBLE dans la vidéo diminue franchement.
- **NE régresse PAS** : `test_bounce_regression` (felix Kalman ≥0.72), `test_bounce_wasb_regression` (≥0.80), `test_vx_veto` (le veto felix ≥0.90), `test_event_confusion_regression`, `test_sharp_turns`, `test_far_coverage` — verts à chaque itération. felix est le garde-fou.
- Zéro hardcoding ; anti-surapprentissage.

## Livrable — CR (OBLIGATOIRE)
`docs/research/legacy-bounce-shot-sep-CR.md` : journal d'itérations (bounce/hit recall + confusion sur le `_stats.json` LIVE), ce qui a récupéré les rebonds ratés + tué les frappes en trop, preuve felix intact, l'idée la plus réutilisable. Commit + push. VRAIS chiffres mesurés.
