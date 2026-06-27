# PROMPT — Confusion rebond/frappe S1 : re-valider + activer la méthodo en PROD (Backend) — SESSION 1/3

> **Session AUTONOME ULTRACODE.** Lance en `/ultracode`.
> ⛔ **NE POSE JAMAIS DE QUESTION à l'utilisateur. Aucun outil de clarification.**
> Décide seule, documente, ne bloque jamais. Une question = run raté.

## Contexte — le problème (constaté par l'utilisateur sur la vraie vidéo)
Repo CourtSide-CV. Lire `CLAUDE.md`, `PROJET.md`, `docs/pm.md`, et les mémoires `courtside-event-methodo-prod-gap`, `courtside-bounce-shot-methodo`, `courtside-ball-density-apex-bounce`, `courtside-far-player-selection-not-detection`.

**La plupart des REBONDS sont émis comme des FRAPPES dans la vidéo.** Mesuré sur le clip demo3 régénéré (chemin LEGACY par défaut) : **2 bounces / 13 shots** prédits vs GT **9 bounces / 8 shots** → 7 rebonds ratés, 5 frappes en trop.

**La solution existe déjà** : `vision/events.py` `classify_events` (méthodo classify+alternance+firewall+sharp-turns) donne **confusion 0/0, bounce F1 0.842** sur le cache. Elle est câblée derrière `--event-methodo` (défaut OFF). **Elle échouait en PROD (confusion 4/4) UNIQUEMENT parce que la pose du joueur far était à 23%** (le firewall a besoin de la pose far pour marquer les frappes far comme `hit_like`).

## CE QUI A CHANGÉ (pourquoi maintenant)
**La pose far vient d'être corrigée : 0.9% → 84.9%** (far-select mergé). Le firewall n'est PLUS affamé. **Ton job n°1 : re-mesurer si `--event-methodo` reproduit enfin le 0/0 en PROD**, et si oui, le rendre le chemin par défaut.

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-methodoprod feat/event-methodo-prod
```
Branche `feat/event-methodo-prod` (depuis `feat/accuracy-overhaul`). Commit/push par itération.
⚠️ `courtside-edit-targets-main-tree` : édite SOUS `/Users/vuong/Documents/CourtSide-CV-methodoprod/...`. Fichiers : `run_pipeline_8s.py` (le flag/défaut) + `vision/events.py` SI un ajustement firewall est nécessaire pour la prod.

## Étapes
1. **Mesure la prod ACTUELLE** end-to-end avec la pose far dense :
   `python run_pipeline_8s.py tennis.mp4 -s 73 -d 13 --device cpu --event-methodo --dump-bounces /tmp/m.json`
   Score le `_stats.json` produit (bounces + shots) contre la GT (`tests/fixtures/bounces/tennis_demo3.bounces.json` + `tests/fixtures/shots/tennis_demo3.shots.json`) via la matrice de confusion de `tools/event_eval/`. **Note confusion_H→B, confusion_B→H, bounce F1, hit F1 RÉELS en prod.**
2. **Compare au cache** (`python tools/event_eval/run_demo3.py` → méthodo 0/0, F1 0.842). Le gap résiduel prod-vs-cache, s'il reste, vient d'où ? (mesure la couverture pose far live du run : doit être ~84% maintenant, plus 23%).
3. **Si la prod reproduit ~0/0** : propose de passer `--event-methodo` ON par défaut (le PM tranchera, mais documente la reco + garde un `--no-event-methodo` pour le legacy). Vérifie que felix end-to-end ne casse pas (felix pauvre en events → la méthodo doit dégrader proprement, cf. CR existant : felix ON = 0 bounce/0 hit, parité OFF).
4. **Si un gap résiduel demeure** : diagnostique (pas un fix aveugle) — est-ce la pose far encore insuffisante à certains contacts ? la densité balle ? Documente et, si c'est dans ton scope (firewall), corrige en mesurant.

## Contraintes dures
- **Cible** : la confusion rebond/frappe DISPARAÎT dans le `_stats.json` de PROD (pas juste le cache). confusion_H→B = 0, bounce recall nettement meilleur que les 2/9 legacy actuels.
- **NE régresse PAS** : `test_bounce_regression` ≥0.72, `test_bounce_wasb_regression` ≥0.80, `test_vx_veto`, `test_event_confusion_regression`, `test_sharp_turns`, `test_far_coverage` — verts.
- **Mesure sur du LIVE, jamais seulement le cache** (la leçon `courtside-event-methodo-prod-gap`).
- Zéro hardcoding.

## Livrable — CR (OBLIGATOIRE)
`docs/research/event-methodo-prod-CR.md` : la confusion PROD avant/après (chiffres réels du `_stats.json`), reproduit-elle le 0/0 du cache maintenant que la pose far est dense ? reco sur le défaut du flag, preuve felix non régressé. Commit + push. VRAIS chiffres mesurés.
