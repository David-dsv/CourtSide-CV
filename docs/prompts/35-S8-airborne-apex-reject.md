# PROMPT — S8 : rejeter les APEX EN VOL (le fantôme de rebond en haut / "deep out compté") — Backend

> **Session AUTONOME ULTRACODE.** Lance en `/ultracode`.
> ⛔ **NE POSE JAMAIS DE QUESTION. Aucun outil de clarification.** Décide, documente, livre.

## Contexte
Repo CourtSide-CV. Lire `CLAUDE.md`, `docs/pm.md`, le CR
`docs/research/event-rootcause-2026-06-28-CR.md`, les mémoires `courtside-hits-overfire-device`
(état post-S7 + le mur DP restant), `courtside-s3-event-recall-negative`,
`courtside-edit-targets-main-tree` (⚠️ les pièges worktree, dont le **DESTRUCTIVE cwd-reset
variant** : JAMAIS de symlink/chemin relatif, **absolu uniquement** ; le venv du worktree est
absent → utilise `/Users/vuong/Documents/CourtSide-CV/venv/bin/python`).

**Retour utilisateur (2026-06-30), DIAGNOSTIQUÉ par la mesure :** sur le hero render il reste un
**rebond fantôme en haut** (l'utilisateur : "double détection en haut", "un deep qui est OUT mais
quand même compté"). C'est le **fantôme @231** (813,191), tout en haut, classé `deep`, balle
hors-court.

## Le diagnostic PRÉCIS (mesuré — exploite-le, ne re-suppose pas)
La trajectoire balle autour de f231 : y monte 224→219→…→**191** (f231) puis **redescend**
210→213→216… C'est un **APEX EN VOL** (sommet de l'arc de la balle, elle retombe), PAS un rebond
au sol. `detect_turning_points` + `detect_sharp_turns` firent (un coude existe) mais le
classifieur le prend pour un BOUNCE.

**Le discriminant est PROPRE et VÉRIFIÉ** (sur le dump canonique post-S1, k=3, pente-y médiane
par côté ; y croît vers le BAS de l'écran) :
- **Rebond au sol = y-MIN (creux)** : la balle DESCEND (vy_pre>0) puis REMONTE (vy_post<0), ou
  descend-puis-se-tasse (les knees near de S7).
- **Apex en vol = y-MAX (sommet)** : la balle MONTE (vy_pre<0) puis REDESCEND (vy_post>0).
- **Fantôme @231 : vy_pre=−5.7, vy_post=+6.3 → Y-MAX franc.**
- **Sur les 9 rebonds GT : 0 sont des y-max.** (b369/b456 = y-min ; les autres = y-min ou knee ;
  AUCUN ne monte-puis-redescend.) → rejeter les y-max francs ne touche **aucun** vrai rebond.

## Le fix
Dans `vision/events.py`, **interdire qu'un candidat apex-en-vol (y-MAX franc) soit labellisé
BOUNCE** : `vy_pre < 0` (la balle monte, en px/frame avec y vers le bas) ET `vy_post > 0` (elle
redescend) sur la même fenêtre half-window que les features de direction existantes → ce n'est pas
un rebond au sol, c'est un sommet d'arc. Le point d'ancrage propre = là où un candidat devient une
ancre/un score BOUNCE dans `classify_events` (lis le firewall `_reward` + le calcul `vy_bounce` /
`bscore` autour de la ligne ~430-490). Réutilise la MÊME lecture de pente que `_direction_features`
(half-window, gap/teleport guards, RAW track + is_real) — ne réinvente pas un calcul vy parallèle.
⚠️ Un apex-en-vol peut aussi être lu comme `vy_flip` (la balle inverse son vy) → assure-toi que le
gate s'applique AVANT que l'ancre vy MANDATORY ne soit posée (sinon le DP ne peut plus la skipper).

## ⚠️ INPUT CANONIQUE
Benchmarks sur **`tennis.mp4 -s 73 -d 13`** (chemin ABSOLU `/Users/vuong/Documents/CourtSide-CV/tennis.mp4`,
PAS de symlink). L'extrait `data/output/tennis_demo3.mp4` est dégradé — ne l'utilise pas.

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-s8apex feat/s8-airborne-apex-reject
```
Branche depuis `feat/accuracy-overhaul` (HEAD à jour). ⚠️ édite SOUS
`/Users/vuong/Documents/CourtSide-CV-s8apex/...`. **Touche UNIQUEMENT `vision/events.py`.**

## Les thermomètres + les 3 pièges worktree
- Replay (byte-identique au _stats.json) : `tools/event_eval/live_pool.py` sur
  `tests/fixtures/methodo/demo3_methodo_inputs_fullvideo_postS1.json`. ⚠️ PIÈGE IMPORT : cet outil
  vit dans le main tree → il importe le `vision/events.py` du MAIN tree, PAS le tien. Pour mesurer
  TON code : soit lance `run_pipeline_8s.py` DEPUIS ton worktree (chemin vidéo ABSOLU), soit
  force-load `importlib.util.spec_from_file_location('vision.events','<worktree>/vision/events.py')`.
- LIVE (arbitre) : `venv/bin/python run_pipeline_8s.py /Users/vuong/Documents/CourtSide-CV/tennis.mp4
  -s 73 -d 13 --device mps --match-mode -o /tmp/s8/out.mp4` → `score_stats.py /tmp/s8/out_stats.json`.
  ⚠️ PIÈGE OUTPUT : le `_stats.json` part où dit `-o` ; vérifie le mtime. Lance ≥2 runs.

## Cible mesurée (à reproduire/battre, replay contrôlé WITH vs WITHOUT)
Baseline hero (post-S7) : bounce F1 **0.889**, spurious **[(108,HIT),(231,BOUNCE),(487,HIT)]**.
Après S8 : le **(231,BOUNCE) disparaît** → spurious à 2, **bounce F1 ≥ 0.889 (idéalement ↑** car
1 FP de moins : P monte), **0 vrai rebond perdu** (les 9 GT inchangés), confusion H→B=0 tenue.

## Contraintes dures
- **0 vrai rebond perdu** : les 9 rebonds GT restent détectés (vérifie b62..b569 un par un).
- **confusion_H→B = 0** + firewall intact (le gate ne doit pas transformer un rebond en frappe).
- **Planchers** : bounce F1 ≥ 0.889, hit F1 ≥ 0.625 (input canonique).
- **felix + tous les tests verts** : `test_event_confusion_regression` (0/0, bF1 0.889),
  `test_bounce_regression`, `test_bounce_wasb_regression`, `test_vx_veto`, `test_sharp_turns`,
  `test_near_court_knee`, `test_static_fp_reject`, `test_ball_anchored_prox`.
- **Zéro hardcoding** : le gate = un SIGNE de pente (vy_pre<0 ET vy_post>0), pas un seuil pixel ni
  une bande tunée. Ancré sur la physique (un sommet d'arc n'est pas un rebond), pas sur 1 clip.
- **Anti-surapprentissage** : vérifie que le gate ne fire QUE sur des apex (0 sur les 9 GT) et ne
  dépend pas d'un magic number — un y-max franc est un y-max franc à toute échelle.

## Livrable — CR (OBLIGATOIRE)
`docs/research/s8-airborne-apex-reject-CR.md` : bounce F1 + liste spurious AVANT/APRÈS (replay
contrôlé + ≥2 runs LIVE), preuve que les 9 GT restent détectés et que le fantôme @231 (et tout
autre apex) est supprimé, confusion 0 + firewall intact, felix + tests verts, le gate exact + la
preuve 0/9 GT y-max. Commit + push. VRAIS chiffres mesurés.

## Note (hors-scope, ne PAS faire ici)
L'utilisateur veut AUSSI afficher "OUT" sur les vrais rebonds hors-court. C'est S9 (in/out
calling), SÉPARÉ : il dépend des limites du court (le hero clip n'a PAS d'homographie — angle
rasant, auto-court abstient). S8 supprime juste le fantôme apex ; le badge OUT viendra après que la
calibration court soit résolue. NE touche pas au court / à l'homographie ici.
