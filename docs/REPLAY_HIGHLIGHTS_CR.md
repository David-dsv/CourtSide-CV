# CR — Replay slow-mo + Vidéo de synthèse 60s (Session A)

**Branche :** `feat/replay-highlights` (partie de `main` @ `3ff0985`, **pas** de `feat/accuracy-overhaul`)
**Commits :**
```
b807a2f feat(replay): --frame-offset flag for stats-absolute → annotated-file frames
54dc833 fix(reel): stats-absolute → annotated-file frame offset + bounce-only fallback
2241dbe feat(reel): tools/make_highlight_reel.py — 60s highlight reel
a9b2ef4 feat(replay): tools/make_replay.py — slow-mo replay of an exchange
```
**Périmètre :** backend uniquement. Aucune modification du frontend (`web/`) ni des modèles ML. Aucune modification de `run_pipeline_8s.py`. Deux nouveaux outils dans `tools/`.

---

## 1. Ce qui marche

### Feature 1 — `tools/make_replay.py` (replay slow-mo) ✅
Extrait `[start_frame, end_frame]` de la **vidéo annotée**, rejoue au ralenti (`setpts` vidéo + `atempo` audio chaîné), ré-injecte l'audio ralenti du même facteur. Validé :
- **Slow-mo 0.25 (4×)** avec audio → 23.46s pour 5.88s source. ✓
- **Slow-mo extrême 0.1 (8×)** avec audio (chaîne `atempo=0.5×4`) → 30.02s. ✓
- **Source sans audio** (cas réel : le `VideoWriter` du pipeline strippe l'audio) → replay silencieux, warning clair. ✓
- **Bornes hors-range** clamped proprement + warning. ✓
- **`--slowmo` invalide (>1.0 ou ≤0)** rejeté avec erreur claire. ✓
- **`--frame-offset`** pour mapper les frames absolues du `stats.json` vers les frames du fichier annoté. ✓

### Feature 2 — `tools/make_highlight_reel.py` (reel 60s) ✅
Lit `_stats.json`, score les moments, concatène en ordre chronologique avec cartons texte, mute l'audio. Validé :
- **Chemin shots+rallies** (felix : 7 shots, 20 bounces, 0 rallies) → rallies dérivées des shots (gap >2.5s), scoring best-shot (speed×quality) + review (pire qualité). ✓
- **Chemin bounces-only** (clip pipeline `main` : 0 shots, 13 bounces) → fallback sur clusters de bounces. ✓
- **Offset de frames** absolu→fichier géré automatiquement depuis `frame_range`. ✓
- **Audio** muxé depuis une source alignée (`--audio-source`), sinon silencieux. ✓
- **Budget ≤60s** respecté (truncation chronologique). ✓
- **Dédup des chevauchements** (>50% overlap → fusion, garde le label du meilleur score). ✓

### Dépendances
- `ffmpeg` + `ffprobe` obligatoires (vérifiés au démarrage, erreur claire sinon). macOS : `brew install ffmpeg`.
- `cv2`, `numpy`, `PIL` (déjà dans le projet). `loguru` (idem) avec fallback `logging`.

---

## 2. Ce qui ne marche pas / limites à connaître (honnêtement)

| # | Limite | Détail / contournement |
|---|---|---|
| **L1** | **L'audio du reel n'est pas synchro par-segment.** | Le reel mute l'audio pris **au début** (`offset 0`) de `--audio-source`, trimé à la durée totale du reel. Pour un reel à un seul clip c'est correct, mais l'audio ne correspond **pas** au temps-source exact de chaque segment highlight (les segments sont extraits sans audio puis concaténés). Pour un vrai sync audio-per-segment, il faudrait extraire l'audio de chaque fenêtre source et le concaténer aussi — pas fait ici (coût ffmpeg ×N, et l'audio de highlight n'a souvent pas besoin d'être exact). **Action frontend :** passer `--audio-source` = clip source brut aligné au `frame_range[0]`, ou accepter un reel silencieux. |
| **L2** | **`atempo` a une borne inférieure de 0.5 par filtre.** | Géré : on chaîne les `atempo=0.5` (slowmo 0.1 = 4 filtres). Pas de bug, mais l'audio très ralenti (<0.25) devient grave/artificiel — c'est une limite physique de l'algorithme, pas du code. |
| **L3** | **Les replays des clips annotés du pipeline sont silencieux** car `run_pipeline_8s.py`/`VideoWriter` ne ré-injectent pas l'audio (CLAUDE.md le documente). | Pour un replay **avec son**, il faut d'abord muxer l'audio source sur l'annoté (voir §5 commande), OU que le frontend garde le clip source brut comme source audio. |
| **L4** | **Sélection du reel peut sous-remplir le budget 60s.** | Le dédup des chevauchements tourne **avant** le budgeting. Si tous les top-moments se chevauchent (même échange), le reel peut faire <60s même s'il y a assez de matériel. Vu en test : 15s et 14s au lieu de 60s. Défendable (évite la redondance) mais à savoir. Réglable via `--top-rallies/--top-best-shots/--top-review`. |
| **L5** | **Cartons texte :Police PIL fallback.** | Utilise Helvetica/Arial système ; fallback `cv2` si absent. Rendu correct sur macOS. Police non-embarquée → aspect potentiellement différent sur Linux serveur. |
| **L6** | **Perf :** chaque segment est ré-encodé (libx264 preset fast). | Reel 60s ≈ 3-5s de traitement sur ce Mac. Replay 6s source → ~2s. Negligeable vs le pipeline. Pour aller plus vite : `-preset ultrafast` ou stream-copy si même codec (non fait : garde la robustesse). |
| **L7** | **Le reel suppose `frame_range` dans le stats.json.** | Toujours présent dans les stats du pipeline. Si absent, fallback `[0, n_frames]` (offset 0). |

---

## 3. Fichiers produits + durées (test réel)

Sorties générées pendant la validation (toutes H.264, lisibles) :

| Fichier | Durée | Frames | Vidéo | Audio |
|---|---|---|---|---|
| `data/output/tennis_reeltest_replay.mp4` | **24.42s** | 306 | h264 | none (source annotée sans audio) |
| `data/output/tennis_reeltest_annotated_reel.mp4` | **13.80s** | 690 | h264 | aac (muxé depuis source alignée) |
| `data/output/felix_FINAL_clean_reel.mp4` | **15.32s** | 919 | h264 | aac (silencieux : `anullsrc`) |
| `/tmp/replay_audio_test.mp4` *(replay avec audio, fixture)* | **23.46s** | 294 | h264 | aac (ralenti) |

> Le reel tennis fait 13.8s et felix 15.3s (pas 60s) → limite **L4** (dédup). Pour un reel qui remplit 60s, utiliser un clip source plus long / plus d'échanges.

---

## 4. Signatures CLI exactes (pour le frontend)

### `make_replay.py`
```
usage: make_replay.py [-h] --start-frame START_FRAME --end-frame END_FRAME
                      [--fps FPS] [--slowmo SLOWMO]
                      [--frame-offset FRAME_OFFSET] [-o OUT] video
```
| Arg | Requis | Défaut | Sens |
|---|---|---|---|
| `video` (positionnel) | oui | — | chemin vidéo **annotée** |
| `--start-frame` | oui | — | frame de début (voir offset ci-dessous) |
| `--end-frame` | oui | — | frame de fin (inclusive) |
| `--fps` | non | auto (ffprobe) | fps source |
| `--slowmo` | non | `0.25` | facteur (0, 1.0]. 0.25 = 4× lent |
| `--frame-offset` | non | `0` | **soustraire** des frames start/end. = `stats.frame_range[0]` si les frames viennent du stats.json |
| `-o/--out` | non | `data/output/<stem>_replay_<start>.mp4` | sortie |

**Sortie :** `data/output/<video_stem>_replay_<start_frame>.mp4` (exit 0). Exit ≠0 + message sur erreur (ffmpeg manquant, segment vide, slowmo invalide).

**⚠️ Frontend — mapping frames stats → fichier annoté :** les frames dans `stats.json` (`shots[].frame`, `bounces[].frame`) sont **absolues (source)**. La vidéo annotée ne contient que la fenêtre analysée → frame 0 du fichier annoté == `stats.frame_range[0]`. **Passez toujours `--frame-offset <stats.frame_range[0]>`** quand les frames viennent du stats.json. Sinon le replay cherchera au-delà de la fin du fichier.

### `make_highlight_reel.py`
```
usage: make_highlight_reel.py [-h] --stats STATS --video VIDEO [-o OUT]
                              [--max-duration MAX_DURATION]
                              [--card-seconds CARD_SECONDS]
                              [--top-rallies TOP_RALLIES]
                              [--top-best-shots TOP_BEST_SHOTS]
                              [--top-review TOP_REVIEW]
                              [--audio-source AUDIO_SOURCE]
```
| Arg | Requis | Défaut | Sens |
|---|---|---|---|
| `--stats` | oui | — | chemin `<name>_stats.json` |
| `--video` | oui | — | vidéo annotée correspondante |
| `-o/--out` | non | `data/output/<video_stem>_reel.mp4` | sortie |
| `--max-duration` | non | `60.0` | cap duréeee (s) |
| `--card-seconds` | non | `1.2` | durée d'un carton texte (s) |
| `--top-rallies` | non | `5` | nb max de rallies sélectionnés |
| `--top-best-shots` | non | `3` | nb max "frappe du match" |
| `--top-review` | non | `2` | nb max "à revoir" |
| `--audio-source` | non | `--video` | piste audio à muxer (clip source brut recommandé) |

**Sortie :** `data/output/<video_stem>_reel.mp4` (exit 0). Gère l'offset de frames **automatiquement** depuis `stats.frame_range` (pas besoin de flag).

---

## 5. Commandes exactes pour reproduire

```bash
# 0) Prérequis : sur la branche feat/replay-highlights
git checkout feat/replay-highlights

# 1) Pipeline canonique sur ~15s de tennis.mp4 (produit l'annoté + le stats.json)
python run_pipeline_8s.py tennis.mp4 -s 60 -d 15 --device mps \
  -o data/output/tennis_reeltest_annotated.mp4
# → data/output/tennis_reeltest_annotated.mp4  +  _stats.json

# 2) Replay slow-mo d'un échange (frames 3017→3322 du stats, offset = frame_range[0]=3000)
python tools/make_replay.py data/output/tennis_reeltest_annotated.mp4 \
  --start-frame 3017 --end-frame 3322 --fps 50 --slowmo 0.25 --frame-offset 3000
# → data/output/tennis_reeltest_annotated_replay_17.mp4 (24.4s)

# 3) (Optionnel) Replay AVEC son : d'abord muxer l'audio source sur l'annoté
ffmpeg -y -i data/output/tennis_reeltest_annotated.mp4 \
  -ss 60 -t 15 -i tennis.mp4 -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest \
  /tmp/tennis_annotated_audio.mp4
python tools/make_replay.py /tmp/tennis_annotated_audio.mp4 \
  --start-frame 17 --end-frame 322 --fps 50 --slowmo 0.25   # ici offset 0 (déjà muxé)

# 4) Reel 60s depuis le stats.json (offset géré auto)
python tools/make_highlight_reel.py \
  --stats data/output/tennis_reeltest_annotated_stats.json \
  --video data/output/tennis_reeltest_annotated.mp4 \
  --max-duration 60
# → data/output/tennis_reeltest_annotated_reel.mp4 (silencieux, l'annoté n'a pas d'audio)

# 5) Reel AVEC son : passer le clip source brut comme audio-source
ffmpeg -y -ss 60 -t 15 -i tennis.mp4 -vn -acodec aac /tmp/reel_audio.m4a
python tools/make_highlight_reel.py \
  --stats data/output/tennis_reeltest_annotated_stats.json \
  --video data/output/tennis_reeltest_annotated.mp4 \
  --audio-source /tmp/reel_audio.m4a --max-duration 60
# → reel avec piste audio aac

# 6) Vérifier qu'un output est lisible
ffprobe -v error -show_entries format=duration:stream=codec_type,codec_name <file>.mp4
```

---

## 6. Notes d'intégration frontend (important)

1. **Offset de frames — point critique.** `make_replay.py` expose `--frame-offset` ; le frontend doit passer `stats.frame_range[0]` quand il sélectionne un échange depuis les frames du stats.json. `make_highlight_reel.py` gère l'offset **tout seul** (lit `frame_range`).
2. **Audio :** ni l'annoté du pipeline ni le reel par défaut n'ont de son utile. Pour du son, le frontend doit fournir le clip **source brut** à `--audio-source` (reel) ou muxer l'audio d'abord (replay). Recommandation produit : conserver le clip source brut en plus de l'annoté.
3. **Zéro hardcoding respecté :** toutes les bornes en frames, durées en secondes, ratios de fps. Aucun nombre magique lié à une vidéo. La taille des cartons et polices sont des ratios de la hauteur de frame.
4. **Scoring du reel (défendable, simple) :**
   - top rallies = qualité moyenne des frappes **near-side** (le joueur filmé) ;
   - top best shots = `speed_kmh × quality` ;
   - top review = `quality` la plus basse ;
   - fallback bounces-only = clusters (>2.5s gap) scorés par vitesse moyenne.
5. **Le scoring dépend de la qualité du `stats.json`** : si le pipeline `main` n'émet pas de `shots` (cas vu), le reel tombe sur le fallback bounces — moins riche mais fonctionnel.

---

## 7. Algorithme / détails techniques clés (leçons apprises)

- **`-ss` et `-t` DOIVENT précéder `-i`** dans ffmpeg pour le slow-mo. En option de *sortie*, `-t` cappe la durée de sortie à la durée source et **annule silencieusement** le ralenti (le output paraît correct mais n'est pas ralenti). C'est le bug #1 qu'on a debuggé.
- **`atempo` borne [0.5, 100]** : pour slowmo <0.5 on chaîne `atempo=0.5`. Géré génériquement.
- **`stats.json` frames sont absolues** ; la vidéo annotée ne contient que `frame_range`. Sans offset, toute extraction seek au-delà de la fin du fichier → `concat: Output file does not contain any stream`.
- **Concat demuxer ffmpeg** (`-f concat -c copy`) nécessite segments uniformes (fps/size/pixfmt) → on normalise le fps et on ré-encode en h264 chaque segment.
