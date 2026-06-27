# PROMPT — Far-select INTÉGRATEUR : partir de A, greffer le meilleur de B (Backend) — SESSION 3/5

> **Session AUTONOME ULTRACODE.** Lance en `/ultracode`. Les 2 sessions far-select
> ont FINI et poussé leur CR — tu peux démarrer immédiatement.
> ⛔ **NE POSE JAMAIS DE QUESTION à l'utilisateur. Aucun outil de clarification.**
> Décide seule, documente, ne bloque jamais. Une question = run raté.

## Le verdict (mesuré + vérifié indépendamment par le PM)
Les deux approches, scorées sur la GT humaine (`tools/pose_gt/measure_far_coverage.py`, baseline 0.9%) :

| | far-CORRECT demo3 | far-CORRECT felix | généralise ? |
|---|---|---|---|
| **A** `feat/farselect-conf-motion` | **84.9%** ✅ | **87.9%** ✅ | **OUI (les deux)** |
| **B** `feat/farselect-courtband-ball` | 81.3% | **4.5%** ❌ CASSE felix | non (surapprend demo3) |

**A DOMINE B** : A bat B sur demo3 ET préserve felix, alors que la centralité PURE de B détruit felix (elle choisit la structure centrale sur l'angle corner de felix). **Pars donc de A comme BASE**, ne fusionne pas 50/50.

## L'insight central (de A, à conserver)
> Le vrai joueur far = la box **la plus HAUTE sur le court au-dessus du filet** (signal scale-free, indépendant de l'angle), PAS la plus centrale. Score unifié `hcv` = `h/hmax + 0.3·conf + 1.2·central + 0.3·valid` sur une gate on-court. **La validité doit être PONDÉRÉE, jamais une gate dure** (le far minuscule ne pose pas → un reject nkp≥10 jette la bonne box). conf détecteur far baissée à 0.08 (rend le far détectable).

## Ce que B apporte (à tester en greffe, GARDER seulement si ça aide sans casser felix)
1. `is_geometric_far_player` : sélection far par géométrie sans nkp (B et A convergent là-dessus — déjà dans A via la validité pondérée). Vérifie que A le fait aussi bien.
2. **Bande de court géométrique** (dérivée des baselines/zones) pour rejeter les gradins — potentiellement complémentaire à la hauteur de A.
3. **Proximité balle far-side** comme tiebreak quand l'échange est côté far.
4. Note de B : `net_y=0.5·fh` classe mal le near/far sur angle rasant felix → **dériver net_y du filet détecté** quand dispo (amélioration transverse utile).

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-farINT feat/farselect-integrated
```
Branche `feat/farselect-integrated` (depuis `feat/accuracy-overhaul`). Commit/push par itération.
⚠️ `courtside-edit-targets-main-tree` : édite SOUS `/Users/vuong/Documents/CourtSide-CV-farINT/...`. **Touche `vision/pose.py`** + un **nouveau** `tests/test_far_coverage.py`.

## Étape 1 — partir du code de A
Récupère le `vision/pose.py` de A : `git checkout feat/farselect-conf-motion -- vision/pose.py` (dans ton worktree). Confirme le baseline : `python tools/pose_gt/measure_far_coverage.py` ≈ **84.9% demo3**. Lis les DEUX CR (`docs/research/farselect-conf-motion-CR.md` et `farselect-courtband-ball-CR.md`) en entier.

## Étape 2 — greffer + itérer
Teste chaque idée de B (bande de court, balle tiebreak, net_y détecté) UNE par UNE : garde seulement si demo3 monte **ET** felix reste ≥ ~85%. **Mesure demo3 ET felix à chaque greffe** (felix proxy : A a un check felix — réutilise-le ; ne casse jamais felix). Itère sans plafond jusqu'à battre 84.9% demo3 ou plateau prouvé.

## Contraintes dures
- **Cible** : far-CORRECT demo3 ≥ 84.9% (battre A) ET felix ≥ ~85% (ne pas régresser comme B). Si aucune greffe n'améliore, livre A tel quel — c'est déjà le meilleur.
- **NE régresse PAS** felix ni les tests : `test_bounce_regression` ≥0.72, `test_bounce_wasb_regression` ≥0.80, `test_vx_veto`, `test_event_confusion_regression`, `test_sharp_turns` — verts à chaque itération.
- **Zéro hardcoding** ; anti-surapprentissage (A note un plateau ±grid ≥80% sur les 2 clips — reste dans ce bassin).
- **Plafond connu** : demo3 ceiling = 89.3% (24/225 GT sans candidat = trou de DÉTECTION, pas sélection). Ne vise pas 100% — viser ~89%.

## Étape 3 — verrouiller
Ajoute `tests/test_far_coverage.py` qui asserte far-CORRECT demo3 ≥ un plancher sûr (ex. 0.80) sur la GT committée. Rapide, pas de GPU si possible (ou marque-le slow).

## Livrable — CR (OBLIGATOIRE)
`docs/research/farselect-integrated-CR.md` : ce que tu as gardé de A, ce que tu as greffé de B (et ce que tu as rejeté + pourquoi), far-CORRECT demo3+felix avant/après, preuve felix intact, le nouveau test. Commit + push sur `feat/farselect-integrated`. VRAIS chiffres mesurés uniquement.
