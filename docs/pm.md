# pm.md — Playbook Chef de Projet (orchestration multi-sessions)

> Ce fichier est la **source de vérité** pour toute session jouant le rôle de
> **chef de projet** sur CourtSide-CV. Une session chef de projet n'écrit pas
> (ou peu) de code elle-même : elle **découpe le travail en chantiers** et les
> **délègue à d'autres sessions** via des prompts écrits dans des `.md`.

## Le modèle d'orchestration

L'utilisateur (humain) peut lancer **autant de sessions Claude en parallèle**
que nécessaire. Le chef de projet n'a qu'à :

1. **Découper** un objectif en chantiers **disjoints** (aucun chevauchement de
   fichiers entre sessions → intégration sans conflit).
2. **Écrire un prompt** par chantier dans un fichier `.md`
   (ex. `docs/prompts/<nom>-session.md`), selon le **format canonique** ci-dessous.
3. **Donner les `.md` à l'utilisateur**, qui lance les sessions.
4. **Recevoir les CR** (comptes-rendus) que l'utilisateur renvoie.
5. **Synthétiser + intégrer** : fusionner les branches, résoudre les points
   transverses, valider (build/lint/tests), committer/pousser.

Le chef de projet reste **dans la boucle** : il lit chaque CR, décide de
l'intégration, et relance des prompts de correction si un chantier est
incomplet. C'est plus rapide, plus parallèle, et plus sûr que de tout faire
séquentiellement dans une seule session.

## Format canonique d'un prompt de session

Chaque `.md` de prompt DOIT contenir :

```
# PROMPT — <Nom court du chantier> (<Frontend|Backend|...>)

## Contexte
- Repo, point d'entrée, fichiers à lire (CLAUDE.md, PROJET.md, modules clés).
- Le PROBLÈME précis (ne pas partir à l'aveugle — diagnostiquer avant).

## Worktree (OBLIGATOIRE — voir ci-dessous)
git worktree add ../CourtSide-CV-<slug> feat/<branche>
Branche : feat/<branche> (depuis <base>). Commit/étape, push après chaque.

## Objectif
Ce que la session doit livrer, avec critères de réussite mesurables.

## Étapes
1. ... 2. ... (exécutoires, avec ré-itération si pertinent).

## Validation
- Tests/build/lint à passer.
- Validation visuelle/empirique contre de vraies données quand applicable.

## CR à la fin (OBLIGATOIRE)
- Ce qui marche / pas (honnêtement).
- Fichiers + signatures/props exposées pour l'intégration.
- Limites connues + TODO backend/frontend.
- Résultats exacts des tests/build.
```

## Règles non-négociables

### 1. Worktree isolé par session (CRITIQUE)
Le repo working tree est **partagé**. Si plusieurs sessions font `git checkout`
dans le même dossier, elles s'écrasent mutuellement (incident vécu : un
checkout a fait disparaître `web/` entier + des éditions orphelines).
→ **Chaque session travaille dans son propre worktree**, sur sa propre branche.
Le chef de projet nettoie les worktrees une fois la branche intégrée :
`git worktree remove ../CourtSide-CV-<slug>`.

### 2. Chantiers disjoints
Deux sessions ne doivent **jamais** modifier les mêmes fichiers. Le découpage
se fait par couche (frontend `web/` vs backend `vision/`/`run_pipeline_8s.py`
vs outils `tools/`). Au moment de l'intégration, le chef de projet
cherry-pick/fast-forward et résout à la main les rares conflits transverses
(typiquement `run_pipeline_8s.py` ou `web/src/lib/types.ts`).

### 3. CR obligatoire
Aucune session ne se termine sans un **compte-rendu structuré** (format ci-dessus)
que l'utilisateur renvoie au chef de projet. Le CR est le **seul lien** entre
la session et le chef de projet (pas d'état partagé).

### 4. Le chef de projet valide, ne fait pas confiance aveuglément
Chaque CR est confronté à la réalité : relire le diff, faire tourner
build/lint/tests, vérifier les signatures annoncées. Si un CR est incomplet ou
inexact, **relancer un prompt de correction** (nouvelle session ou même
worktree) plutôt que d'intégrer du bancal.

## Workflow d'intégration (après réception des CR)

1. Se positionner sur la **branche cible** (souvent `feat/accuracy-overhaul`).
2. **Fast-forward** les branches basées sur la même base (ex. frontend sur
   accuracy-overhaul).
3. **Cherry-pick** les branches basées sur `main` (fichiers disjoints → 0
   conflit normalement).
4. **Ré-appliquer à la main** les blocs transverses qui ont divergé
   (typiquement l'émission `_stats.json` dans `run_pipeline_8s.py`).
5. **Résoudre les points transverses** identifiés pendant le découpage
   (cohérence de seuils/algorithme entre Python et TS, source de vérité
   unique, badges de confiance).
6. **Valider bout-en-bout** : `pnpm build` + `pnpm lint` + `tsc` côté web,
   `py_compile` + tests Python côté pipeline.
7. **Committer un commit de synthèse**, pousser, nettoyer les worktrees.

## Mémoire

Le chef de projet maintient la mémoire persistante (`~/.claude/.../memory/`) :
- une note par **décision de spec** non-évidente (pourquoi tel seuil, pourquoi
  tel label est faible confiance),
- une note sur les **leçons d'orchestration** (ex. worktree-par-session,
  collision de branche),
- mettre à jour les notes existantes quand un gap d'intégration est résolu
  (marquer `RESOLVED <commit>`).

## Anti-patterns

- ❌ Le chef de projet écrit tout le code lui-même (gaspille le parallélisme).
- ❌ Deux sessions touchent aux mêmes fichiers (conflits à l'intégration).
- ❌ Intégrer un CR sans le re-vérifier (bancal => dette).
- ❌ Oublier de nettoyer un worktree (encombrement).
- ❌ Sessions qui font `git checkout` dans le working tree principal au lieu
  d'un worktree (collision — l'incident fondateur).
