# Suivi d'avancement (v2, branche `claude/research-v2`)

Reprise apres coupure : lire ce fichier puis `experiments/RESEARCH_LOG.md`, continuer a « En cours ».
L'ancienne tentative (sans axe temporel, plafond 0,31) est archivee sur la branche `claude/research`.

## Decouverte structurante
`ID` est chronologique (pas `DAY_ID`) : t = rang de l'ID de la ligne FR du jour. Voir `src/data.py`.

## Fait
- [x] Chargement + axe temporel, blocs de features temporelles (`src/features.py`), runner, registre
- [x] A : modeles statiques (0,28) ; B : variations jour a jour (0,50) ; C : FR avance/retard (0,556)
- [x] D-E : raffinements DE et FR ; G : 2e etage sur residus des jours voisins (DE 0,78 -> 0,81)
- [x] H : FR sur tendance multi-jours (FR 0,28 -> 0,34), global ~0,595

## En cours
- [ ] J : tendances pour DE, multi-fenetres FR, echelle FR/DE, ensemble final
- [ ] Audit complet (`src/audit.py`), RESEARCH_LOG, README, soumission, verification sur clone propre
