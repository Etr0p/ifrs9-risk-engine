# PLAN DE FUSION — Branche `unified`

> **Objectif** : Fusionner pe-bc (5 secteurs × 2 canaux) et 14-couches (14 classes d'actifs) sur une branche unique `unified`, sans dashboard, avec indépendance totale des deux modèles et garde-fous testés.

---

## ÉTAT DES LIEUX VÉRIFIÉ (pas d'hallucination)

### Branches
| Branche | Commit | Contenu |
|---------|--------|---------|
| `pe-bc` | `dadda70` | 5 secteurs × 2 canaux, BL-CVaR 10 cellules, config.py monolithique, comparator.py monolithique, 14 tests (6620 l.) |
| `14-couches` | `14382ab` | 14 classes d'actifs, BL-CVaR N classes, config/ package (7 fichiers), comparator/ package (6 fichiers), 34 tests (~13275 l.) |
| `master` | `b97ee5c` | 14-couches sans Volcker (ancêtre de 14382ab) |

### Ancêtre commun
`39ecd8d` — les deux branches ont divergé depuis ce commit.

### Architecture pe-bc (`dadda70`)
- `ifrs9_cockpit/config.py` — monolithique, 1522 lignes, contient `SectorConfig`, 5 `SECTORS`, `PREDEFINED_SCENARIOS` (11 scénarios), `BASEL_CONFIG`, etc.
- `ifrs9_cockpit/config/calibration.py` (388 l.) + `config/generators.py` (504 l.) — sous-modules
- `ifrs9_cockpit/engine/comparator.py` — monolithique, 1892 lignes, contient `PortfolioComparator` + BL-CVaR 10 cellules intégré
- `ifrs9_cockpit/engine/` — 15 modules (dont hmm_regime.py, conformal.py, compliance_gates.py, gflownet.py, rmt.py, signatures.py, sobol_analysis.py, tda.py)
- `ifrs9_cockpit/dashboard/` — callbacks.py (1747 l.), charts.py (1307 l.), etc.
- `ifrs9_cockpit/app.py` (1340 l.) + `dash_app.py` (75 l.)
- `ifrs9_cockpit/analytics/ai_analyst.py` (1782 l.) — monolithique
- `ifrs9_cockpit/tests/` — 14 fichiers, 6620 lignes totales

### Architecture 14-couches (`14382ab`)
- `ifrs9_cockpit/config/` — package avec 7 fichiers :
  - `__init__.py` (re-exports), `sectors.py` (888 l. = SectorConfig + AssetClassProfile + 14 ASSET_CLASSES), `scenarios.py` (369 l. = 12 scénarios dont Volcker), `models.py` (440 l.), `basel.py` (170 l.), `rules.py` (179 l.), `dashboard.py` (95 l.)
- PAS de `config.py` monolithique
- `ifrs9_cockpit/engine/comparator/` — package avec 6 fichiers :
  - `__init__.py` (76 l.), `__main__.py` (226 l.), `crr3.py` (88 l.), `metrics.py` (880 l.), `optimizer.py` (1073 l. = OptimizerMixin BL-CVaR N classes), `sensitivity.py` (282 l.)
- PAS de `engine/comparator.py` monolithique
- `ifrs9_cockpit/synthetic_generator/` — package (12+ positionneurs pour 14 classes)
- `ifrs9_cockpit/virtual_cro/` — package (5 fichiers)
- `ifrs9_cockpit/causal/` — package (2 fichiers)
- `ifrs9_cockpit/data/fetch_*.py` — 12 fichiers de collecte
- `ifrs9_cockpit/analytics/ai_analyst/` — package (3 fichiers)
- `ifrs9_cockpit/dashboard/callbacks/` — package (8 fichiers)
- `ifrs9_cockpit/dashboard/charts/` — package (10 fichiers)
- `ifrs9_cockpit/dashboard/components/` — package (6 fichiers)
- `ifrs9_cockpit/tests/` — 34 fichiers, ~13275 lignes

### Code mort identifié (à supprimer)
- `.claude/worktrees/angry-edison/` (46 fichiers, ~19K lignes) — snapshot obsolète
- `ifrs9_cockpit/.claude/worktrees/competent-wozniak/` (46 fichiers, ~19K lignes) — idem
- `ifrs9_cockpit/engine/conformal.py` (239 l.) — jamais importé, ne résout pas le bon problème
- `ifrs9_cockpit/engine/compliance_gates.py` (295 l.) — jamais importé, redondant
- `ifrs9_cockpit/diag_v44.py` (112 l.) — orphelin
- `ifrs9_cockpit/synthetic_generator_v4.py` (21 l. ou 1661 l. selon branche) — shim backward compat
- Scripts racine : ~26 fichiers / ~10K lignes (outils d'analyse one-shot)

### Code à CONSERVER pour réintégration (Étape 3)
- `ifrs9_cockpit/engine/hmm_regime.py` (767 l.) — détecteur HMM 3 régimes (Hamilton 1989)

---

## DETTE TECHNIQUE — REMISE À NIVEAU ARCHITECTURALE

### A. Migration Polars (CRITIQUE)

**pe-bc est 100% pandas** (290+ références, 0 polars). 14-couches est **hybride** (640+ polars dans le moteur, ~260 pandas restants dans models/data/export).

#### Fichiers pe-bc à migrer vers polars (portage dans unified)

Le code pe-bc qui arrive dans unified doit être en polars, pas en pandas. La base 14-couches a déjà le `frame_compat.py` (bridge pandas↔polars) pour la transition progressive.

**Priorité 1 — Optimiseur pe-bc** (le code qu'on porte activement) :
- `optimizer_pebc.py` (extrait de comparator.py) : actuellement pandas (`pd.DataFrame`, `.values`, `.loc[]`). Doit être converti en polars natif (`pl.DataFrame`, `.filter()`, `.select()`). Environ 400 lignes d'optimiseur à migrer.

**Priorité 2 — Générateur pe-bc** :
- `data/generator.py` : 24 refs pandas sur pe-bc, aussi 24 refs pandas sur 14-couches. La version 14-couches EST la même (pas migrée non plus). → Migrer UNE SEULE FOIS dans unified.

**Priorité 3 — Modules partagés encore en pandas** (dette commune aux deux branches) :

| Module | pandas refs (14-couches) | Statut |
|--------|--------------------------|--------|
| `models/pd_model.py` | 28 | NON migré |
| `models/lgd_model.py` | 8 | NON migré |
| `models/ead_model.py` | 7 | NON migré |
| `models/pe_model.py` | 6 | NON migré |
| `models/woe.py` | 10 | NON migré |
| `data/generator.py` | 24 | NON migré |
| `training/train.py` | 12 | Partiel |

(Les modules ai_analyst, causal, fetch_*.py, export sont supprimés — plus dans le scope polars.)

**Stratégie de migration** :
1. Le code **moteur** (engine/) doit être 100% polars — c'est la cible
2. Les **modèles** (models/) utilisent `frame_compat` en bridge — scikit-learn/TabNet attendent du pandas/numpy, on convertit aux frontières
3. Le **générateur** (data/generator.py) et **training** (train.py) : migrer polars natif

### B. Nettoyage agressif — NE GARDER QUE CE QUI SERT AU MÉMOIRE

Le mémoire a besoin de : (1) faire tourner les deux optimiseurs pour produire FC/RAROC/EVA, (2) entraîner les modèles, (3) valider par les tests. **Tout le reste dégage.**

#### Modules à SUPPRIMER (inutiles pour le mémoire)

| Module | Lignes | Raison de suppression |
|--------|--------|----------------------|
| `dashboard/` (tout) | ~6850 | Pas de dashboard |
| `dashboard/callbacks/` | ~2500 | Pas de dashboard |
| `dashboard/charts/` | ~2000 | Pas de dashboard |
| `dashboard/components/` | ~1200 | Pas de dashboard |
| `app.py` | 1340 | Entry point dashboard |
| `dash_app.py` | 75 | Entry point dashboard |
| `assets/watchdog.js` | ~50 | Dashboard live-reload |
| `virtual_cro/` (5 fichiers) | ~1900 | Alertes CRO = feature dashboard, importé UNIQUEMENT par dashboard + ai_analyst |
| `analytics/virtual_cro.py` | 526 | Wrapper de virtual_cro/ |
| `analytics/ai_analyst/` (3 fichiers) | ~600 | Prose dashboard, importé par dashboard + export seulement |
| `analytics/ai_analyst.py` (pe-bc) | 1782 | Version monolithique idem |
| `causal/` (2 fichiers) | ~530 | Importé UNIQUEMENT par dashboard |
| `api.py` (FastAPI) | 331 | Serveur runtime, pas de calcul |
| `data/fetch_*.py` (12 fichiers) | ~800 | ETL données réelles, le mémoire utilise du synthétique |
| `models/gnn_pd_model.py` | ~300 | Jamais importé par rien |
| `engine/contagion.py` | 228 | Importé UNIQUEMENT par dashboard |
| `engine/vrp.py` | 178 | Importé UNIQUEMENT par dashboard |
| `engine/conformal.py` | 239 | Dead code (jamais importé) |
| `engine/compliance_gates.py` | 295 | Dead code (jamais importé) |
| `tests/smart_test_selector.py` | 460 | Gadget, pas essentiel |
| `tests/test_dashboard.py` | ~1042 | Plus de dashboard |
| `tests/test_virtual_cro.py` | 681 | Plus de virtual_cro |
| `tests/test_causal.py` | 526 | Plus de causal |
| `config/dashboard.py` | 95 | Couleurs/palettes dashboard |
| `export/audit_trail_latex.py` | 1396 | Export dashboard déguisé — les tableaux mémoire sortent des scripts racine |

**Total supprimé : ~24 400+ lignes**

#### Modules à GARDER (pipeline de calcul du mémoire)

| Module | Lignes | Rôle mémoire |
|--------|--------|--------------|
| **config/** (sectors, scenarios, models, basel, rules) | ~2050 | Définit secteurs, classes, scénarios |
| **engine/comparator/** (init, crr3, metrics, optimizer, sensitivity) | ~2625 | Optimiseur 14-classes BL-CVaR |
| **engine/comparator/optimizer_pebc.py** (à créer) | ~400 | Optimiseur pe-bc BL-CVaR 10 cellules |
| **engine/ecl_calculator.py** | 638 | Calcul ECL |
| **engine/pe_calculator.py** | 650 | Calcul PE (NAV, MOIC, distress) |
| **engine/staging.py** | 252 | Staging IFRS 9 (S1/S2/S3) |
| **engine/balance_sheet_ecl.py** | ~200 | ECL bilan (importé par comparator) |
| **engine/hmm_regime.py** | 767 | HMM 3 régimes (étape 3) |
| **engine/gflownet.py** | 620 | Importé par training + comparator/sensitivity |
| **engine/rmt.py** | 180 | Importé par comparator/optimizer |
| **engine/tda.py** | 331 | Importé par training |
| **engine/signatures.py** | 290 | Importé par training |
| **engine/sobol_analysis.py** | 287 | Importé par training |
| **models/** (pd, lgd, ead, pe, woe) | ~2650 | Modèles ML |
| **data/generator.py** | 902 | Générateur synthétique 5 secteurs |
| **synthetic_generator/** (14 fichiers) | ~3000 | Générateur 14 classes |
| **training/train.py** | 243 | Pipeline d'entraînement |
| ~~export/audit_trail_latex.py~~ | ~~1396~~ | **SUPPRIMÉ** — les tableaux mémoire sont produits par les scripts racine (extract_memoir_tables.py, calc_franchise_cost.py), pas par ce module. C'est un export dashboard déguisé. |
| **schemas.py** (Pandera) | 254 | Validation données |
| **utils/** (helpers, logging, frame_compat, dataset_bundle) | ~485 | Utilitaires |

**Total gardé : ~16 000 lignes** (vs ~51 000 sur 14-couches = -69%)

### C. Scénarios : unification

- pe-bc a **11 scénarios** (pas de Volcker) dans `config.py`
- 14-couches a **12 scénarios** (avec Volcker) dans `config/scenarios.py`
- Unified doit avoir **12 scénarios** (Volcker inclus — déjà présent sur la base 14-couches)
- Les tests pe-bc qui hardcodent `len(PREDEFINED_SCENARIOS) == 11` doivent être mis à jour vers 12

### D. Migration Polars — modules restants

Le tableau P3 est réduit (plus de ai_analyst, causal, fetch) :

| Module | pandas refs | Action |
|--------|------------|--------|
| `models/pd_model.py` | 28 | `frame_compat` bridge (scikit-learn/TabNet = pandas) |
| `models/lgd_model.py` | 8 | `frame_compat` bridge |
| `models/ead_model.py` | 7 | `frame_compat` bridge |
| `models/pe_model.py` | 6 | `frame_compat` bridge |
| `models/woe.py` | 10 | `frame_compat` bridge |
| `data/generator.py` | 24 | Migrer polars natif |
| `training/train.py` | 12 | Migrer polars natif |

---

## ÉTAPE 1+2 : CRÉER LA BRANCHE `unified`

### 1.1 — Créer la branche de sécurité
```bash
git checkout 14-couches          # base = 14382ab (le plus modulaire)
git checkout -b unified           # nouvelle branche
git checkout -b unified-backup    # backup de sécurité
git checkout unified              # revenir sur unified
```

### 1.2 — Supprimer TOUT ce qui ne sert pas au mémoire

**Code mort** :
```
rm -rf .claude/worktrees/
rm -rf ifrs9_cockpit/.claude/worktrees/
rm ifrs9_cockpit/engine/conformal.py
rm ifrs9_cockpit/engine/compliance_gates.py
rm ifrs9_cockpit/diag_v44.py
rm ifrs9_cockpit/synthetic_generator_v4.py
```

**Dashboard** (tout) :
```
rm -rf ifrs9_cockpit/dashboard/
rm ifrs9_cockpit/app.py
rm ifrs9_cockpit/dash_app.py
rm -rf ifrs9_cockpit/assets/
rm ifrs9_cockpit/config/dashboard.py
```

**Modules CRO / AI Analyst / Causal** (= features dashboard) :
```
rm -rf ifrs9_cockpit/virtual_cro/
rm ifrs9_cockpit/analytics/virtual_cro.py
rm -rf ifrs9_cockpit/analytics/ai_analyst/
rm -rf ifrs9_cockpit/causal/
```

**Modules importés uniquement par le dashboard** :
```
rm ifrs9_cockpit/engine/contagion.py
rm ifrs9_cockpit/engine/vrp.py
```

**API runtime** (pas de serveur dans le mémoire) :
```
rm ifrs9_cockpit/api.py
```

**Data fetchers** (mémoire = données synthétiques) :
```
rm ifrs9_cockpit/data/fetch_*.py
```

**Modèles morts** :
```
rm ifrs9_cockpit/models/gnn_pd_model.py
```

**Tests des modules supprimés** :
```
rm ifrs9_cockpit/tests/test_dashboard.py
rm ifrs9_cockpit/tests/test_virtual_cro.py
rm ifrs9_cockpit/tests/test_causal.py
rm ifrs9_cockpit/tests/smart_test_selector.py
```

**Export** (pas utilisé pour le mémoire — les tableaux sortent des scripts racine) :
```
rm -rf ifrs9_cockpit/export/
```

**NE PAS supprimer** :
- `hmm_regime.py` (réintégré à l'étape 3)
- Les scripts racine (.py à la racine — outils d'analyse pour le mémoire)

**Après suppression** : grep pour trouver tous les imports cassés et les nettoyer :
```bash
# Chercher les imports des modules supprimés
grep -rn "virtual_cro\|ai_analyst\|causal\|contagion\|vrp\|conformal\|compliance_gates\|gnn_pd\|dashboard\|audit_trail\|from ifrs9_cockpit.app\|from ifrs9_cockpit.api\|from ifrs9_cockpit.export" ifrs9_cockpit/ --include="*.py"
```
Retirer chaque import trouvé. Si c'est un import conditionnel (`try/except`), supprimer le bloc. Si c'est un import dur, adapter le module.

**Cas explicites à traiter** :
- `conftest.py` ligne 19 : `from ifrs9_cockpit.tests.smart_test_selector import (pytest_addoption, pytest_collection_modifyitems)` → **supprimer ces 3 lignes** (sinon 100% des tests cascadent en erreur)
- `config/__init__.py` : vérifier qu'il n'importe pas depuis les sous-modules supprimés (dashboard.py)

### 1.3 — Smoke test post-nettoyage

**AVANT de toucher à quoi que ce soit d'autre**, valider que le code restant tourne :
```bash
pytest --tb=short 2>&1 | tail -20
```
Si des tests cassent à cause d'imports des modules supprimés mal nettoyés, les corriger MAINTENANT. C'est le filet de sécurité — ne pas continuer si ce smoke test échoue.

### 1.4 — Porter le modèle pe-bc dans l'architecture 14-couches

#### 1.4.1 — Config : rien à faire
Les 5 `SectorConfig` sont DÉJÀ dans `config/sectors.py` de 14-couches (lignes 99-245). Les `SECTORS`, `SECTOR_NAMES`, `SEGMENTS` sont déjà exportés par `config/__init__.py`. Vérifier que TOUTES les constantes de pe-bc/config.py existent dans le config/ package de 14-couches. Si des constantes manquent, les ajouter dans le sous-module approprié.

**Vérification nécessaire** : Comparer exhaustivement les exports de pe-bc/config.py avec ceux de 14-couches/config/__init__.py. Lister toute constante manquante.

#### 1.4.2 — Optimizer pe-bc (BL-CVaR 10 cellules)
Le comparator pe-bc a un optimiseur BL-CVaR sur 10 cellules (5 sect. × 2 canaux). Le comparator 14-couches a un optimiseur BL-CVaR sur N classes (14 classes). **Les deux doivent coexister.**

**Action** : Créer `ifrs9_cockpit/engine/comparator/optimizer_pebc.py` qui contient l'optimiseur 10 cellules extrait du pe-bc/comparator.py. Ce fichier doit :
- Contenir une classe `PebcOptimizerMixin` (ou fonctions standalone)
- Importer depuis `ifrs9_cockpit.config` (pas depuis config.py monolithique)
- Utiliser `SECTORS` (pas `ASSET_CLASSES`)
- Exposer `optimize_allocation_pebc()` retournant le même format que pe-bc

**Méthodes à extraire** du pe-bc/comparator.py (lignes 875-1500+) :
- `_build_corr_matrix_10()` (ligne 886)
- `_spread_compression()` (ligne 983)
- `_compute_cvar()` (ligne 1013)
- `_softmax_weights_10()` (ligne 1049)
- `_softmax_10()` (ligne 1075)
- `optimize_allocation()` → renommer `optimize_allocation_pebc()` (ligne 1103)
- `_stress_intensity()` (ligne 1404)
- `_asymmetric_illiquidity()` (ligne 1421)
- `_asymmetric_vol_multiplier()` (ligne 1436)
- `_bl_confidence()` (ligne 1450)
- `_asymmetric_pe_band()` (ligne 1468)
- `_adaptive_step()` (ligne 1490)
- `_adjust_for_lcr()` (ligne 1495)
- `_adjust_for_nsfr()`
- `_adjust_for_irrbb()`
- `_enforce_market_caps()` (ligne 1560)

**NOTE** : Le `comparator.py` pe-bc (1892 l.) et le `comparator.py` pré-refactoring de 14-couches sont **identiques** (diff = 0 lignes). Le code 14-couches est la version refactorisée en package. Les méthodes asymétriques (`_stress_intensity`, `_bl_confidence`, etc.) existent déjà dans `optimizer.py` de 14-couches. L'optimizer_pebc.py ne doit contenir que les méthodes **spécifiques au modèle 10 cellules** (celles qui utilisent `SECTORS` et la boucle 5+5). Les méthodes communes (asymétrie, CVaR Monte Carlo) peuvent être importées depuis le module existant.

#### 1.4.3 — Métriques pe-bc
Les métriques du comparator pe-bc (lignes 238-874) qui n'existent pas dans le comparator/ package de 14-couches doivent être identifiées et portées.

**Vérification nécessaire** : Diff méthodique des méthodes de `PortfolioComparator` entre pe-bc et 14-couches. Lister :
- Méthodes identiques (déjà couvertes) → rien à faire
- Méthodes modifiées (logique similaire, signatures différentes) → garder les deux
- Méthodes absentes de 14-couches → porter dans un nouveau mixin

#### 1.4.4 — Engine modules
Les modules suivants existent sur LES DEUX branches et sont gardés (importés par training ou comparator) :
- `gflownet.py` (620 l.) — importé par training + comparator/sensitivity
- `rmt.py` (180 l.) — importé par comparator/optimizer
- `signatures.py` (290 l.) — importé par training
- `sobol_analysis.py` (287 l.) — importé par training
- `tda.py` (331 l.) — importé par training

Vérifier que la version 14-couches de chacun est identique ou supérieure à celle de pe-bc. Si identique → rien à faire. Si pe-bc a des améliorations → merger manuellement.

#### 1.4.5 — Générateur de données
- pe-bc utilise `ifrs9_cockpit/data/generator.py` qui produit un 4-tuple `(df_credit, df_pe, df_history, df_balance_sheet)` pour les 5 secteurs
- 14-couches utilise `ifrs9_cockpit/synthetic_generator/` package qui génère les positions pour les 14 classes
- **Les deux doivent coexister** : le générateur pe-bc pour le modèle 5-secteurs, le synthetic_generator pour le modèle 14-classes.

### 1.5 — Porter les tests pe-bc

#### 1.5.1 — Adapter les imports
Les tests pe-bc importent depuis `ifrs9_cockpit.config` (qui résolvait vers config.py monolithique). Sur unified, `ifrs9_cockpit.config` résout vers `config/__init__.py` qui re-exporte tout. **Les imports devraient donc fonctionner sans modification** grâce aux re-exports.

**Vérification nécessaire** : Faire un `grep -r "from ifrs9_cockpit.config" ifrs9_cockpit/tests/` sur pe-bc et vérifier que chaque symbole importé est re-exporté par le `config/__init__.py` de 14-couches.

#### 1.5.2 — Adapter les imports comparator
Les tests pe-bc importent `from ifrs9_cockpit.engine.comparator import PortfolioComparator`. Sur 14-couches, le comparator est un package. Vérifier que `PortfolioComparator` est exporté par `comparator/__init__.py`.

#### 1.5.3 — Marqueurs pytest
Ajouter des marqueurs pour isoler les deux suites de tests :

```python
# Dans conftest.py
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "pebc: Tests du modèle pe-bc (5 secteurs × 2 canaux)")
    config.addinivalue_line("markers", "fourteen: Tests du modèle 14-couches (14 classes d'actifs)")
    config.addinivalue_line("markers", "shared: Tests partagés (invariants, config, etc.)")
```

**Fichiers tests pe-bc** (à marquer `@pytest.mark.pebc`) :
- `test_comparator.py` (pe-bc version → renommer `test_comparator_pebc.py`)
- `test_config.py` (pe-bc version → fusionner avec celui de 14-couches ou renommer)
- `test_cro_analytics.py` (pe-bc version → comparer et fusionner)
- `test_dashboard.py` → SUPPRIMER (pas de dashboard)
- `test_generator.py` (pe-bc version)
- `test_governance.py` (pe-bc version → comparer)
- `test_invariants.py` (pe-bc version → comparer)
- `test_lgd_ead.py` → probablement shared
- `test_metrics.py` → probablement shared
- `test_pd_model.py` → probablement shared
- `test_pe.py` → probablement shared
- `test_staging_ecl.py` → probablement shared

**Fichiers tests 14-couches** (à marquer `@pytest.mark.fourteen`) :
- `test_multi_asset.py`
- `test_comparator.py` (14-couches version)
- Tous les `test_*_positions.py` (12 fichiers)
- `test_hmm_rmt_wiring.py`
- ~~test_virtual_cro.py~~ SUPPRIMÉ
- ~~test_causal.py~~ SUPPRIMÉ

**Fichiers tests partagés** (à marquer `@pytest.mark.shared`) :
- `test_pd_model.py`, `test_pe.py`, `test_lgd_ead.py`, `test_staging_ecl.py`, `test_metrics.py`
- `test_training.py`

#### 1.5.4 — Résolution des conflits de nommage

**Principe** : ne PAS dupliquer des centaines de lignes de tests identiques. Diff chaque fichier, ne porter que les tests **spécifiques** pe-bc qui n'existent pas déjà sur 14-couches.

Les fichiers suivants existent sur LES DEUX branches avec des contenus DIFFÉRENTS :
- `test_comparator.py` : pe-bc=1121 l. vs 14-couches=1523 l.
- `test_config.py` : pe-bc=664 l. vs 14-couches=668 l.
- `test_cro_analytics.py` : pe-bc=844 l. vs 14-couches=970 l.
- `test_governance.py` : pe-bc=425 l. vs 14-couches=905 l.
- `test_invariants.py` : pe-bc=498 l. vs 14-couches=521 l.
- `test_generator.py` : pe-bc=306 l. vs 14-couches=307 l.

**Stratégie** :
1. **test_comparator.py** : Garder la version 14-couches. Extraire UNIQUEMENT les tests pe-bc qui testent l'optimiseur 10 cellules → nouveau fichier `test_optimizer_pebc.py`, marqué `@pytest.mark.pebc`.
2. **test_config.py** : Diff les deux. Les tests qui valident SECTORS (communs) sont déjà couverts. Ajouter seulement les tests pe-bc manquants dans le fichier 14-couches existant, avec `@pytest.mark.pebc` sur les tests spécifiques.
3. **test_cro_analytics.py** : Diff et fusionner uniquement les tests non-redondants.
4. **test_governance.py** : **CAS SPÉCIAL** — la version pe-bc importe `contagion` et `vrp` (modules supprimés). **Retirer ces tests avant portage.** Ne porter que les tests governance qui ne dépendent pas de modules supprimés.
5. **test_invariants.py** : Diff et fusionner — les invariants mathématiques sont compatibles.
6. **test_generator.py** : Diff — quasi identiques (306 vs 307 l.). Garder la version 14-couches sauf si pe-bc a un test spécifique au 4-tuple `generate_dataset()` pour les 5 secteurs.

### 1.6 — Garde-fous d'indépendance

#### 1.6.1 — Tests d'isolation fonctionnels (pas d'introspection AST)
Créer `ifrs9_cockpit/tests/test_model_isolation.py` :

L'isolation se vérifie **par les résultats**, pas par inspection du code source. Si quelqu'un mélange les modèles, les dimensions ne matcheront pas et les tests fonctionnels casseront naturellement.

```python
"""Tests d'isolation fonctionnels entre pe-bc et 14-couches.

Vérifie que chaque modèle produit des résultats cohérents, indépendamment.
"""
import pytest

@pytest.mark.pebc
class TestPebcIsolation:
    def test_pebc_pipeline_end_to_end(self, compare_results):
        """pe-bc produit un résultat BL-CVaR 10 cellules cohérent."""
        comp = compare_results
        result = comp.optimize_allocation_pebc()
        assert result["method"] == "BL-CVaR-10C"
        assert len(result["class_weights"]) == 10  # 5 secteurs × 2 canaux

    def test_pebc_uses_sectors(self):
        """pe-bc s'appuie sur 5 SECTORS, pas sur ASSET_CLASSES."""
        from ifrs9_cockpit.config import SECTORS
        assert len(SECTORS) == 5

@pytest.mark.fourteen
class TestFourteenIsolation:
    def test_fourteen_pipeline_end_to_end(self, compare_results):
        """14-couches produit un résultat BL-CVaR N-classes cohérent."""
        comp = compare_results
        result = comp.optimize_allocation()
        assert result["method"] != "BL-CVaR-10C"
        assert len(result["class_weights"]) == 14

    def test_fourteen_uses_asset_classes(self):
        """14-couches s'appuie sur 14 ASSET_CLASSES."""
        from ifrs9_cockpit.config import ASSET_CLASSES
        assert len(ASSET_CLASSES) == 14
```

#### 1.6.2 — Configuration pytest.ini
```ini
[pytest]
markers =
    pebc: Tests du modèle pe-bc (5 secteurs × 2 canaux)
    fourteen: Tests du modèle 14-couches (14 classes d'actifs)
    shared: Tests partagés entre les deux modèles
    slow: Tests lents (standalones)

# Commandes utiles :
# pytest -m pebc        → lance uniquement les tests pe-bc
# pytest -m fourteen    → lance uniquement les tests 14-couches
# pytest -m shared      → lance uniquement les tests partagés
# pytest                → lance TOUS les tests
```

#### 1.6.3 — CI guard (optionnel)
Un pre-commit hook ou CI check qui vérifie :
1. Tout nouveau fichier `test_*.py` doit contenir au moins un marqueur `pebc`/`fourteen`/`shared`
2. `optimizer_pebc.py` ne doit pas importer `ASSET_CLASSES`
3. `optimizer.py` ne doit pas hardcoder `n=10`

### 1.7 — Validation finale Étape 1+2

Après toutes les modifications :
```bash
# Tests pe-bc
pytest -m pebc -v

# Tests 14-couches
pytest -m fourteen -v

# Tests partagés
pytest -m shared -v

# Tests d'isolation
pytest ifrs9_cockpit/tests/test_model_isolation.py -v

# TOUS les tests
pytest -v
```

**Critère de succès** : TOUS les tests passent, y compris les tests d'isolation.

---

## ÉTAPE 3 : RÉINTÉGRER HMM

### 3.1 — État actuel
`ifrs9_cockpit/engine/hmm_regime.py` (767 lignes) existe sur TOUTES les branches mais n'est JAMAIS importé. C'est un détecteur de régime HMM (Hamilton 1989) avec 3 états : Contraction, Recovery, Expansion.

### 3.2 — Intégration
L'HMM doit être câblé dans le pipeline partagé (utilisable par les deux modèles) :

1. **Vérifier** que `hmm_regime.py` est déjà fonctionnel (pas de syntax errors, imports valides)
2. **Câbler** dans le pipeline commun :
   - L'HMM fournit un `regime_state` (0/1/2) utilisable par les deux optimiseurs
   - L'optimiseur pe-bc peut moduler son `cvar_alpha` selon le régime
   - L'optimiseur 14-couches aussi (déjà câblé sur `b1b3c25` — récupérer cette logique)
3. **Tests** : Le fichier `test_hmm_rmt_wiring.py` (146 l.) existe sur 14-couches — le réutiliser et l'enrichir

### 3.3 — Commit de référence
Le commit `b1b3c25` ("câbler HMM cvar_alpha dynamique et RMT Marchenko-Pastur dans l'optimiseur BL-CVaR") contient la logique d'intégration HMM+RMT. Cherry-picker ou s'en inspirer.

---

## RISQUES ET PRÉCAUTIONS

### Risques identifiés
1. **Import breakage** : pe-bc teste `from ifrs9_cockpit.config import X` où X vient de config.py. Sur unified, X doit être re-exporté par config/__init__.py. → **Vérification exhaustive nécessaire**
2. **Comparator name collision** : pe-bc a `engine/comparator.py` (fichier), 14-couches a `engine/comparator/` (dossier). Ils ne peuvent PAS coexister. → **On garde le dossier** (14-couches) et on ajoute `optimizer_pebc.py` dedans.
3. **Config.py monolithique** : Si un module pe-bc importe un symbole qui n'est pas re-exporté → import error. → **Lister exhaustivement les exports**
4. **generate_dataset()** : Le 4-tuple doit rester compatible. Vérifier que la signature est identique sur les deux branches.
5. **Tests flaky** : Les tests Monte Carlo (BL-CVaR) peuvent être non-déterministes. → **Seed fixe** via `RANDOM_SEED`
6. **Migration polars — scope** : pe-bc est 100% pandas. Migrer l'optimiseur est obligatoire (cohérence moteur), mais migrer TOUS les models/ d'un coup est ambitieux. La Phase D peut être itérative (utiliser `frame_compat` en bridge temporaire pour les models/ si la migration complète prend trop de temps).
7. **Migration polars — API differences** : polars n'a pas `.loc[]`, `.iloc[]`, `.values`, `.apply()` au sens pandas. Chaque occurrence doit être réécrite avec `.filter()`, `.select()`, `.to_numpy()`, `.map_elements()`. C'est un travail mécanique mais volumineux (~290 refs sur pe-bc).
8. **TabNet/scikit-learn** : Les modèles ML (pd_model.py, lgd_model.py) passent des DataFrames à `.fit()` / `.predict()`. scikit-learn attend numpy ou pandas. → Garder pandas en entrée/sortie des modèles ML, convertir en polars uniquement pour le pipeline interne. Le `frame_compat.py` sert exactement à ça.
9. **Scénarios 11→12** : Les tests pe-bc qui hardcodent `len(PREDEFINED_SCENARIOS) == 11` ou boucle sur 11 scénarios casseront. → grep + fix systématique.

### Précautions
- **Ne JAMAIS supprimer un fichier sans vérifier qu'il n'est importé nulle part** (grep avant rm)
- **Garder la branche `unified-backup`** intacte tant que tous les tests ne passent pas
- **Committer fréquemment** avec des messages descriptifs
- **Ne pas modifier la logique métier** — on déplace et on réorganise, on ne change pas les calculs

---

## CHECKLIST DE VALIDATION

### Phase A — Fondations (nettoyage agressif)
- [ ] Branche `unified` créée depuis `14-couches` (14382ab)
- [ ] Branche `unified-backup` créée (filet de sécurité)
- [ ] Worktrees supprimés (.claude/worktrees/)
- [ ] Code mort supprimé (conformal, compliance_gates, diag_v44, synthetic_generator_v4)
- [ ] Dashboard supprimé (dashboard/, app.py, dash_app.py, assets/)
- [ ] Virtual CRO supprimé (virtual_cro/, analytics/virtual_cro.py)
- [ ] AI Analyst supprimé (analytics/ai_analyst/)
- [ ] Causal supprimé (causal/)
- [ ] API supprimé (api.py)
- [ ] Data fetchers supprimés (data/fetch_*.py)
- [ ] Modules dashboard-only supprimés (contagion.py, vrp.py)
- [ ] Modèle mort supprimé (gnn_pd_model.py)
- [ ] config/dashboard.py supprimé
- [ ] Tests des modules supprimés retirés (test_dashboard, test_virtual_cro, test_causal, smart_test_selector)
- [ ] export/ supprimé (audit_trail_latex = export dashboard déguisé)
- [ ] conftest.py : import smart_test_selector retiré (lignes 19-22)
- [ ] config/__init__.py : import dashboard retiré si présent
- [ ] TOUS les imports cassés nettoyés (grep + fix)
- [ ] **SMOKE TEST** : `pytest --tb=short` PASS avant de continuer

### Phase B — Portage pe-bc
- [ ] Exports config pe-bc vs config/__init__.py vérifiés exhaustivement
- [ ] Constantes manquantes portées dans config/ sous-modules
- [ ] pe-bc/config/calibration.py contenu vérifié et intégré si manquant
- [ ] pe-bc/config/generators.py contenu vérifié et intégré si manquant
- [ ] optimizer_pebc.py créé dans engine/comparator/ (extrait de pe-bc)
- [ ] optimizer_pebc.py migré en polars natif
- [ ] PebcOptimizerMixin câblé dans comparator/__init__.py
- [ ] PortfolioComparator expose optimize_allocation_pebc() — vérifié par `python -c "...assert hasattr(...)"`
- [ ] utils/scorecard.py porté (si utilisé)

### Phase C — Tests
- [x] Diff de chaque fichier test conflictuel — extraction des tests spécifiques pe-bc uniquement
- [x] test_governance.py pe-bc : tests contagion/vrp retirés avant portage (modules supprimés)
- [x] test_optimizer_pebc.py créé (66 tests optimiseur 10 cellules)
- [x] Assertions 11 scénarios → 12 déjà à jour (Volcker présent depuis 14-couches)
- [x] Marqueurs pytest ajoutés (pebc/fourteen/shared) sur TOUS les fichiers (27 fichiers)
- [x] test_model_isolation.py créé (tests fonctionnels coexistence dual-optimizer)
- [x] pytest.ini mis à jour avec marqueurs
- [x] conftest.py mis à jour avec --all/--slow flags + collection modifiers
- [x] compute_raroc_eva() enrichi avec profit_rate (nécessaire pour optimizer_pebc)

### Phase D — Migration polars
- [ ] data/generator.py migré en polars
- [ ] models/pd_model.py migré (ou frame_compat bridge)
- [ ] models/lgd_model.py migré
- [ ] models/ead_model.py migré
- [ ] models/pe_model.py migré
- [ ] models/woe.py migré
- [ ] training/train.py migré
- [ ] ~~export/audit_trail_latex.py~~ SUPPRIMÉ (Phase A)

### Phase E — Validation
- [ ] `pytest -m pebc -v` → PASS
- [ ] `pytest -m fourteen -v` → PASS
- [ ] `pytest -m shared -v` → PASS
- [ ] `pytest ifrs9_cockpit/tests/test_model_isolation.py -v` → PASS
- [ ] `pytest -v` (TOUS) → PASS
- [ ] Aucun import pandas dans engine/ (sauf frame_compat)
- [ ] Aucun import pandas dans engine/comparator/

### Phase F — HMM
- [ ] hmm_regime.py vérifié (syntax, imports)
- [ ] hmm_regime.py câblé dans le pipeline partagé
- [ ] test_hmm_rmt_wiring.py enrichi et PASS
- [ ] Commit final avec message descriptif

---

## ORDRE D'EXÉCUTION RECOMMANDÉ

### Phase A — Nettoyage agressif (commits atomiques)
1. Créer les branches (`unified` depuis `14-couches` + `unified-backup`)
2. Supprimer code mort (worktrees, conformal, compliance_gates, diag_v44, synthetic_generator_v4)
3. Supprimer dashboard (dashboard/, app.py, dash_app.py, assets/)
4. Supprimer modules inutiles au mémoire (virtual_cro/, analytics/ai_analyst/, analytics/virtual_cro.py, causal/, api.py, data/fetch_*.py, models/gnn_pd_model.py, config/dashboard.py, engine/contagion.py, engine/vrp.py, export/)
5. Supprimer tests des modules supprimés (test_dashboard, test_virtual_cro, test_causal, smart_test_selector)
6. `grep` tous les imports cassés et les nettoyer — en particulier :
   - **conftest.py ligne 19** : retirer `from ifrs9_cockpit.tests.smart_test_selector import ...` (sinon 100% des tests cascadent)
   - **config/__init__.py** : retirer import de `dashboard` si présent
7. Commit "chore: strip to memoir-essential code (~24K lignes supprimées)"
8. **SMOKE TEST** : `pytest --tb=short` — tout doit passer. Si ça casse, corriger AVANT de continuer.

### Phase B — Portage pe-bc (le coeur de la fusion)
9. Vérifier exhaustivement les exports config pe-bc vs config/__init__.py — lister les manquants
10. Vérifier contenu de pe-bc/config/calibration.py et config/generators.py — identifier ce qui manque dans 14-couches/config/
11. Porter les constantes manquantes dans les sous-modules config/ appropriés
12. Extraire l'optimiseur pe-bc → `optimizer_pebc.py` dans engine/comparator/
13. **Migrer optimizer_pebc.py de pandas vers polars**
14. **Câbler le mixin** : ajouter `PebcOptimizerMixin` dans comparator/__init__.py pour que `PortfolioComparator` expose `optimize_allocation_pebc()`
15. **Vérifier le câblage** : `python -c "from ifrs9_cockpit.engine.comparator import PortfolioComparator; assert hasattr(PortfolioComparator, 'optimize_allocation_pebc')"`
16. Vérifier que `utils/scorecard.py` est nécessaire — si oui, le porter
17. Commit "feat: port pe-bc BL-CVaR 10-cell optimizer (polars)"

**IMPORTANT** : l'étape 15 DOIT passer avant de toucher aux tests pe-bc. Sinon les tests importent une méthode qui n'existe pas.

### Phase C — Portage des tests pe-bc
18. Identifier les symboles importés par les tests pe-bc et vérifier les re-exports config/__init__.py
19. Diff chaque fichier test conflictuel — extraire UNIQUEMENT les tests spécifiques pe-bc
20. **test_governance.py pe-bc** : retirer les tests qui importent `contagion` et `vrp` (modules supprimés) avant portage
21. Créer `test_optimizer_pebc.py` avec les tests optimiseur 10 cellules
22. Adapter les assertions `== 11` scénarios vers `== 12` (Volcker ajouté)
23. Ajouter marqueurs pytest (pebc/fourteen/shared) sur TOUS les fichiers tests
24. Créer test_model_isolation.py (tests fonctionnels, pas d'introspection AST)
25. Mettre à jour pytest.ini avec les marqueurs
26. Commit "feat: merge pe-bc tests with isolation markers"

### Phase D — Migration polars (dette commune)
21. Migrer `data/generator.py` de pandas vers polars (commun aux deux modèles)
22. Migrer `models/` vers polars (ou frame_compat bridge) — pd_model, lgd_model, ead_model, pe_model, woe
23. Migrer `training/train.py` vers polars complet
24. Migrer `export/audit_trail_latex.py` si temps
25. NE PAS migrer `data/fetch_*.py` (ETL one-shot, pandas OK)
26. Commit "refactor: complete polars migration for engine and models"

### Phase E — Validation
27. `pytest -m pebc -v` → PASS
28. `pytest -m fourteen -v` → PASS
29. `pytest -m shared -v` → PASS
30. `pytest ifrs9_cockpit/tests/test_model_isolation.py -v` → PASS
31. `pytest -v` (TOUS) → PASS
32. Commit final si corrections nécessaires

### Phase F — HMM (Étape 3)
33. Vérifier hmm_regime.py (syntax, imports)
34. Câbler HMM dans le pipeline (s'inspirer de commit `b1b3c25`)
35. Enrichir test_hmm_rmt_wiring.py
36. Commit "feat: integrate HMM regime detector in shared pipeline"
