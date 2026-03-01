# Cadre Reglementaire — IFRS 9 Risk Cockpit

## 1. Normes Implementees (Priorite 1)

### 1.1 IFRS 9 B5.5.25 — Exemption Staging Souverain

**Principe** : Les instruments financiers a faible risque de credit (typiquement souverain AAA/AA zone euro) peuvent etre maintenus en Stage 1 sans evaluation SICR systematique.

**Implementation** :
- Champ `exempt_from_staging: bool` dans `AssetClassProfile`
- `staging_cliff_ecl()` retourne uniquement l'ECL 12 mois si `exempt_from_staging=True`
- Seul le profil `sovereign` est exempt (AAA/AA domestique, RW=0%)

**Reference** : IFRS 9, paragraphe B5.5.25 ; BCE/ECB Interpretive Guidance 2018.

### 1.2 CRR3 Art. 124-125 — Schedule LTV Credit Immobilier

**Principe** : Le Risk Weight des expositions immobilieres residentielles depend de la tranche LTV (Loan-to-Value), avec une grille progressive.

**Implementation** :
- Champ `ltv_distribution: Dict[float, float]` dans `AssetClassProfile`
- Fonction `mortgage_rw_blended()` applique la grille CRR3 :
  | LTV | RW |
  |-----|-----|
  | <= 50% | 20% |
  | <= 60% | 25% |
  | <= 70% | 30% |
  | <= 80% | 35% |
  | <= 90% | 50% |
  | <= 100% | 70% |
- Distribution representative FR/EU : RW pondere ~33%

**Reference** : CRR3 Art. 124-125 (effective Jan 2025).

### 1.3 CRR3 Art. 242-270 — SEC-SA Titrisation

**Principe** : Les expositions de titrisation sont ponderees selon l'approche standardisee SEC-SA, avec des RW differencies par tranche.

**Implementation** :
- Champ `securitisation_mix: Dict[str, float]` dans `AssetClassProfile`
- Fonction `securitisation_rw_blended()` applique les RW SEC-SA :
  | Tranche | RW |
  |---------|-----|
  | Senior STS | 20% |
  | Senior non-STS | 60% |
  | Mezzanine | 100% |
  | Junior | 250% |
- Mix representatif post-GFC : RW pondere ~42%

**Reference** : CRR3 Art. 242-270 ; Regulation (EU) 2017/2402 (STS Framework).

### 1.4 NSFR Basel III — Ratio de Financement Stable Net

**Principe** : NSFR = ASF / RSF >= 100%. Chaque classe d'actifs consomme du financement stable proportionnellement a son illiquidite.

**Implementation** :
- Champ `rsf_weight: float` dans `AssetClassProfile`
- Champs `nsfr_target` et `asf_deposit_coverage` dans `BaselConfig`
- Fonction `compute_nsfr()` calcule le ratio et la conformite
- Poids RSF calibres :
  | Classe | RSF |
  |--------|-----|
  | Souverain | 0% |
  | Interbancaire | 0% |
  | Trade Finance | 10% |
  | Covered Bonds | 15% |
  | Mortgage | 35% |
  | Corporate | 50% |
  | Consumer | 50% |
  | Project Finance | 85% |
  | Structured | 85% |
  | Private Equity | 100% |

**Reference** : BCBS d295 (2014) ; CRR Art. 428a-428ai.

## 2. Fonction Pivot : `effective_rw()`

Fonction de dispatch utilisee dans `balance_sheet_ecl.py`, `metrics.py`, et `optimizer.py` :
- `retail_mortgage` → `mortgage_rw_blended(ltv_distribution)`
- `structured_products` → `securitisation_rw_blended(securitisation_mix)`
- Autres → `profile.rw_crr3` (fallback plat)

## 3. Limites Connues (Priorite 2/3)

Les normes suivantes ne sont **pas implementees** dans le moteur actuel. Elles sont documentees comme limites du prototype et axes d'extension possibles.

### Priorite 2 (impact materiel, complexite moderee)

| Norme | Description | Impact |
|-------|-------------|--------|
| **CVA** (CRR3 Art. 381-386) | Credit Valuation Adjustment pour derives OTC | Absent car pas d'exposition derives dans le portefeuille synthetique |
| **Large Exposure** (CRR Art. 387-403) | Limite de concentration 25% CET1 par contrepartie | HHI name-level est un proxy, mais pas la contrainte 25% formelle |
| **Output Floor Phase-In** (CRR3 Art. 465) | Trajectoire 50%→72.5% (2025-2030) | RW IRB plancher non phase-in, utilisation du SA comme proxy |
| **IRRBB** (CRR3 Art. 84a) | Risque de taux du banking book (NII sensitivity) | Impact materiel sur mortgage/ProjFin, non modelise |

### Priorite 3 (impact limite ou couvert indirectement)

| Norme | Description | Statut |
|-------|-------------|--------|
| **EBA/GL/2020/06** | Origination & monitoring (DTI, DSTI) | Couvert indirectement par debt_ratio et credit_score |
| **EU Covered Bond Directive 2019/2162** | Cover pool eligibility, supervision | Profil parametrique simpliste (double recours via faible PD/LGD) |
| **ICC Banking Commission** | Trade Finance default survey | PD=0.8% alignee sur survey ICC 2023 |
| **Pilier 2 SREP** | Add-ons P2R / P2G au-dessus du Pilier 1 | CET1 target = 13% inclut implicitement ~2% de P2R |
| **Risque souverain migration** | Downgrade AA→A et impact RW | Pas de modele de migration de rating |
| **CRR3 Art. 208-210** | Revalorisation immobiliere periodique | LTV statique, pas de reval dynamique |
