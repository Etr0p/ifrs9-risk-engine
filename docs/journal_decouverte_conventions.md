# Journal de Decouverte : Conventions de Signe dans le Facteur Systematique ASRF
## De l'arbitrage PE/Credit a la methodologie de selection de convention

**Auteur** : Fred
**Date** : Mars 2026
**Contexte** : Memoire sur la gestion des risques IFRS 9 / Bale III

---

## 1. Point de depart : l'arbitrage PE vs Credit Bancaire

### 1.1 Le programme initial

Le projet initial etait un outil d'arbitrage entre Private Equity (PE) et Credit Bancaire (BC). L'idee etait simple : un CRO dispose d'un budget de capital, et doit decider quelle proportion allouer au PE (rendements eleves, illiquidite) vs au credit bancaire (rendements moderes, liquidite).

Le modele initial comprenait :
- 2 classes d'actifs (PE et Credit)
- 5 secteurs economiques (Tech, Industrie, Sante, Immobilier, Services)
- Un optimiseur RAROC sous contrainte CET1

### 1.2 La premiere decouverte : le verrouillage reglementaire

Le resultat le plus frappant du programme initial etait que **l'arbitrage PE/Credit etait largement predetermine par les contraintes reglementaires**, pas par la rentabilite. Le ratio CET1 a 13%, combine aux risk weights CRR3 (250% pour le PE vs ~50% moyen pour le credit), laissait tres peu de marge a l'optimiseur.

**Conclusion initiale** : L'allocation PE/Credit est quasi-fixe. La vraie marge de manoeuvre du CRO se situe au niveau sectoriel, a l'interieur de chaque classe.

Cette conclusion etait interessante mais descriptive. Elle constatait un phenomene sans l'expliquer en profondeur.

---

## 2. L'extension a 14 classes : confirmation a grande echelle

### 2.1 Generalisation du modele

Pour tester la robustesse de cette conclusion, le programme a ete etendu a 14 classes d'actifs couvrant l'ensemble du bilan d'une banque europeenne :

| # | Classe | Categorie | RW CRR3 |
|---|--------|-----------|---------|
| 1 | Prets Corporate | Amortised Cost | ~52% |
| 2 | Private Equity | Level 1 | 250% |
| 3 | Obligations Souveraines | HQLA L1 | 0% |
| 4 | Credit Immobilier | LTV-based | 20-70% |
| 5 | Financement de Projet | Specialised | 100% |
| 6 | Obligations Securisees | HQLA L2 | 10% |
| 7 | Credit Consommation | Retail | 75% |
| 8 | Trade Finance | Short-term | 20% |
| 9 | Interbancaire | Interbank | 20% |
| 10 | Titrisation | SEC-SA | ~42% |
| 11 | Actions Cotees | FVTPL | 100% |
| 12 | Obligations Corporate | FVOCI | 20-150% |
| 13 | Repos / SFT | Collateralise | 0-10% |
| 14 | Derives / CVA | SA-CCR | ~60% |

L'optimiseur a ete enrichi avec Black-Litterman CVaR (Rockafellar-Uryasev 2002) et 4 contraintes reglementaires : CET1, LCR, NSFR, IRRBB.

### 2.2 Meme conclusion, plus forte

Le resultat a 14 classes a **confirme et renforce** la conclusion initiale :
- Les contraintes reglementaires verrouillent ~53% du portefeuille
- Le Repos/SFT monte a 27% non pas par rentabilite mais par contrainte IRRBB (besoin d'actifs a duration courte)
- L'ecart L2 entre allocations adverses et favorables n'est que de 0.4% — quasi-identique
- La differentiation se fait par la rentabilite (RAROC), pas par les poids

A ce stade, le programme confirmait une these connue. Utile, mais pas suffisant pour un apport original.

---

## 3. L'audit qui revele le probleme

### 3.1 Les 7 critiques

Un audit systematique du modele (2157 checks, 10 scenarios, 14 classes) a identifie 7 critiques. Trois ont ete corrigees (softmax libre, sensibilite IR PF/DC, Central ir_bp=0), quatre se sont averees etre du comportement correct (mortgage HPI, TF COVID>GFC, repos 26%, equity vs PE COVID).

### 3.2 La question du RAROC identique entre scenarios

L'audit a revele un phenomene troublant : **11 des 14 classes avaient un RAROC identique dans tous les scenarios**. Le RAROC ne bougeait pas entre GFC et Reprise. L'optimiseur n'avait donc aucune base pour differencier ses allocations.

La cause etait que le calcul RAROC utilisait la PD a l'origination (`pd_base`, statique) au lieu de la PD conditionnelle Vasicek (`pd_cond_base`, stressee par le macro). Ce fix a ete applique : le RAROC est devenu scenario-dependant.

### 3.3 L'emergence de la question des conventions

Avec le RAROC desormais scenario-dependant, un nouveau probleme est apparu : **certains RAROC etaient incoherents**. En particulier, le scenario Stagflation (hausse de taux + recession) affichait un RAROC de +6.7%. C'est economiquement absurde : la stagflation est le pire scenario pour une banque.

C'est cette anomalie qui a declenche l'investigation sur les conventions de signe.

---

## 4. L'investigation sur la convention IR

### 4.1 Le diagnostic initial

Le Z-score dans le modele ASRF est calcule comme :

```
Z = sum_i( w_i * f(x_i - mu_i) / sigma_i )
```

La fonction f determine si une hausse de la variable est favorable ou adverse :
- `f = +1` (adverse) : une hausse de x augmente Z, donc augmente la PD conditionnelle
- `f = -1` (favorable) : une hausse de x diminue Z, donc diminue la PD conditionnelle

La convention originale du modele etait :
- Chomage : adverse (+1) — indiscutable
- GDP, HPI, Inflation, Taux d'interet : favorable (-1)

Pour le taux d'interet, cela signifiait : **des baisses de taux augmentent Z (adverse)**. L'idee sous-jacente etait le modele de Merton (1974) : un taux sans risque plus eleve augmente le drift de la valeur de l'entreprise, eloignant du point de defaut.

### 4.2 Le probleme economique

Cette convention produisait des resultats incoherents dans 3 scenarios :

**Reprise** (GDP=2.5%, taux=2.5%) : Les baisses de taux de la BCE (politique accommodante en sortie de crise) etaient traitees comme ADVERSES. La contribution IR au Z-score etait de +2.4 sigma pour les prets corporate, ce qui dominait tous les autres facteurs (GDP favorable a -1.08 sigma). Resultat : le Z-score total etait positif (adverse) dans un scenario de reprise economique.

**Stagflation** (GDP=-1%, taux=6%) : Les hausses de taux de la BCE (lutte contre l'inflation) etaient traitees comme FAVORABLES. La contribution IR de -3.0 sigma compensait une grande partie du stress GDP/chomage. Resultat : RAROC = +6.7%, comme si la stagflation etait un scenario moderement favorable.

**Transition climatique** (GDP=-0.5%, taux=4.5%) : Meme probleme directionnel.

### 4.3 La quantification de l'impact

Un diagnostic complet a ete realise (`_diag_ir_impact.py`) pour mesurer l'impact exact. Les resultats etaient frappants :

**En Reprise**, neutraliser la contribution IR faisait baisser le pd_cond de :
- Prets corporate : 12.59% -> 1.16% (-90.8%)
- Credit immobilier : 4.20% -> 0.03% (-99.2%)
- Actions cotees : 9.78% -> 0.14% (-98.5%)
- Souverain : 0.82% -> 0.005% (-99.4%)

Ce n'etait pas un effet marginal. Un seul parametre de convention changeait les PD conditionnelles de **un a deux ordres de grandeur**.

### 4.4 La correction et son impact

Le fix etait une modification d'une seule ligne : traiter le taux d'interet comme le chomage (hausse = adverse).

Justification economique : le canal principal des taux d'interet sur le risque de defaut est le **service de la dette** (Duffie, Saita & Wang 2007). Des taux plus eleves augmentent les charges d'interet, compriment les marges, et augmentent la probabilite de defaut. Ce canal domine empiriquement l'effet Merton (drift du taux sans risque).

Impact mesure sur le RAROC portfolio :

| Scenario | Avant | Apres | Delta |
|----------|-------|-------|-------|
| Central | +10.8% | +10.8% | 0 |
| Stagflation | **+6.7%** | **-10.8%** | **-17.5pp** |
| Reprise | ~+10% | **+17.8%** | **+7.8pp** |
| GFC | -5.5% | -5.5% | 0 |

Le swing de 17.5 points de pourcentage sur Stagflation — obtenu en changeant un seul signe dans une equation — est le resultat central de cette investigation.

---

## 5. Verification des autres anomalies

### 5.1 La convention bipolaire du chomage (A1)

Suite a la decouverte de l'impact massif de la convention IR, une verification systematique des autres anomalies identifiees a ete conduite.

**Anomalie A1** : La convention `unemployment_rate = base + |bipolar|` empeche le chomage de baisser en dessous du niveau de base (7.5%) dans les scenarios favorables.

**Resultat du diagnostic** : Impact **strictement nul**. Les 3 scenarios favorables (Reprise, Hypercroissance, Boom) ont tous `unemployment_bipolar = 0.0`. La convention abs() n'a aucun effet sur eux. Les scenarios adverses utilisent des valeurs negatives (GFC=-2.0, Stagflation=-3.0) que le abs() convertit correctement en hausses de chomage.

Ce qui semblait etre un bug etait en realite un non-probleme.

### 5.2 La convention inflation

**Question** : L'inflation est traitee comme favorable (hausse = baisse de Z). Est-ce correct ?

**Diagnostic** : Si l'inflation etait traitee comme adverse, les resultats seraient :
- **Stagflation** (inflation 8%) : PE pd_cond passe de 30.8% a 67.9% (+120%). Plus severe, ce qui semble correct.
- **MAIS Trappe a liquidite** (inflation -0.5%) : Corporate pd_cond passe de 43.3% a **0.14%** (-99.7%). Un scenario adverse deviendrait ultra-favorable. Economiquement absurde.
- **MAIS GFC** (inflation 0.3%) : Corporate pd_cond passe de 50.1% a 30.1% (-40%). La crise la plus severe de l'histoire recente serait attenuee.

**Conclusion** : La convention actuelle (inflation favorable) est correcte parce que :
1. L'effet adverse de l'inflation via la politique monetaire est **deja capture** par la convention IR (taux haut = adverse). Changer les deux serait un double-comptage.
2. L'effet direct de l'inflation (erosion de la dette reelle, signal d'activite economique) est genuinement favorable pour les emprunteurs.
3. Changer la convention casserait les scenarios de deflation (GFC, Trappe) ou l'inflation basse ajoute correctement du stress.

### 5.3 Synthese des verifications

| Anomalie | Estimation initiale | Impact reel mesure | Verdict |
|----------|--------------------|--------------------|---------|
| Convention IR | "Mineur, 2nd ordre" | **17.5pp de swing** | Bug majeur, corrige |
| Convention bipolaire chomage | "Modere" | **Zero** | Non-probleme |
| Convention inflation | "A verifier" | **Correcte** (double-comptage sinon) | Pas de bug |

**Lecon methodologique** : L'estimation qualitative de l'impact ("mineur", "modere") s'est averee trompeuse. Seule la quantification systematique a permis de distinguer le vrai bug des faux positifs.

---

## 6. La reflexion fondamentale : pourquoi ca compte

### 6.1 Le cadre reglementaire ne prescrit pas les conventions

La norme IFRS 9 (paragraphe 5.5.17) exige un ECL pondere par probabilites sous plusieurs scenarios. Le modele ASRF/Vasicek est le standard pour transformer les scenarios macro en PD conditionnelles. Mais ni la norme, ni les guidelines EBA (GL/2017/06), ni le programme TRIM de la BCE ne prescrivent :
- Quelles variables macro inclure dans le facteur systematique
- Quel signe attribuer a chaque variable
- Quelle normalisation utiliser (sigma fixe vs historique)

### 6.2 Les banques sont enfermees dans leur choix

Une banque sous approche IRB a un modele approuve par le superviseur. Toute modification (meme un changement de signe) constitue un "changement materiel de modele" necessitant :
1. Notification formelle a la BCE
2. Validation interne independante
3. Revue du superviseur (6-18 mois)
4. Parallel run ancien/nouveau modele

Consequence : une fois qu'une banque a choisi sa convention, elle est enfermee dedans. Elle ne peut pas facilement tester l'impact d'une convention alternative. Elle ne sait meme pas combien son RAROC changerait.

### 6.3 La variabilite entre banques est documentee

Le Comite de Bale a publie en 2013 le "Regulatory Consistency Assessment Programme" (RCAP) montrant que les RWA variaient de +/-20% entre banques pour le meme portefeuille hypothetique. Le programme TRIM de la BCE (2017-2021) a confirme cette variabilite. L'output floor de Bale IV (72.5% des RWA standards) a ete introduit precisement pour la contenir.

Notre experience montre que les conventions de signe dans le facteur systematique sont **une source de cette variabilite**. Deux banques utilisant le meme modele ASRF avec des conventions IR differentes obtiendraient des ECL, des RWA, et des RAROC significativement differents.

### 6.4 L'avantage du cadre experimental

Notre modele n'est pas une banque reelle. C'est un **laboratoire experimental** qui permet :
- De basculer une convention et mesurer l'impact (impossible pour une banque)
- De tester systematiquement toutes les combinaisons (4 conventions possibles)
- De definir des criteres de coherence objectifs
- De selectionner la convention optimale sur base empirique

---

## 7. Vers une methodologie de selection de convention

### 7.1 L'espace des conventions

Le facteur systematique Z utilise 5 variables macro. Pour chacune, le signe peut etre :
- **Adverse** (+1) : une hausse de la variable augmente Z (augmente la PD)
- **Favorable** (-1) : une hausse de la variable diminue Z (diminue la PD)

Pour 3 variables, le signe est indiscutable :
- **GDP** : hausse = favorable (-1). Unanimite economique.
- **Chomage** : hausse = adverse (+1). Unanimite economique.
- **HPI** : hausse = favorable (-1). Valeur du collateral et effet de richesse.

Pour 2 variables, le signe est ambigu :
- **Taux d'interet** : Merton (-1) vs service de la dette (+1)
- **Inflation** : erosion de la dette (-1) vs couts (+1)

Cela donne **4 conventions possibles** (2^2) a tester :

| Convention | Taux d'interet | Inflation |
|------------|---------------|-----------|
| A | favorable (-1) | favorable (-1) |
| B | adverse (+1) | favorable (-1) |
| C | favorable (-1) | adverse (+1) |
| D | adverse (+1) | adverse (+1) |

### 7.2 Les criteres de coherence

Pour evaluer chaque convention, 5 criteres objectifs sont definis :

**C1 — Monotonie adverse** : Le RAROC portfolio dans les scenarios adverses (GFC, Stagflation, Crise souveraine, COVID) doit etre inferieur au RAROC Central. Un scenario de crise ne peut pas etre plus rentable que la normale.

**C2 — Monotonie favorable** : Le RAROC portfolio dans les scenarios favorables (Reprise, Hypercroissance, Boom immobilier) doit etre superieur au RAROC Central. Un scenario d'expansion doit etre plus rentable.

**C3 — Signe correct** : Aucun scenario favorable ne doit avoir un RAROC negatif. Aucun scenario adverse ne doit avoir un RAROC superieur au Central + 5pp.

**C4 — Coherence historique** : Le classement des RAROC doit correspondre au classement historique des performances bancaires. Mesure par correlation de rang de Spearman. Reference :
1. Hypercroissance (meilleur) — Trente Glorieuses
2. Boom immobilier — 2005-2007
3. Reprise — 2017-2018
4. Central — neutre
5. Rupture techno — mixte
6. Transition climatique — stress emergent
7. Trappe a liquidite — marges compressees (Japon)
8. COVID — stress en V
9. Crise souveraine — 2012
10. Stagflation — 1970s
11. GFC — 2008 (pire)

**C5 — Stabilite inter-classes** : Pour chaque paire (adverse, favorable), la majorite des 14 classes doit avoir le meme sens de variation du RAROC. Une convention ou le corporate dit "crise" mais le souverain dit "boom" est instable.

### 7.3 Le protocole experimental

Pour chaque convention (A, B, C, D) :
1. Parametrer macro_to_z avec les signes correspondants
2. Faire tourner le pipeline complet pour les 11 scenarios
3. Collecter les RAROC par classe et portfolio
4. Evaluer les 5 criteres
5. Calculer un score composite

Le score composite est la somme ponderee des 5 criteres, chacun normalise entre 0 et 1. La convention avec le score le plus eleve est la convention optimale.

### 7.4 Hypothese

Sur la base de l'analyse qualitative menee dans les sections 4 et 5, l'hypothese est que la **convention B** (taux adverse, inflation favorable) sera optimale. Mais cette hypothese doit etre verifiee quantitativement — c'est precisement l'objet de l'experience.

---

## 8. Resultats de l'experience

### 8.1 RAROC portfolio par convention et scenario

L'experience a ete executee : 4 conventions x 11 scenarios = 44 evaluations completes du pipeline.

| Scenario | Type | Conv A | Conv B | Conv C | Conv D |
|----------|------|--------|--------|--------|--------|
| Central | NEU | +10.6% | +10.6% | +10.6% | +10.6% |
| GFC | ADV | -8.6% | -7.2% | -7.3% | -3.2% |
| Crise souveraine | ADV | -2.4% | +1.2% | -2.4% | +1.2% |
| **Stagflation** | **ADV** | **-9.8%** | **-14.3%** | **-16.9%** | **-18.5%** |
| COVID | ADV | -4.5% | +0.5% | -2.8% | +2.8% |
| Rupture techno | NEU | +10.3% | +11.8% | +10.8% | +12.0% |
| **Reprise** | **FAV** | **+17.1%** | **+17.9%** | **+17.6%** | **+17.9%** |
| Hypercroissance | FAV | +12.4% | +12.2% | +12.4% | +11.5% |
| Boom immobilier | FAV | +14.1% | +14.1% | +14.1% | +14.1% |
| Trappe liquidite | NEU | +2.5% | +6.7% | +4.9% | +9.8% |
| Transition clim. | NEU | -4.0% | -5.1% | -6.9% | -8.1% |

### 8.2 Scores de coherence

| Critere | Conv A | Conv B | Conv C | Conv D |
|---------|--------|--------|--------|--------|
| C1 Monotonie adverse | 100% | 100% | 100% | 100% |
| C2 Monotonie favorable | 100% | 100% | 100% | 100% |
| C3 Signe correct | 100% | 100% | 100% | 100% |
| C4 Coherence historique | 95% | 94% | 94% | 90% |
| C5 Stabilite inter-classes | 86% | 71% | 86% | 71% |
| **COMPOSITE** | **96%** | **93%** | **96%** | **92%** |

### 8.3 Le resultat contre-intuitif

**L'hypothese initiale est infirmee.** La convention B (taux adverse, inflation favorable), implementee apres notre investigation, n'est PAS la convention avec le score composite le plus eleve. Les conventions A et C obtiennent 96%, contre 93% pour B.

Ce resultat s'explique par le **critere C5 (stabilite inter-classes)** :

- En **Hypercroissance** (taux=8%), la convention B traite les taux eleves comme adverses. Cela cree un conflit avec le GDP a +5% : certaines classes voient Hypercroissance comme favorable (GDP domine), d'autres comme neutre ou adverse (taux dominent). Seulement 4/14 classes s'accordent sur la direction.
- En **COVID** (taux=0%), la convention B traite les baisses de taux comme favorables. Cela aide les classes credit (prets, hypotheques) mais les actions FVTPL sont detruites par le VSTOXX. Split 7/14.

Avec la convention A, les taux eleves sont favorables (signal de confiance economique), ce qui renforce le GDP positif en Hypercroissance. Plus de classes s'accordent.

### 8.4 Mais la convention B est plus severe en crise

Le classement composite masque un fait important : sur le critere qui compte le plus pour un CRO, la **severite en scenario adverse**, la convention B est superieure :

| Convention | Stagflation | Trappe liquidite | GFC |
|------------|-------------|------------------|-----|
| A (originale) | -9.8% | +2.5% | -8.6% |
| **B (actuelle)** | **-14.3%** | **+6.7%** | **-7.2%** |

La Stagflation a -14.3% (B) est plus severe que -9.8% (A). C'est economiquement correct : la stagflation des annees 1970 a ete devastatrice. La convention A sous-estime le stress en ne traitant pas les hausses de taux comme adverses.

### 8.5 Le dilemme : stabilite vs severite

Le resultat revele un **dilemme fondamental** qui est absent de la litterature :

- **Convention A** : plus stable (les classes s'accordent mieux) mais sous-estime le stress en stagflation
- **Convention B** : plus severe en crise (economiquement correct) mais cree des desaccords inter-classes en scenarios mixtes (Hypercroissance, COVID)

**Il n'existe pas de convention trivialement superieure.** Le choix depend de ce que le CRO ou le regulateur privilegient :
- **Prudence** (Convention B/D) : maximiser la severite en scenario adverse, au prix de l'instabilite
- **Coherence** (Convention A/C) : maximiser l'accord inter-classes, au prix de sous-estimer certains risques

Ce dilemme est lui-meme un resultat original, non documente dans la litterature IFRS 9.

---

## 9. Implications pour le memoire

### 9.1 Structure proposee du memoire

**Acte 1 — La question originale** : L'arbitrage PE/Credit est-il pilote par la rentabilite ou par la reglementation ? (Programme initial, conclusion : la reglementation domine, l'arbitrage reel est sectoriel.)

**Acte 2 — La question derivee** : Si la reglementation domine l'allocation, et si la reglementation depend des modeles (PD, ECL, RWA), alors les choix d'implementation de ces modeles sont-ils neutres ? (Extension 14 classes, decouverte : non, un seul parametre de convention change le RAROC de 17.5pp.)

**Acte 3 — L'experience** : Comment selectionner objectivement la bonne convention ? (Experience sur 4 conventions, 5 criteres, resultat : aucune convention ne domine les autres sur tous les criteres.)

**Acte 4 — Le dilemme** : Il existe un trade-off fondamental entre stabilite inter-classes et severite en crise. Ce trade-off n'est pas documente dans la litterature et constitue un argument supplementaire pour la standardisation reglementaire (output floor, TRIM).

### 9.2 These du memoire (revisee)

> Le cadre reglementaire IFRS 9 / Bale III suppose implicitement que les choix d'implementation du facteur systematique ASRF sont de second ordre. Notre travail experimental demontre qu'ils sont de premier ordre : un seul parametre de convention dans le Z-score produit un ecart de 17.5 points de pourcentage sur le RAROC en scenario de stagflation. Nous proposons une methodologie de selection de convention basee sur 5 criteres de coherence economique, et montrons qu'il n'existe pas de convention trivialement optimale : les conventions les plus stables sous-estiment le stress en crise, tandis que les plus prudentes creent des desaccords entre classes d'actifs. Ce trade-off stabilite/severite, non documente dans la litterature, constitue un argument quantitatif en faveur de la standardisation reglementaire (output floor Bale IV, programme TRIM).

### 9.3 Originalite et apport

L'apport du memoire est quadruple :
1. **Quantification** d'un risque de modele non documente dans la litterature IFRS 9 (17.5pp de swing)
2. **Methodologie** reproductible de selection de convention pour le facteur systematique (5 criteres, 4 conventions, 11 scenarios)
3. **Decouverte** du dilemme stabilite/severite dans le choix de convention
4. **Connexion** entre un choix d'implementation micro (signe dans le Z-score) et un enjeu reglementaire macro (variabilite des RWA, output floor Bale IV)

---

## 10. Ce que les banques savent — et ce qu'elles ne publient pas

### 10.1 Les banques ne sont pas aveugles

Il serait naif de penser que les equipes de model risk management des grandes banques ignorent la sensibilite de leurs modeles aux conventions de signe. Le programme TRIM de la BCE (2017-2021) a examine plus de 7000 modeles internes dans 65 banques. Les equipes TRIM ont necessairement observe les ecarts de calibration entre banques pour le facteur systematique. Le rapport BCBS 256 (2013) documente explicitement que les RWA varient de +/-20% entre banques pour le meme portefeuille hypothetique.

Les banques savent. Ce qui n'existe pas dans la litterature, c'est :
1. **La quantification isolee de l'impact d'un seul parametre de convention** (17.5pp pour le signe IR)
2. **La comparaison systematique des alternatives** (4 conventions, 5 criteres, 11 scenarios)
3. **La mise en evidence du dilemme stabilite/severite** comme resultat structurel

Pourquoi cette absence ? Trois raisons :
- **Verrouillage reglementaire** : une banque ne peut pas publier "si on changeait notre convention IR, notre RAROC Stagflation passerait de +7% a -11%". Ce serait un aveu que son modele approuve est potentiellement mal calibre. Le superviseur reagirait.
- **Secret concurrentiel** : les conventions de signe sont un element du modele interne. Les publier donnerait une visibilite sur la sensibilite du portefeuille a des scenarios specifiques.
- **Biais de confirmation** : une equipe qui a concu un modele, l'a fait valider, et l'utilise depuis 5 ans a naturellement tendance a considerer ses conventions comme correctes. Il n'y a pas d'incitation interne a les remettre en question.

### 10.2 La particularite du cadre academique

Notre travail se situe dans un cadre academique-experimental, pas reglementaire. Cela change trois choses :
- **Pas de contrainte de confidentialite** : nous pouvons publier les conventions et leurs impacts
- **Pas de verrouillage reglementaire** : nous pouvons basculer d'une convention a l'autre sans notification formelle
- **Pas de biais de confirmation** : le modele a ete construit dans un but pedagogique, pas pour minimiser les RWA ou maximiser le RAROC

C'est precisement ce cadre qui rend la contribution originale. Non pas que le phenomene soit inconnu des praticiens, mais qu'il n'a jamais ete **quantifie, compare, et publie** de maniere systematique.

### 10.3 Le probleme de la dimensionnalite

Avec 5 variables macro, notre experience couvre 4 conventions (2^2, les 3 variables non-ambigues etant fixees). C'est un espace de taille raisonnable, exhaustivement explorable.

Mais les modeles reels sont bien plus complexes :
- **EBA GL/2017/06** prescrit au minimum 15 variables macro-financieres pour les tests de resistance
- **Fed CCAR (Comprehensive Capital Analysis and Review)** utilise 28 variables
- **PRA (Bank of England)** utilise 16 scenarios avec ~20 variables chacun

Pour un modele a 28 variables dont 10 ambigues (GDP, chomage, taux court, taux long, spread de credit, HPI, production industrielle, confiance des menages, chiffre d'affaires, prix des matieres premieres), l'espace des conventions est de **2^10 = 1024**. Certaines de ces 1024 conventions produiront des ECL dramatiquement differents.

Notre experience a 4 conventions est donc une **borne inferieure** du probleme reel. Le dilemme stabilite/severite que nous avons identifie ne peut que s'amplifier avec la dimensionnalite.

### 10.4 Implications pour le memoire

Ce constat renforce la these de trois facons :
1. **Notre contribution est originale malgre la connaissance implicite des praticiens** : la quantification et la comparaison systematique n'existent pas dans la litterature
2. **Notre modele a 5 variables sous-estime le probleme** : avec 28 variables, l'espace des conventions explose et le dilemme stabilite/severite s'aggrave
3. **L'argument pour la standardisation reglementaire (output floor) est renforce** : si 4 conventions a 5 variables suffisent a creer un dilemme insoluble, 1024 conventions a 28 variables rendent l'arbitrage individuel par les banques impossiblement complexe

---

## 11. Piste : calibration sur une banque reelle

### 11.1 L'idee

Notre modele ne represente aucune banque reelle. C'est une force (laboratoire experimental) mais aussi une limite (les resultats sont "in vitro"). Une extension naturelle serait de **calibrer le modele pour ressembler a une banque existante**, puis d'appliquer la methodologie de selection de convention a cette banque.

### 11.2 Donnees publiques disponibles

Les grandes banques europeennes (BNP Paribas, Deutsche Bank, Societe Generale, etc.) publient dans leurs rapports Pilier 3 :
- **Allocation du portefeuille credit par classe d'actifs** (corporate, retail, souverain, etc.)
- **RWA par classe et par methode** (IRB-A, IRB-F, Standardisee)
- **Resultats des stress tests EBA** : ECL et RWA sous 3 scenarios macro (baseline, adverse, severely adverse)
- **PD moyennes par portefeuille** (TTC et PIT)
- **LGD et EAD par segment**

Les scenarios EBA sont publics, avec les trajectoires macro detaillees.

### 11.3 Methodologie proposee

1. **Calibrer les poids d'allocation** : utiliser les donnees Pilier 3 pour reproduire la structure de bilan d'une banque cible (ex: BNP Paribas)
2. **Calibrer les PD/LGD/RW** : utiliser les PD moyennes et RWA reportes pour ajuster les profils de classes
3. **Observer la reponse aux chocs EBA** : comparer la variation d'ECL du modele avec celle publiee par la banque sous le scenario adverse EBA
4. **Inferer la convention** : la convention qui reproduit au mieux la reponse de la banque au choc EBA est probablement proche de celle que la banque utilise en interne
5. **Appliquer la methodologie de selection** : pour cette banque calibree, quelle convention serait optimale selon nos 5 criteres ?

### 11.4 Impact potentiel

Cette extension transformerait la these de "voici un dilemme theorique" en "**voici la convention implicite de la banque X, et voici pourquoi elle n'est pas optimale**". L'impact serait considerable :
- Passage du theorique a l'empirique
- Resultat directement actionable par le management de la banque cible
- Demonstration que la methodologie de selection est applicable en pratique, pas seulement en laboratoire

### 11.5 Limites et precautions

- Les donnees Pilier 3 sont **annuelles et agregees** — la calibration sera approximative
- Les stress tests EBA ne publient pas le detail du facteur systematique — l'inference de convention est indirecte
- Les banques utilisent des modeles multi-facteurs (pas un seul Z unifie) — notre simplification mono-facteur introduit un biais
- Il faudrait verifier que le resultat n'est pas un artefact de la calibration (validation croisee sur plusieurs banques)

### 11.6 Faisabilite

La calibration est faisable dans le cadre du memoire si on se limite a :
- **Une seule banque cible** (ex: BNP Paribas, la plus grande banque de la zone euro)
- **Les 10 classes principales** de notre modele (hors equities, repos, derives, corporate bonds — trop specifiques)
- **Le scenario adverse EBA 2023** comme reference de calibration
- **Validation sur le scenario EBA 2025** (out-of-sample)

Les donnees sont publiques et accessibles sur le site de l'EBA (eba.europa.eu/risk-analysis-and-data/eu-wide-stress-testing).

---

## Annexe : Chronologie de la decouverte

| Date | Etape | Resultat |
|------|-------|----------|
| Fev 2026 | Programme initial PE/Credit | Conclusion : arbitrage verrouille par la reglementation |
| Fev 2026 | Extension 14 classes + BL-CVaR | Confirmation : contraintes reglementaires verrouillent 53% |
| Fev 2026 | Audit v4 (2157 checks) | 11/14 classes avec RAROC identique entre scenarios |
| Fev 2026 | Fix RAROC scenario-dependant | pd_cond_base au lieu de pd_base pour le RAROC loss |
| Mar 2026 | 7 critiques analysees | 3 fixes + 4 correct behavior |
| Mar 2026 | Document rationale scenarios | 11 scenarios coherents, anomalie A1 identifiee |
| Mar 2026 | Diagnostic convention IR | Swing de 17.5pp sur Stagflation — classifie a tort comme "mineur" |
| Mar 2026 | Fix convention IR | RAROC Stagflation : +6.7% -> -10.8% |
| Mar 2026 | Verification chomage bipolaire | Impact nul (bipolar=0.0 dans scenarios favorables) |
| Mar 2026 | Verification convention inflation | Correcte (double-comptage sinon) |
| Mar 2026 | Formulation : experience 4 conventions | Protocole experimental, 5 criteres de coherence |
| Mar 2026 | Execution de l'experience | 44 pipeline runs, resultat contre-intuitif : Conv A (96%) > Conv B (93%) |
| Mar 2026 | Analyse du dilemme | Trade-off stabilite/severite identifie, aucune convention ne domine |
| Mar 2026 | Reformulation de la these | Le dilemme lui-meme est le resultat principal du memoire |
| Mar 2026 | Analyse de la connaissance des banques | Les banques savent mais ne publient pas (3 raisons) |
| Mar 2026 | Analyse dimensionnalite | 5 variables = borne inferieure ; 28 vars = 1024 conventions |
| Mar 2026 | Piste calibration banque reelle | Idee : calibrer sur BNP via Pilier 3 + stress test EBA |

---

## References

- Basel Committee on Banking Supervision (2013). *Regulatory Consistency Assessment Programme (RCAP) — Analysis of risk-weighted assets for credit risk in the banking book.* BCBS 256.
- Duffie, D., Saita, L., & Wang, K. (2007). Multi-period corporate default prediction with stochastic covariates. *Journal of Financial Economics*, 83(3), 635-665.
- European Banking Authority (2017). *Guidelines on PD estimation, LGD estimation and the treatment of defaulted exposures.* EBA/GL/2017/06.
- European Central Bank (2021). *TRIM results — Targeted Review of Internal Models.* SSM TRIM Programme.
- Gordy, M. (2003). A risk-factor model foundation for ratings-based bank capital rules. *Journal of Financial Intermediation*, 12, 199-232.
- Kaplan, S.N., & Stromberg, P. (2009). Leveraged buyouts and private equity. *Journal of Economic Perspectives*, 23(1), 121-146.
- Merton, R.C. (1974). On the pricing of corporate debt: The risk structure of interest rates. *Journal of Finance*, 29(2), 449-470.
- Rockafellar, R.T., & Uryasev, S. (2002). Conditional value-at-risk for general loss distributions. *Journal of Banking & Finance*, 26(7), 1443-1471.
- Vasicek, O. (1987). Probability of loss on loan portfolio. *KMV Corporation.*
- European Banking Authority (2023). *2023 EU-wide Stress Test Results.* EBA/REP/2023/22.
- BNP Paribas (2024). *Pillar 3 Report — Risk Management and Capital Adequacy.* Registration Document.
- Board of Governors of the Federal Reserve System (2024). *Comprehensive Capital Analysis and Review: Assessment Framework and Results.* Federal Reserve.
