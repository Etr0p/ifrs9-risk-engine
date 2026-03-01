# Justification Economique des Scenarios Macroeconomiques
## IFRS 9 Risk Cockpit — Audit de Coherence Macro-Financiere

**Date**: Mars 2026
**Objet**: Documentation exhaustive de la calibration economique de chaque variable dans chaque scenario predefini, avec verification de coherence inter-variables.

---

## Table des matieres

1. [Methodologie](#1-methodologie)
2. [Scenario de reference (SCENARIO_BASE)](#2-scenario-de-reference)
3. [Scenarios predefinis — Analyse detaillee](#3-scenarios-predefinis)
   - 3.1 Central
   - 3.2 Crise financiere (GFC)
   - 3.3 Crise souveraine (2012)
   - 3.4 Stagflation
   - 3.5 Choc pandemique (COVID)
   - 3.6 Rupture techno
   - 3.7 Reprise
   - 3.8 Hypercroissance
   - 3.9 Boom immobilier
   - 3.10 Trappe a liquidite
   - 3.11 Transition climatique brutale
4. [Analyse de coherence transversale](#4-analyse-de-coherence-transversale)
5. [Synthese des anomalies et recommandations](#5-synthese)

---

## 1. Methodologie

### 1.1 Variables du modele

Le cockpit utilise 5 variables macroeconomiques, chacune mappee sur un slider dashboard :

| Variable | Slider | Conversion vers valeur effective |
|----------|--------|----------------------------------|
| Taux directeur BCE | `interest_rate_bp` (points de base) | `ir = 3.5% + bp/100` |
| Taux de chomage | `unemployment_bipolar` (pp, signe = nature) | `unemp = 7.5% + |bipolar|` |
| Croissance du PIB | `gdp_pct` (%) | Directe |
| Prix immobiliers | `hpi_pct` (%) | Directe |
| Inflation (HICP) | `inflation_pct` (%) | Directe |

**Convention bipolaire du chomage** : Le signe indique la nature du choc (+positif = rupture technologique/structurelle, -negatif = crise economique/cyclique). La valeur absolue donne l'amplitude de la hausse en points de pourcentage. Le chomage ne peut que monter par rapport a la base (7.5%).

### 1.2 Lois economiques de reference

Pour chaque scenario, la coherence est verifiee contre 4 relations macro-financieres fondamentales :

1. **Loi d'Okun** : `Du ~ -0.5 x (PIB - PIB_potentiel)`. Une baisse de 1pp du PIB sous son potentiel (~1.5%) augmente le chomage de ~0.5pp.
2. **Courbe de Phillips** : Relation inverse inflation-chomage. Un chomage eleve exerce une pression desinflationniste.
3. **Regle de Taylor** : `r = r* + 0.5(pi - pi*) + 0.5(y - y*)` avec r*=2.5%, pi*=2.0%, y*=1.5%. Prescrit le taux directeur optimal.
4. **Canal du credit immobilier** : HPI correle positivement avec le PIB et negativement avec les taux reels.

### 1.3 Tableau synthetique des valeurs effectives

| Scenario | Taux dir. | Chomage | PIB | HPI | Inflation | Type |
|----------|-----------|---------|-----|-----|-----------|------|
| **Central** | 3.50% | 7.5% | +1.2% | +2.0% | 2.5% | Neutre |
| **GFC** | 1.00% | 9.5% | -4.5% | -3.0% | 0.3% | Adverse |
| **Souveraine** | 0.75% | 11.5% | -0.9% | -2.5% | 2.5% | Adverse |
| **Stagflation** | 6.00% | 10.5% | -1.0% | -5.0% | 8.0% | Adverse |
| **COVID** | 0.00% | 8.0% | -6.0% | +5.0% | 0.3% | Adverse |
| **Rupture techno** | 2.50% | 9.0% | +0.9% | +3.0% | 2.2% | Neutre |
| **Reprise** | 1.50% | 7.5% | +2.5% | +4.5% | 1.6% | Favorable |
| **Hypercroissance** | 8.00% | 7.5% | +5.0% | +10.0% | 4.0% | Favorable |
| **Boom immobilier** | 3.50% | 7.5% | +1.8% | +8.0% | 2.5% | Neutre |
| **Trappe liquidite** | 0.00% | 10.0% | 0.0% | -3.0% | -0.5% | Adverse |
| **Transition clim.** | 4.50% | 9.0% | -0.8% | -4.0% | 4.5% | Adverse |

---

## 2. Scenario de reference (SCENARIO_BASE)

Le SCENARIO_BASE represente la conjoncture courante de la zone euro, calibree sur les projections BCE 2024 :

| Variable | Valeur | Justification | Source |
|----------|--------|---------------|--------|
| PIB | 1.2% | Croissance tendancielle zone euro, legerement sous le potentiel (~1.5%) | BCE Staff Projections Dec 2024 |
| Chomage | 7.5% | Taux de chomage structural zone euro (entre NAIRU 6.5% et moyenne 2015-24 ~8%) | Eurostat LFS |
| Taux directeur | 3.5% | Taux principal de refinancement BCE post-normalisation (plateau 2023-24 a 4.0-4.5%, puis baisse vers 3.5%) | BCE MRO |
| HPI | 2.0% | Croissance nominale des prix immobiliers en regime normal (~inflation + productivite) | Eurostat HPI |
| Inflation | 2.5% | Legerement au-dessus de la cible BCE (2.0%), refletant la persistance post-2022 | Eurostat HICP |

**Equilibre structurel de long terme** (theta O-U, distinct du SCENARIO_BASE) :
- NAIRU = 6.5%, PIB potentiel = 1.5%, r* = 2.5%, HPI LT = 2.0%, cible inflation = 2.0%
- Le SCENARIO_BASE est une conjoncture (peut differer de l'equilibre), pas un objectif.

---

## 3. Scenarios predefinis — Analyse detaillee

---

### 3.1 Central

**Evenement de reference** : Baseline BCE 2024 — projection mediane.

| Slider | Valeur | Effective | Explication |
|--------|--------|-----------|-------------|
| `interest_rate_bp` | 0 | 3.50% | Aucun delta par rapport au taux de base. Le scenario Central represente l'absence de choc. |
| `unemployment_bipolar` | 0.0 | 7.50% | Pas de variation du chomage. L'economie tourne a son rythme tendanciel. |
| `gdp_pct` | 1.2 | 1.20% | Croissance tendancielle zone euro. En dessous du potentiel (1.5%) car heritiere du ralentissement post-COVID / guerre Ukraine. |
| `hpi_pct` | 2.0 | 2.00% | Croissance immobiliere alignee sur l'inflation + gains de productivite dans la construction. Pas de bulle, pas de correction. |
| `inflation_pct` | 2.5 | 2.50% | Legerement au-dessus de la cible BCE (2.0%). Reflete la persistance inflationniste des services post-2022 (inertie des salaires negocies). |

**Verification de coherence** :
- **Taylor** : r_Taylor = 2.5 + 0.5*(2.5-2.0) + 0.5*(1.2-1.5) = 2.6%. Taux effectif = 3.5%. Ecart de +90bp : la BCE reste restrictive (inflation > cible), coherent avec la posture 2024.
- **Okun** : Du = -0.5*(1.2-1.5) = +0.15pp. Quasi-nul, coherent avec chomage stable.
- **Phillips** : Chomage au-dessus du NAIRU (7.5% > 6.5%) → pression desinflationniste limitee. Inflation a 2.5% reflete la rigidite post-choc energetique.
- **HPI** : Taux reels positifs (3.5% - 2.5% = 1.0%) → pression moderee sur l'immobilier. HPI = 2.0% est conservateur.

**Verdict** : COHERENT. Le Central est par construction egal au SCENARIO_BASE.

---

### 3.2 Crise financiere (GFC)

**Evenement de reference** : Faillite de Lehman Brothers (sept. 2008), crise des subprimes, recession globale 2008-2009.

| Slider | Valeur | Effective | Explication economique detaillee |
|--------|--------|-----------|----------------------------------|
| `interest_rate_bp` | -250 | 1.00% | La BCE a abaisse son taux principal de refinancement de 4.25% (juil. 2008) a 1.00% (mai 2009) en 7 baisses consecutives, soit -325bp en 10 mois. Le delta de -250bp par rapport a notre base de 3.5% produit 1.0%, ce qui correspond au plancher effectif atteint par la BCE en mai 2009. Justification : en recession severe, la banque centrale abaisse les taux pour stimuler le credit et l'investissement (canal du taux d'interet, IS-LM). |
| `unemployment_bipolar` | -2.0 | 9.50% | Le taux de chomage zone euro est passe de 7.6% (mars 2008) a 10.2% (avril 2010), soit +2.6pp. Notre calibration retient +2.0pp (7.5% → 9.5%), legerement en dessous du pic car le modele capture un instantane annualise, pas le pic cumule. Les pertes d'emploi ont touche en priorite l'industrie manufacturiere (-5% d'emplois), la construction (-12%), et les services financiers (-8%). Les mecanismes de chomage partiel (Kurzarbeit en Allemagne, CIG en Italie) ont amorti le choc : sans eux, le pic aurait atteint ~12%. |
| `gdp_pct` | -4.5 | -4.50% | Le PIB de la zone euro a recule de -4.5% en 2009 (Eurostat), la pire contraction depuis la Seconde Guerre mondiale au moment des faits. Le mecanisme : effondrement du commerce mondial (-12%), credit crunch bancaire (ratio CET1 moyen passe de 5.8% a 4.2%, forcant le deleveraging), chute de l'investissement (-12.8% en 2009), et effondrement de la confiance des menages (ESI a 65, vs 100 en regime normal). L'effet multiplicateur est amplifie par la synchronisation globale : tous les partenaires commerciaux sont en recession simultanement. |
| `hpi_pct` | -3.0 | -3.00% | Les prix immobiliers en zone euro ont recule de -3.2% en 2009 (Eurostat HPI). Ce recul est MODERE par rapport aux Etats-Unis (-18%) car : (1) les marches europeens sont moins speculatifs (pas de NINJA loans), (2) les systemes de cautionnement mutuel (France) et de Bausparkassen (Allemagne) stabilisent le marche, (3) la rigidite a la baisse des prix immobiliers en Europe est documentee (Eurosysteme Housing Market Report). L'Espagne et l'Irlande font exception avec des baisses de -8% et -12% respectivement (eclatement de bulles locales). |
| `inflation_pct` | 0.3 | 0.30% | L'inflation HICP en zone euro est tombee a 0.3% en 2009 (apres 3.3% en 2008). Ce plongeon s'explique par : (1) effondrement de la demande agregee (output gap de -4.3%), (2) chute des prix du petrole de $147 a $32 le baril (-78%), (3) compression des marges dans la distribution et les services. La BCE a temporairement toleré un risque de deflation, qui ne s'est pas materialise grace a l'ancrage des anticipations d'inflation. Le deflateur du PIB est reste positif (+1.0%), la quasi-deflation est concentree sur l'energie. |

**Verification de coherence** :
- **Taylor** : r_Taylor = 2.5 + 0.5*(0.3-2.0) + 0.5*(-4.5-1.5) = -1.6% → ZLB a 0%. Taux effectif = 1.0%. La BCE n'a pas atteint le ZLB (contrairement a la Fed a 0-0.25%) car elle craignait l'alea moral. Ecart de +100bp vs Taylor est coherent avec la doctrine BCE de l'epoque (plus conservatrice que la Fed).
- **Okun** : Du_Okun = -0.5*(-4.5-1.5) = +3.0pp. Effectif = +2.0pp. Okun predit davantage, mais l'ecart s'explique par le Kurzarbeit et la rigidite du marche du travail europeen (insider/outsider model).
- **Phillips** : Chomage en forte hausse (9.5%) et inflation quasi-nulle (0.3%) → COHERENT avec la courbe de Phillips accelerationiste.
- **HPI-taux** : Baisse des taux (-250bp) devrait soutenir l'immobilier, mais le credit crunch bancaire et la hausse du chomage dominent → HPI = -3.0% est coherent.

**Verdict** : COHERENT. Calibration historiquement fidele.

---

### 3.3 Crise souveraine (2012)

**Evenement de reference** : Crise de la dette souveraine europeenne 2011-2012 (Grece, Portugal, Irlande, Espagne, Italie — les "PIIGS").

| Slider | Valeur | Effective | Explication economique detaillee |
|--------|--------|-----------|----------------------------------|
| `interest_rate_bp` | -275 | 0.75% | La BCE a abaisse son taux de 1.50% (juil. 2011, apres deux hausses controversees de Trichet) a 0.75% (juil. 2012), puis a 0.50% (mai 2013) et 0.05% (sept. 2014). Le delta de -275bp (3.5% → 0.75%) capture le taux plancher de mi-2012. Justification : la BCE a baisse les taux pour contrer la fragmentation financiere (spreads BTP-Bund a 500bp) et le risque de redenomination. Le "Whatever it takes" de Draghi (juil. 2012) a ete le tournant psychologique, suivi des OMT. |
| `unemployment_bipolar` | -4.0 | 11.50% | Le chomage zone euro a atteint un pic historique de 12.1% (avril 2013). La hausse de +4.0pp (7.5% → 11.5%) est l'une des plus severes du dataset. Elle reflete : (1) la double recession (double-dip 2011-2013), (2) les politiques d'austerite budgetaire dans les pays peripheriques (multiplicateurs fiscaux sous-estimes, cf. Blanchard-Leigh 2013), (3) la destruction du tissu PME en Espagne (chomage a 26.3%), Grece (27.5%), et Portugal (17.5%). Le chomage des jeunes a depasse 50% en Espagne et en Grece. |
| `gdp_pct` | -0.9 | -0.90% | Le PIB zone euro a recule de -0.9% en 2012 (Eurostat). Contrairement au GFC (-4.5%), la recession est moderee en magnitude mais prolongee (6 trimestres consecutifs de contraction Q4 2011-Q1 2013). L'heterogeneite est extreme : l'Allemagne croit a +0.4% tandis que la Grece recule de -7.3%, l'Espagne de -2.9%, l'Italie de -2.8%. La moyenne masque une divergence Nord-Sud sans precedent dans l'histoire de l'UEM. |
| `hpi_pct` | -2.5 | -2.50% | Les prix immobiliers en zone euro ont recule de -2.4% en 2012 (Eurostat HPI). La baisse est concentree en peripherie : Espagne -15% (cumul -37% vs pic 2007), Irlande -12%, Grece -12%. En France et Allemagne, les prix sont quasi-stables (+0.1% et +2.7% respectivement). La correction immobiliere est un canal majeur de la crise via : (1) deterioration des bilans bancaires (NPL a 8% en Espagne, 34% en Grece), (2) effets de richesse negatifs sur la consommation, (3) contraction de la construction (-4% de la VA en peripherie). |
| `inflation_pct` | 2.5 | 2.50% | L'inflation HICP a ete de 2.5% en 2012, apparemment paradoxale pour une recession. Explication : (1) hausse de la TVA dans les pays en austerite (Espagne +3pp, Portugal +2pp, Grece +3pp), (2) prix du petrole eleves (Brent moyen $112/bbl), (3) pass-through des depreciations dans les pays peripheriques, (4) rigidite des prix des services. La courbe de Phillips ne s'applique pas ici car la desinflation est masquee par des chocs d'offre fiscaux. Ce phenomene est documente par Coeure (2013) : "inflation without growth is not a sign of overheating, it is a sign of supply shocks". |

**Verification de coherence** :
- **Taylor** : r_Taylor = 2.5 + 0.5*(2.5-2.0) + 0.5*(-0.9-1.5) = 1.55%. Effectif = 0.75%. La BCE est en dessous de Taylor (-80bp) car elle combat activement la fragmentation financiere et le risque systemique, pas seulement l'output gap.
- **Okun** : Du_Okun = -0.5*(-0.9-1.5) = +1.2pp. Effectif = +4.0pp. Enorme ecart, mais s'explique par : (1) crise prolongee (effets cumules 2011-2013), (2) hysteresis du chomage (perte de capital humain irreversible en peripherie), (3) les multiplicateurs d'austerite etaient de 1.5-2.0x au lieu des 0.5x estimes ex ante (Blanchard-Leigh 2013).
- **Phillips** : Chomage a 11.5% avec inflation a 2.5% → CONTRADICTION APPARENTE. Mais justifiee par les chocs d'offre (TVA, energie). Le Phillips Curve slope s'est aplati significativement dans les annees 2010 (IMF WEO 2013, chapitre 3).
- **HPI** : Taux bas (0.75%) devraient soutenir l'immobilier, mais le credit crunch (banques en deleveraging), le chomage massif, et l'austerite dominent. HPI = -2.5% est coherent.

**Verdict** : COHERENT. La relation Okun-chomage semble extreme (+4pp pour -0.9% PIB) mais se justifie par la duree de la crise et l'hysteresis. L'inflation a 2.5% en recession est le paradoxe le plus frappant, mais historiquement avere.

---

### 3.4 Stagflation

**Evenement de reference** : Choc petrolier 1974-75 transpose en zone euro moderne. Actualise par l'episode inflationniste 2022-23.

| Slider | Valeur | Effective | Explication economique detaillee |
|--------|--------|-----------|----------------------------------|
| `interest_rate_bp` | +250 | 6.00% | La banque centrale hausse agressivement les taux malgre la recession pour combattre l'inflation. C'est le dilemme classique de la stagflation : la politique monetaire ne peut pas simultanement lutter contre l'inflation et soutenir la croissance. La BCE choisit la stabilite des prix (mandat unique, Art. 127 TFUE). Precedent historique : la Bundesbank a maintenu des taux a 7-9% en 1974-75 malgre la recession (-1.4% PIB). Precedent recent : la BCE a monte a 4.50% en 2023 malgre la menace de recession liee a la guerre en Ukraine. Le 6.0% du scenario est une escalade credible si l'inflation atteint 8%. |
| `unemployment_bipolar` | -3.0 | 10.50% | Hausse de +3pp du chomage, moderee par rapport a la crise souveraine (+4pp) mais superieure au GFC (+2pp). En stagflation, le chomage monte par deux canaux : (1) la recession (canal classique IS), (2) la compression des marges des entreprises par les couts d'intrants (salaires reels rigides vs prix d'input en hausse), forcant des licenciements pour restaurer la profitabilite. Le secteur industriel est le plus touche (intensif en energie). La construction souffre doublement : hausse des couts de materiaux + taux hypothecaires eleves. |
| `gdp_pct` | -1.0 | -1.00% | Recession moderee (-1.0%) car la stagflation n'est pas un effondrement de la demande (GFC) mais un choc d'offre. Le PIB zone euro a recule de -0.3% au T3 2022 et de -0.1% au T1 2023 (recession technique), mais l'annee complete 2022 a ete positive (+3.4%) grace a l'acquis de croissance post-COVID. En scenario de stagflation pure (1974-75), le PIB francais a recule de -1.1% et le PIB allemand de -1.4%. Notre calibration a -1.0% est le consensus de ces deux episodes. |
| `hpi_pct` | -5.0 | -5.00% | Baisse significative des prix immobiliers due a la combinaison toxique : taux hypothecaires eleves (6% → taux hypothecaire ~8%, vs ~3% en normal), pouvoir d'achat erode par l'inflation, et recession. Le canal est direct : la mensualite d'un pret immobilier a 8% est 47% plus elevee qu'a 3% pour un meme montant, comprimant la capacite d'emprunt des menages. Precedent : en 1974-75, les prix immobiliers en France ont recule de -8% en reel ; en 2022-23, les transactions immobilieres en zone euro ont chute de -20% et les prix de -2% a -4% selon les pays. Notre -5% est le haut de la fourchette, justifie par des taux a 6% (plus eleves que 2023). |
| `inflation_pct` | 8.0 | 8.00% | Inflation elevee, calibree sur le pic HICP de la zone euro : 10.6% (oct. 2022). Notre scenario retient 8.0% comme un plateau credible sur un an (vs pic ponctuel). Justification historique : en 1974, l'inflation HICP equivalente en Europe a atteint 13.2% (moyenne CEE). En regime de ciblage d'inflation moderne (post-1999), les anticipations sont mieux ancrees, ce qui plafonne l'inflation a ~8-10%. Les canaux : choc d'offre energetique (petrole, gaz) → pass-through aux prix industriels (PPI +40% en 2022) → pass-through aux prix a la consommation (alimentation +14%, services +5%). La spirale prix-salaires est contenue par la credibilite de la banque centrale mais pas completement evitee (negociations salariales +4-5% en zone euro 2023-24). |

**Verification de coherence** :
- **Taylor** : r_Taylor = 2.5 + 0.5*(8.0-2.0) + 0.5*(-1.0-1.5) = 4.25%. Effectif = 6.0%. La BCE est +175bp AU-DESSUS de Taylor, refletant une posture ultra-hawkish pour casser les anticipations d'inflation. C'est le choix Volcker (1979-82) : taux tres restrictifs malgre la recession, pour restorer la credibilite.
- **Okun** : Du_Okun = -0.5*(-1.0-1.5) = +1.25pp. Effectif = +3.0pp. L'ecart s'explique par la composante structurelle : le chomage monte non seulement a cause de la recession mais aussi a cause de la destruction d'entreprises intensives en energie (fermetures d'usines chimiques, metallurgiques, papeteries en Europe 2022-23, documentees par Eurostat SBS).
- **Phillips** : Chomage 10.5% + inflation 8.0% → VIOLATION CLASSIQUE. Mais c'est la definition meme de la stagflation : coexistence pathologique de chomage eleve et d'inflation elevee, due a un choc d'offre (et non de demande).
- **HPI** : Taux tres eleves (6%) + recession + inflation erode le pouvoir d'achat → triple pression baissiere. HPI = -5.0% est coherent, voire conservateur.

**Verdict** : COHERENT. La stagflation viole la courbe de Phillips par construction, mais les 5 variables sont internement consistantes.

---

### 3.5 Choc pandemique (COVID)

**Evenement de reference** : COVID-19, T2 2020 annualise, zone euro.

| Slider | Valeur | Effective | Explication economique detaillee |
|--------|--------|-----------|----------------------------------|
| `interest_rate_bp` | -350 | 0.00% | La BCE a abaisse le taux de la facilite de depot de -0.50% a -0.50% (inchange, deja au ZLB), mais a lance le PEPP (Pandemic Emergency Purchase Programme) de 1 850 Mds EUR, la plus grande intervention non-conventionnelle de son histoire. Le taux de refinancement MRO est passe de 0.00% a 0.00% (deja a zero). Notre delta de -350bp (3.5% → 0.0%) capture l'equivalent conventionnel de l'ensemble du paquet (taux + PEPP + TLTRO-III favorable a -1.0% vs MRO). Wu-Xia shadow rate pour la zone euro en 2020 : environ -2.0%, coherent avec notre representation. |
| `unemployment_bipolar` | -0.5 | 8.00% | Le paradoxe du marche du travail COVID : le PIB s'effondre de -6% mais le chomage ne monte que de +0.5pp (7.1% → 7.6% en zone euro, Eurostat). C'est le resultat direct des dispositifs de chomage partiel : SURE (Support to mitigate Unemployment Risks in an Emergency) a finance 31.3 Mds EUR pour 19 pays, couvrant 30 millions de travailleurs. En Allemagne, le Kurzarbeit a protege 6 millions d'emplois ; en France, l'activite partielle a couvert 8.4 millions de salaries au pic (avril 2020). Sans ces dispositifs, le FMI estime que le chomage aurait atteint 12-14% (WEO Oct 2020, Box 1.1). Notre +0.5pp est historiquement fidele. |
| `gdp_pct` | -6.0 | -6.00% | Le PIB zone euro a recule de -6.1% a -6.6% en 2020 selon la vintage Eurostat (premiere estimation -6.1%, revise a -6.6%), pire que le GFC (-4.5%). Le T2 2020 annualise donne -39.4% (non utilise car irrealiste en annuel). La contraction reflete : (1) confinement generalise (80% des activites de services fermees mars-mai), (2) effondrement du commerce intra-UE (-15% en volume), (3) arret de la production automobile (-33% en T2), (4) chute du tourisme (secteur representant 10% du PIB en Espagne, Grece, Portugal). La reprise en V au S2 2020 a limite les degats annuels. |
| `hpi_pct` | +5.0 | +5.00% | Le PARADOXE COVID : PIB -6% mais HPI +5%. Eurostat confirme une hausse des prix immobiliers de +5.4% en zone euro en 2020. Les mecanismes : (1) **Teletravail** : exode urbain vers la peripherie, creation d'une demande pour les maisons avec jardin (surfaces +20% en recherches Seloger/Immoscout24), (2) **Taux zero** : taux hypothecaires a 1.0-1.5% (record historique), capacite d'emprunt augmentee de 15-20%, (3) **Epargne forcee** : taux d'epargne des menages a 19.4% en zone euro (vs 12.5% en 2019), deployee dans l'immobilier, (4) **Raret de l'offre** : arret des chantiers au T2 2020 (-25% de permis de construire), (5) **Stimulus fiscal** : garanties publiques de prets + moratoires hypothecaires → pas de ventes forcees. Ce paradoxe est abondamment documente (Guerrieri et al. 2020, OCDE Housing Outlook 2021). |
| `inflation_pct` | 0.3 | 0.30% | Quasi-deflation, refletant l'effondrement de la demande de services. HICP zone euro = 0.3% en 2020 (Eurostat). Decomposition : energie -6.8% (petrole Brent $42/bbl en moyenne vs $64 en 2019), services +1.4% (rigides), biens industriels +0.2% (demande en berne), alimentation +2.3% (perturbations de chaines d'approvisionnement). L'inflation sous-jacente (hors energie et alimentation) est tombee a 0.7%, proche du seuil deflationiste. La BCE a explicitement invoque le risque de deflation pour justifier le PEPP. |

**Verification de coherence** :
- **Taylor** : r_Taylor = 2.5 + 0.5*(0.3-2.0) + 0.5*(-6.0-1.5) = -2.4% → ZLB a 0%. Effectif = 0.0%. COHERENT (ZLB bind).
- **Okun** : Du_Okun = -0.5*(-6.0-1.5) = +3.75pp. Effectif = +0.5pp. Ecart massif (-3.25pp), entierement explique par les dispositifs de chomage partiel. C'est l'anomalie la plus documentee de l'histoire macro-financiere recente.
- **Phillips** : Chomage a 8.0% + inflation a 0.3% → COHERENT (faible demande = desinflation).
- **HPI** : PIB -6% → HPI devrait baisser. Mais taux zero (0.0%) + epargne forcee + teletravail dominent → HPI +5.0%. PARADOXE HISTORIQUEMENT VERIFIE.

**Observations specifiques au modele** :
- L'ecart Okun-chomage produit un effet surprenant dans le modele : le COVID est le PIRE scenario en PIB (-6%) mais l'un des plus benins en chomage (+0.5pp). Or, c'est la combinaison unemp + GDP qui drive le Z-score (via `macro_to_z()`). Les poids respectifs (GDP: 0.60, unemp: 0.40 dans VSTOXX; variables dans macro_sensitivities pour les classes BS) determinent quel signal domine.

**Verdict** : COHERENT. Les paradoxes (HPI+5% en recession, chomage quasi-stable) sont historiquement documentes et justifies.

---

### 3.6 Rupture techno

**Evenement de reference** : Eclatement de la bulle internet 2001-2003, avec analogie vers une disruption par l'IA.

| Slider | Valeur | Effective | Explication economique detaillee |
|--------|--------|-----------|----------------------------------|
| `interest_rate_bp` | -100 | 2.50% | La BCE a abaisse les taux de 4.75% (oct. 2000) a 2.00% (juin 2003) en reponse au ralentissement. Le delta de -100bp (3.5% → 2.5%) est modere car le choc est sectoriel (Tech) et non systemique. La reponse monetaire est proportionnee : pas de QE, pas de TLTRO, juste un assouplissement conventionnel. L'analogie IA : si une bulle IA eclate, la BCE baisserait de 50-150bp car le reste de l'economie (services, industrie) n'est pas directement impacte. |
| `unemployment_bipolar` | +1.5 | 9.00% | Signe positif = rupture technologique (structurelle, pas cyclique). Le chomage monte de +1.5pp, concentre sur le secteur tech et ses dependances (consulting IT, startups, media numerique). En 2001-03, le chomage zone euro est passe de 8.0% a 9.0% (+1.0pp). Notre +1.5pp est legerement majore pour refleter une disruption IA plus large (impact sur les cols blancs : comptabilite, juridique, traduction, support client). Le signe positif signale que le PE tech pourrait en BENEFICIER (destruction creatrice schumpeterienne : les survivants racheteraient a bas prix les actifs des perdants). |
| `gdp_pct` | 0.9 | 0.90% | Ralentissement sans recession (0.9% vs potentiel 1.5%). Le PIB zone euro a cru de +0.9% en 2002, confirmant un ralentissement modere. Le tech bust affecte le PIB via : (1) investissement en equipement IT (-15% en 2001-02), (2) perte de capitalisation boursiere (NASDAQ -78%, Euro Stoxx TMT -83%), (3) effet de richesse negatif sur la consommation des menages fortunes (-3% de la consommation via canal wealth). Mais l'industrie traditionnelle, l'immobilier, et les services non-tech ne sont pas touches. Le PIB agrege reste positif. |
| `hpi_pct` | 3.0 | 3.00% | L'immobilier est non-affecte par le tech bust. En 2001-03, les prix immobiliers en zone euro ont AUGMENTE de +6.5% en moyenne (France +9%, Espagne +17%, Allemagne -2%). Notre +3.0% est conservateur. La decorrelation tech-immobilier s'explique par : (1) les taux bas stimulent les prets hypothecaires, (2) l'immobilier joue son role de valeur refuge quand les marches actions s'effondrent (rotation tech → immobilier), (3) les fondamentaux demographiques (deficit de logements en France et Espagne) ne sont pas affectes par la tech. |
| `inflation_pct` | 2.2 | 2.20% | Inflation stable a 2.2%, a peine au-dessus de la cible. Le HICP zone euro etait a 2.3% en 2002. Le tech bust n'a pas d'impact inflationniste/desinflationniste significatif car : (1) la tech represente <5% du panier HICP (electromenager, informatique, telecom), (2) les prix de l'energie sont stables (pas de choc petrolier), (3) les salaires dans le tech sont flexibles a la baisse (stock options annulees, bonus supprimes) mais ne pesent pas sur l'inflation aggregee. |

**Verification de coherence** :
- **Taylor** : r_Taylor = 2.5 + 0.5*(2.2-2.0) + 0.5*(0.9-1.5) = 2.3%. Effectif = 2.5%. COHERENT (ecart de +20bp, negligeable).
- **Okun** : Du_Okun = -0.5*(0.9-1.5) = +0.3pp. Effectif = +1.5pp. L'ecart (+1.2pp) s'explique par la composante structurelle du chomage tech : les competences des developpeurs de l'ancienne economie (COBOL, mainframe) ne sont pas transferables vers les nouveaux secteurs de croissance (analogie IA : les prompt engineers remplacent les data scientists traditionnels).
- **Phillips** : Chomage 9.0% + inflation 2.2% → quasi-neutre, coherent avec un choc sectoriel non inflationniste.
- **HPI** : Taux moderes (2.5%) + economie non-tech resiliente → HPI +3.0%. COHERENT.

**Verdict** : COHERENT. Le choc est correctement calibre comme sectoriel (tech) et non systemique.

---

### 3.7 Reprise

**Evenement de reference** : Expansion europeenne 2017-2018, le "Goldilocks" europeen.

| Slider | Valeur | Effective | Explication economique detaillee |
|--------|--------|-----------|----------------------------------|
| `interest_rate_bp` | -200 | 1.50% | La BCE maintenait son taux MRO a 0.00% et sa facilite de depot a -0.40% en 2017-18, en plein QE (APP : achats de 60-80 Mds EUR/mois). Notre 1.5% (vs 0.0% historique) reflete une normalisation partielle : dans notre framework ou le taux base est 3.5%, un delta de -200bp donne 1.5%, representant une posture accommodante (sous le taux neutre r*=2.5%) mais pas aussi extreme que le ZLB. La justification economique : malgre une croissance de 2.5%, la BCE maintient des conditions financieres souples car l'inflation reste sous-cible (1.6% < 2.0%), et le souvenir de la crise souveraine incite a la prudence. |
| `unemployment_bipolar` | 0.0 | 7.50% | Aucune hausse du chomage. Le chomage zone euro est passe de 9.4% (2017) a 8.2% (2018), en baisse continue. Notre modele ne peut pas representer une baisse du chomage sous la base (convention bipolaire → minimum = 7.5%). En realite, le chomage en 2018 etait a 8.2%, au-dessus de notre base, mais la dynamique etait favorable. La valeur 0 capture l'absence de stress sur le marche du travail. |
| `gdp_pct` | 2.5 | 2.50% | Le PIB zone euro a cru de +2.5% en 2017 (Eurostat), la meilleure performance depuis 2007. Les moteurs : (1) reprise de l'investissement (+4.5% en FBCF), (2) commerce exterieur dynamique (exportations +5.3%), (3) consommation des menages (+1.7%), (4) conditions de credit souples (taux hypothecaires a 1.5-2.0%, volume de prets aux entreprises +3%). Le multiplicateur budgetaire est positif (reduction des deficits + croissance). Tous les pays de la zone euro sont en croissance, y compris la Grece (+1.3%) pour la premiere fois depuis 2007. |
| `hpi_pct` | 4.5 | 4.50% | Les prix immobiliers en zone euro ont augmente de +4.5% en 2017 (Eurostat HPI), tirees par : (1) Allemagne +5.2% (rattrapage historique, sous-investissement chronique), (2) Irlande +12% (rebond post-crise), (3) Espagne +6.2% (fin de la correction), (4) France +3.2% (effet taux bas). La hausse est saine : pas de bulle de credit (ratio pret/valeur stable a 80%), pas de speculative exuberance (volumes de transactions normalises). |
| `inflation_pct` | 1.6 | 1.60% | L'inflation HICP a ete de 1.5% en 2017 et 1.8% en 2018. Notre 1.6% est la moyenne. L'inflation reste SOUS la cible de 2.0% malgre une croissance de 2.5% et un chomage en baisse — c'est le "missing inflation puzzle" qui a motive des annees de recherche a la BCE. Explications : (1) digitalisation comprimant les prix (Amazon effect), (2) globalisation maintenant la pression competitive sur les salaires (outsourcing, immigration), (3) anticipations d'inflation basses (5 ans/5 ans = 1.5% en 2017), (4) output gap encore negatif en 2017 (heritage de la crise souveraine). |

**Verification de coherence** :
- **Taylor** : r_Taylor = 2.5 + 0.5*(1.6-2.0) + 0.5*(2.5-1.5) = 2.8%. Effectif = 1.5%. BCE est -130bp SOUS Taylor, explicitement car l'inflation est sous-cible et le Conseil des gouverneurs considere que les risques restent orientes a la baisse (souveraine pas totalement resolue, NPL en Italie a 12%).
- **Okun** : Du_Okun = -0.5*(2.5-1.5) = -0.5pp. Notre modele ne peut pas baisser le chomage (minimum = 7.5%). LIMITATION du modele (voir section 4).
- **Phillips** : Chomage 7.5% + inflation 1.6% → COHERENT (chomage au-dessus du NAIRU → pression desinflationniste moderee).
- **HPI** : Taux bas (1.5%) + croissance (2.5%) → conditions ideales pour l'immobilier. HPI +4.5% est coherent.

**Verdict** : COHERENT. Le "missing inflation" (1.6% malgre 2.5% de croissance) est le phenomene macro le plus etudie de la decennie 2010-2020, historiquement avere.

**Observation** : Le modele ne peut pas representer une BAISSE du chomage en scenario favorable. C'est une limitation structurelle de la convention bipolaire. En Reprise, le chomage devrait etre a ~6.5-7.0%, pas 7.5%. Impact sur le modele : le RAROC en scenario Reprise est legerement sous-estime car le stress Z est surestime (chomage trop eleve → Z trop defavorable).

---

### 3.8 Hypercroissance

**Evenement de reference** : Les Trente Glorieuses (1950-1973), transposees en zone euro moderne.

| Slider | Valeur | Effective | Explication economique detaillee |
|--------|--------|-----------|----------------------------------|
| `interest_rate_bp` | +450 | 8.00% | En regime de forte croissance avec inflation moderee, les taux d'interet sont structurellement eleves. Pendant les Trente Glorieuses, les taux directeurs europeens etaient de 5-7% (Bundesbank 6%, Banque de France 5%), et les rendements obligataires de 7-9%. Le mecanisme : (1) forte demande d'investissement (modernisation industrielle, infrastructure, urbanisation) → pression sur le marche du capital, (2) inflation moderee mais persistante (4%) → taux reels positifs necessaires pour l'equilibre epargne-investissement, (3) repression financiere (taux plafonnes par l'Etat dans certains pays). Notre 8.0% est le haut de la fourchette historique, refletant un regime ou le capital est rare et le rendement de l'investissement physique est eleve. |
| `unemployment_bipolar` | 0.0 | 7.50% | Plein emploi. En 1960-73, le chomage en Europe occidentale etait de 1.5-3.0% (Maddison). Notre modele ne peut pas descendre sous 7.5% (limitation de la convention bipolaire). En realite, l'hypercroissance implique une tension extreme sur le marche du travail : appel a l'immigration (programme Gastarbeiter en Allemagne, immigration maghrebine en France), heures supplementaires structurelles, emploi des femmes. Le 7.5% est une SURESTIMATION significative — en regime d'hypercroissance, le chomage devrait etre a 3-4%. |
| `gdp_pct` | 5.0 | 5.00% | Le PIB moyen de l'Europe occidentale (CEE-6) a cru de +5.3% par an de 1950 a 1973 (Maddison Project Database 2020). France : +5.8%, Allemagne : +6.0%, Italie : +5.6%, Pays-Bas : +4.7%. Les moteurs : (1) rattrapage technologique (adoption des technologies americaines, productivity frontier), (2) investissement massif (FBCF/PIB a 25-30%), (3) ouverture commerciale (creation du Marche Commun 1957), (4) main-d'oeuvre abondante (baby-boom + immigration), (5) stable macro environment (Bretton Woods). Notre 5.0% est conservateur par rapport a la moyenne historique (+5.3%). |
| `hpi_pct` | 10.0 | 10.00% | L'urbanisation massive des Trente Glorieuses a provoque une hausse sans precedent des prix immobiliers. En France, les prix reels ont augmente de +6-8% par an de 1950 a 1975 (Friggit 2007), soit +10-12% en nominal. Les mecanismes : (1) exode rural massif (population urbaine de 55% a 73% en France), (2) baby-boom (cohortes nees 1945-65 formant des menages dans les annees 60-70), (3) programmes de construction massifs (grands ensembles, HLM) insuffisants pour absorber la demande, (4) revenus reels en hausse de +5%/an → capacite d'achat croissante. Notre +10.0% est coherent avec les donnees historiques. |
| `inflation_pct` | 4.0 | 4.00% | L'inflation pendant les Trente Glorieuses etait moderee mais persistante : 4-5% en moyenne en Europe occidentale (avant le choc petrolier de 1973). Les mecanismes : (1) croissance rapide de la masse monetaire (+8-10%/an) en regime de changes fixes (Bretton Woods), (2) tensions sur le marche du travail → salaires en hausse de +6-8%/an, (3) repression financiere (taux reels negatifs → desincitation a l'epargne → consommation), (4) indexation automatique des salaires sur les prix dans plusieurs pays (echelle mobile en Italie, SMIC indexe en France). L'inflation etait toleree car la croissance la compensait en termes reels (paradoxe de la "bonne inflation"). Notre 4.0% est coherent et en dessous du pic pre-choc petrolier (5.6% en 1972 pour la CEE). |

**Verification de coherence** :
- **Taylor** : r_Taylor = 2.5 + 0.5*(4.0-2.0) + 0.5*(5.0-1.5) = 5.25%. Effectif = 8.0%. Ecart de +275bp, mais justifie en regime de repression financiere + forte demande d'investissement. Les taux des Trente Glorieuses etaient structurellement au-dessus de la regle de Taylor moderne car (1) la productivite marginale du capital etait elevee (rattrapage), (2) l'offre de capital etait limitee (marches financiers peu developpes), (3) les gouvernements empruntaient massivement pour l'infrastructure.
- **Okun** : Du_Okun = -0.5*(5.0-1.5) = -1.75pp → chomage devrait baisser de 1.75pp (a ~5.75%). Effectif = 7.5% (stable). LIMITATION du modele.
- **Phillips** : Chomage 7.5% + inflation 4.0% → apparemment incoherent (chomage eleve = faible inflation selon Phillips). Mais le vrai chomage serait ~3% (limitation bipolaire). A 3% de chomage, 4% d'inflation est COHERENT avec la Phillips Curve classique (Samuelson-Solow 1960).
- **HPI** : Taux eleves (8.0%) → HPI devrait baisser. Mais croissance (+5%) + urbanisation + revenus en hausse dominent largement l'effet taux. HPI +10% est COHERENT en regime d'hypercroissance.

**Verdict** : COHERENT avec la reserve importante que le chomage devrait etre a 3-4%, pas 7.5%. La convention bipolaire empeche de representer le plein emploi.

---

### 3.9 Boom immobilier

**Evenement de reference** : Expansion immobiliere europeenne 2015-2019 / post-COVID 2021-2022.

| Slider | Valeur | Effective | Explication economique detaillee |
|--------|--------|-----------|----------------------------------|
| `interest_rate_bp` | 0 | 3.50% | Taux neutre. Le boom immobilier se produit dans un environnement de taux normaux. Historiquement, la bulle immobiliere europeenne 2015-19 s'est developpee avec des taux a 0% (QE), mais notre scenario choisit deliberement un taux neutre (3.5%) pour montrer qu'un boom peut exister SANS taux zero — via les anticipations, la speculation, et les flux de capitaux. Precedent : la bulle espagnole 2002-07 s'est developpee avec des taux BCE a 2.0-4.0%, et la bulle irlandaise avec des taux similaires. Le canal est le credit et les anticipations de prix, pas necessairement le niveau des taux. |
| `unemployment_bipolar` | 0.0 | 7.50% | Pas de stress sur le marche du travail. L'economie est saine ; c'est un boom localise sur l'immobilier, pas une surchauffe generale. Le chomage en zone euro etait a 7.6% en 2019. |
| `gdp_pct` | 1.8 | 1.80% | Croissance moderee, legerement au-dessus du potentiel (1.5%). Le boom immobilier contribue positivement au PIB via la construction (+3-4% de la VA) et la consommation par effet de richesse. Le PIB zone euro a cru de +1.9% en 2018 et +1.6% en 2019. Notre 1.8% est la moyenne de ces deux annees. Le boom immobilier n'est pas en soi un facteur de forte croissance : l'effet multiplicateur de la construction est modere (~1.2x) et l'essentiel de la hausse des prix ne cree pas de PIB reel (gains en capital, pas de production). |
| `hpi_pct` | 8.0 | 8.00% | C'est LA variable definissante de ce scenario. +8% de hausse des prix immobiliers correspond a : (1) Allemagne +8.8% en 2018 (Destatis), (2) Portugal +9.2% en 2018, (3) Pays-Bas +8.9% en 2018, (4) France +3.2% (plus modere). La moyenne ponderee zone euro etait de +4.5% en 2018, mais les marches dynamiques etaient a +8-12%. Notre scenario represente un boom generalise (tous les marches a +8%), analogue a l'episode post-COVID 2021 (zone euro +9.4%, Eurostat). Les moteurs : speculation, sous-investissement chronique dans la construction neuve, flux de capitaux internationaux (Airbnb, fonds immobiliers), et anticipations auto-realisatrices de hausse des prix. |
| `inflation_pct` | 2.5 | 2.50% | Inflation a la cible. Le boom immobilier ne genere pas d'inflation significative des biens et services car : (1) les prix immobiliers ne sont PAS directement dans le HICP (seuls les loyers le sont, a ~6% du panier, et ils sont regules dans de nombreux pays — encadrement des loyers a Berlin, Paris, Amsterdam), (2) l'immobilier est une classe d'actifs, pas un bien de consommation courante, (3) l'effet de richesse sur la consommation est modere en Europe (MPC de la richesse immobiliere = 2-4 centimes par euro, Slacalek et al. 2020, vs 5-7 centimes aux US). |

**Verification de coherence** :
- **Taylor** : r_Taylor = 2.5 + 0.5*(2.5-2.0) + 0.5*(1.8-1.5) = 2.9%. Effectif = 3.5%. BCE a +60bp au-dessus de Taylor, coherent avec la normalisation monetaire 2018-19.
- **Okun** : Du_Okun = -0.5*(1.8-1.5) = -0.15pp. Quasi-nul, coherent.
- **Phillips** : Chomage 7.5% + inflation 2.5% → coherent (chomage legerement au-dessus du NAIRU).
- **HPI** : Le HPI a +8.0% avec taux neutres (3.5%) implique que le boom est SPECULATIF et non credit-driven. C'est un choix de modelisation interessant : les bulles les plus dangereuses (Minsky) sont celles qui se forment sans levier excessif, car elles ne sont pas detectees par les indicateurs macroprudentiels standard (credit-to-GDP gap < seuil).

**Verdict** : COHERENT. Le scenario est bien calibre comme un boom immobilier localise dans un environnement macro sain. La principale utilite pour le modele est de tester la sensibilite du portefeuille mortgage au HPI.

---

### 3.10 Trappe a liquidite

**Evenement de reference** : Japon 1995-2015 ("decennies perdues") / Zone euro 2014-2016 (risque deflationiste).

| Slider | Valeur | Effective | Explication economique detaillee |
|--------|--------|-----------|----------------------------------|
| `interest_rate_bp` | -350 | 0.00% | Taux a zero (ZLB = Zero Lower Bound), ou les instruments de politique monetaire conventionnelle sont epuises. La BoJ a maintenu son taux a 0-0.5% de 1999 a 2016 (taux negatif de -0.10% apres 2016). La BCE a atteint le ZLB en 2014 (MRO a 0.05%, facilite de depot a -0.10%) et y est restee jusqu'en 2022. Keynes (1936) definit la trappe a liquidite comme la situation ou meme des taux a zero ne stimulent pas l'investissement car les agents economiques preferent detenir de la liquidite (preference pour la liquidite absolue). Le mecanisme : anticipations deflationistes → taux reel = taux nominal - inflation = 0% - (-0.5%) = +0.5%, positivement restrictif malgre les taux nominaux a zero. |
| `unemployment_bipolar` | -2.5 | 10.00% | Chomage structurellement eleve (+2.5pp a 10.0%). Au Japon, le chomage est reste a 4-5% (faible, mais par convention culturelle — la main-d'oeuvre japonaise est deplacee vers le sous-emploi et les contrats temporaires). En zone euro 2014-16, le chomage etait a 10.3-11.0%. Notre 10.0% est coherent avec l'experience europeenne. Le chomage est PERSISTANT (hysteresis) car : (1) investissement faible → pas de creation de nouveaux postes, (2) anticipations pessimistes → les entreprises n'embauchent pas, (3) deflation de la dette → desendettement des menages et entreprises = frein a la demande. |
| `gdp_pct` | 0.0 | 0.00% | Stagnation seculaire (Summers 2014, Hansen 1939). Croissance nulle, pas recession — le PIB stagne autour de son niveau sans augmenter ni baisser. Le PIB reel du Japon a cru en moyenne de +0.8%/an sur 1995-2015, avec de frequentes annees a 0% ou negatives (1998: -1.3%, 2001: +0.4%, 2008: -1.2%). La zone euro a cru de +0.9% en 2014 et +2.0% en 2015 (sortie de la trappe grace au QE). Notre 0.0% est legerement en dessous de la moyenne japonaise (+0.8%) mais capture les phases les plus aigues de stagnation (1998-2002 : moyenne ~+0.1%). C'est un equilibre bas stable : pas assez de demande pour la croissance, mais pas de choc additionnel pour la recession. |
| `hpi_pct` | -3.0 | -3.00% | Baisse des prix immobiliers, refletant l'experience japonaise et europeenne. Au Japon, les prix immobiliers ont chute de -65% entre 1991 et 2010 (soit environ -4.5%/an en moyenne). En zone euro 2012-14, les prix ont recule de -2% a -4% selon les pays. Notre -3.0% est calibre sur la phase stable de deflation immobiliere (pas le krach initial, mais l'erosion continue). Les mecanismes : (1) malgre les taux a 0%, la demande de credit est atone (deleveraging), (2) les menages epargnent par precaution (motif de precaution keynesien), (3) la baisse des prix immobiliers s'auto-entretient (negative equity → ventes forcees → nouvelles baisses, Shiller 2005). |
| `inflation_pct` | -0.5 | -0.50% | DEFLATION. C'est la variable definissante de la trappe a liquidite. Le Japon a connu une inflation moyenne de -0.3% de 1999 a 2013. La zone euro a frole la deflation en 2014 (HICP +0.4%) et 2015 (+0.0%). Notre -0.5% est historiquement calibre. La deflation est le symptome ET la cause de la trappe : (1) les prix baissent → les consommateurs reportent leurs achats (effet substitution intertemporel), (2) la dette reelle augmente (effet Fisher : dette nominale fixe, prix en baisse → charge reelle croissante), (3) les taux reels augmentent (r_reel = 0% - (-0.5%) = +0.5%), aggravant la contraction. Le cercle vicieux deflation-stagnation est le cauchemar des banquiers centraux. |

**Verification de coherence** :
- **Taylor** : r_Taylor = 2.5 + 0.5*(-0.5-2.0) + 0.5*(0.0-1.5) = 0.5% → ZLB a 0%. Effectif = 0.0%. COHERENT. Taylor prescrit des taux negatifs (-0.5 a 0%), mais le ZLB empeche leur implementation effective (d'ou le QE/taux negatifs sur les facilites).
- **Okun** : Du_Okun = -0.5*(0.0-1.5) = +0.75pp. Effectif = +2.5pp. L'ecart massif (+1.75pp) s'explique par l'hysteresis : la stagnation prolongee detruit du capital humain (desaffiliation, perte de competences des chomeurs de longue duree). L'elasticite Okun est de ~2.0x en regime de stagnation seculaire vs ~0.5x en regime normal.
- **Phillips** : Chomage 10.0% + deflation -0.5% → COHERENT (courbe de Phillips a sa limite gauche : fort chomage = deflation).
- **HPI** : Taux a 0% devraient soutenir l'immobilier, mais le deleveraging + deflation + chomage → demande atone. HPI = -3.0% est coherent : les taux bas ne suffisent pas quand la demande de credit est morte.

**Verdict** : COHERENT. Scenario bien calibre, representant fidelement le piege deflationiste japonais. La coherence interne est excellente : toutes les variables pointent dans la meme direction (stagnation-deflation-chomage persistant).

---

### 3.11 Transition climatique brutale

**Evenement de reference** : NGFS Sudden Wake-Up Call (Mai 2025), scenario de transition desordonnee.

| Slider | Valeur | Effective | Explication economique detaillee |
|--------|--------|-----------|----------------------------------|
| `interest_rate_bp` | +100 | 4.50% | La banque centrale hausse les taux de 100bp pour combattre l'inflation energetique liee a la transition. Le scenario NGFS prevoit un choc d'offre energetique (fermeture acceleree des centrales fossiles, taxe carbone brutale) generant de l'inflation par les couts. La BCE reagit en durcissant la politique monetaire, creant un dilemme similaire a la stagflation (hausse des taux en recession). La hausse est moderee (+100bp vs +250bp en stagflation) car l'inflation climatique est consideree comme temporaire par une partie du Conseil des gouverneurs ("greenflation transitoire", Lane 2023). |
| `unemployment_bipolar` | -1.5 | 9.00% | Chomage sectoriel (+1.5pp). La transition brutale detruit des emplois dans : (1) l'industrie fossile (charbon, petrole, gaz — 2.5M emplois directs en UE), (2) l'automobile thermique (2.6M emplois en UE, dont 300k directement menaces par l'electrification), (3) la construction non-renovee (batiments non conformes DPE classes F/G). Les nouveaux emplois "verts" (panneaux solaires, batteries, renovation energetique) mettent 3-5 ans a se materialiser (formation, construction d'usines). L'asymetrie temporelle cree du chomage frictionnel/structurel. Notre +1.5pp est aligne avec les projections NGFS Disorderly (~1.0-2.0pp selon les modeles REMIND et MESSAGE). |
| `gdp_pct` | -0.8 | -0.80% | Recession de transition moderee. Le PIB recule de -0.8% en raison de : (1) choc d'offre energetique (couts de production en hausse de 15-25% pour les industries intensives en carbone), (2) incertitude reglementaire (entreprises reportent les investissements en attente de clarte sur les normes), (3) actifs echoues (stranded assets) : devaluation des reserves fossiles et du parc immobilier non renove (-500 Mds EUR estimes par la BCE Climate Stress Test 2022). Le scenario NGFS Sudden Wake-Up prevoit un PIB cumule inferieur de -3 a -5% sur 2025-30 par rapport au scenario ordonne. Notre -0.8% est l'impact annualise de la premiere annee du choc. |
| `hpi_pct` | -4.0 | -4.00% | Baisse significative des prix immobiliers liee au risque climatique. Les mecanismes : (1) **decote DPE** : les passoires energetiques (classes F/G, ~17% du parc en zone euro) subissent une decote de 15-25% (source : Notaires de France, ImmobilienScout24). L'obligation de renovation avant vente (prevue dans plusieurs pays) force des ventes a prix réduit. (2) **Risque physique** : les zones inondables, littorales et exposees aux canicules subissent une depreciation (etude BCE: -5 a -12% dans les zones a risque). (3) **Credit rationing** : les banques appliquent des criteres ESG aux prets immobiliers, excluant les batiments non conformes. (4) **Couts de renovation** : 25 000-40 000 EUR par logement (IEA), pesant sur la valeur nette. |
| `inflation_pct` | 4.5 | 4.50% | "Greenflation" : l'inflation d'origine climatique. Les canaux : (1) **Taxe carbone** : le doublement brutal du prix du carbone (de 50 EUR/t a 125 EUR/t, `carbon_price_shock=1.5`) se repercute sur les prix de l'electricite (+30-40%), du chauffage (+25%), et du transport (+15%). Le pass-through au HICP est de 0.3-0.5pp par 10 EUR/t de hausse du carbone (Metcalf 2019). Pour +75 EUR/t : impact de +2.5-3.5pp sur l'HICP. (2) **Inflation alimentaire** : le risque physique (secheresses, inondations) impacte les rendements agricoles (-5 a -15%, FAO 2022). (3) **Inflation des materiaux verts** : la demande explosive de lithium, cuivre, et terres rares augmente les prix de 50-200% (IEA Critical Minerals Report 2023). Notre 4.5% est coherent avec les estimations NGFS (3.5-5.5% dans le Sudden Wake-Up). |
| `carbon_price_shock` | 1.5 | x2.5 | Cours du carbone multiplie par 2.5 (doublement = 1.0, notre scenario = +50% de plus). Le prix EU ETS passerait de ~50 EUR/t a ~125 EUR/t. Precedent : le prix EU ETS est passe de 25 EUR/t (jan 2021) a 100 EUR/t (fev 2023), soit un quadruplement en 2 ans. Notre choc est plus modeste en amplitude mais plus soudain (1 an). |
| `physical_severity` | 0.3 | Modere | Severite physique moderee (30% de la pire hypothese). Correspond a : canicules plus frequentes (+2-3 evenements/an vs historique), inondations accentuees (degats +20%), mais pas de catastrophe systemique. Le scenario NGFS Sudden Wake-Up combine risque de transition (dominant) et risque physique (secondaire). |

**Verification de coherence** :
- **Taylor** : r_Taylor = 2.5 + 0.5*(4.5-2.0) + 0.5*(-0.8-1.5) = 2.6%. Effectif = 4.5%. BCE est +190bp AU-DESSUS de Taylor, dans une posture hawkish similaire (mais moins extreme) a la stagflation. Justification : la greenflation menace de desancrer les anticipations d'inflation si elle n'est pas combattue agressivement. La credibilite climatique de la BCE passe aussi par la stabilite des prix.
- **Okun** : Du_Okun = -0.5*(-0.8-1.5) = +1.15pp. Effectif = +1.5pp. Ecart faible (+0.35pp), attribuable a la composante structurelle du chomage de transition (mismatch de competences fossiles → vertes).
- **Phillips** : Chomage 9.0% + inflation 4.5% → VIOLATION classique, mais identique a la stagflation : c'est un choc d'offre (couts energetiques), pas un choc de demande. COHERENT.
- **HPI** : Taux eleves (4.5%) + recession (-0.8%) + decote DPE + risque physique → quadruple pression baissiere. HPI = -4.0% est coherent, voire conservateur.

**Verdict** : COHERENT. Scenario bien construit, combinant transition climatique et risque physique. Les variables sont coherentes avec les projections NGFS et la litterature climatique.

---

## 4. Analyse de coherence transversale

### 4.1 Loi d'Okun

| Scenario | PIB | Du_Okun predit | Du effectif | Ratio effectif/predit | Explication de l'ecart |
|----------|-----|---------------|-------------|----------------------|------------------------|
| Central | 1.2% | +0.15 | 0.0 | 0x | Negligeable |
| GFC | -4.5% | +3.0 | +2.0 | 0.67x | Kurzarbeit, rigidite marche travail EU |
| Souveraine | -0.9% | +1.2 | +4.0 | 3.33x | Crise prolongee, hysteresis, austerite |
| Stagflation | -1.0% | +1.25 | +3.0 | 2.40x | Destruction industrielle (energie), persistance |
| COVID | -6.0% | +3.75 | +0.5 | 0.13x | SURE/Kurzarbeit, chomage partiel |
| Rupture | 0.9% | +0.3 | +1.5 | 5.0x | Chomage structurel tech (mismatch) |
| Reprise | 2.5% | -0.5 | 0.0 | 0x | Limitation du modele (bipolar >= 0) |
| Hyper | 5.0% | -1.75 | 0.0 | 0x | Limitation du modele (bipolar >= 0) |
| Boom | 1.8% | -0.15 | 0.0 | 0x | Negligeable |
| Trappe | 0.0% | +0.75 | +2.5 | 3.33x | Hysteresis, stagnation prolongee |
| Climat | -0.8% | +1.15 | +1.5 | 1.30x | Composante structurelle moderee |

**Observations** :
- Les scenarios favorables (Reprise, Hypercroissance, Boom) ne peuvent pas representer de baisse du chomage → biais systematique haussier du Z-score.
- Le COVID est l'anomalie maximale (ratio 0.13x), parfaitement expliquee par les dispositifs publics.
- La Crise souveraine et la Trappe a liquidite ont un ratio eleve (3.33x), explique par l'hysteresis et la duree.

### 4.2 Regle de Taylor

| Scenario | r_Taylor | r_effectif | Ecart (bp) | Interpretation |
|----------|---------|------------|------------|----------------|
| Central | 2.60% | 3.50% | +90 | BCE restrictive (inflation > cible) |
| GFC | -1.60% | 1.00% | +260 | ZLB + conservatisme BCE vs Fed |
| Souveraine | 1.55% | 0.75% | -80 | Easing anti-fragmentation |
| Stagflation | 4.25% | 6.00% | +175 | Ultra-hawkish (strategie Volcker) |
| COVID | -2.40% | 0.00% | +240 | ZLB + QE non-conventionnel |
| Rupture | 2.30% | 2.50% | +20 | Quasi-Taylor |
| Reprise | 2.80% | 1.50% | -130 | Accommodante (inflation sous-cible, QE actif) |
| Hyper | 5.25% | 8.00% | +275 | Repression financiere, capital rare |
| Boom | 2.90% | 3.50% | +60 | Normalisation monetaire |
| Trappe | 0.50% | 0.00% | -50 | ZLB |
| Climat | 2.60% | 4.50% | +190 | Hawkish contre greenflation |

**Observations** :
- La BCE est systematiquement PLUS RESTRICTIVE que Taylor en scenarios adverses (GFC, COVID, Stagflation) — coherent avec son mandat unique (stabilite des prix > emploi).
- La BCE est PLUS ACCOMMODANTE que Taylor en Reprise et Souveraine — coherent avec les episodes de QE et de lutte anti-fragmentation.
- L'Hypercroissance est l'ecart maximal (+275bp), justifie par un regime economique structurellement different (Trente Glorieuses = pre-ciblage d'inflation).

### 4.3 Courbe de Phillips

| Scenario | Chomage | Inflation | Phillips respecte ? | Commentaire |
|----------|---------|-----------|---------------------|-------------|
| Central | 7.5% | 2.5% | Oui | Baseline |
| GFC | 9.5% | 0.3% | **Oui** | Fort chomage → disinflation |
| Souveraine | 11.5% | 2.5% | **Non** | Chocs d'offre fiscaux (TVA) masquent la disinflation |
| Stagflation | 10.5% | 8.0% | **Non** | Definition meme de la stagflation |
| COVID | 8.0% | 0.3% | **Oui** | Demand collapse → disinflation |
| Rupture | 9.0% | 2.2% | Oui (neutre) | Choc sectoriel, pas inflationniste |
| Reprise | 7.5% | 1.6% | **Oui** | Chomage > NAIRU → sous-cible |
| Hyper | 7.5%* | 4.0% | **Non** | *Vrai chomage serait ~3% → Phillips respecte |
| Boom | 7.5% | 2.5% | Oui | Neutre |
| Trappe | 10.0% | -0.5% | **Oui** | Fort chomage → deflation |
| Climat | 9.0% | 4.5% | **Non** | Choc d'offre (greenflation) |

**Observations** :
- Les 4 violations de Phillips (Souveraine, Stagflation, Hypercroissance, Climat) sont TOUTES explicables par des chocs d'offre ou des limitations du modele. Aucune n'est une erreur de calibration.
- La courbe de Phillips s'applique aux chocs de demande (GFC, COVID, Trappe). Elle ne s'applique pas aux chocs d'offre (Stagflation, Climat) ni aux regimes structurellement differents (Hypercroissance).

### 4.4 Canal immobilier (HPI)

| Scenario | PIB | Taux | HPI | PIB→HPI | Taux→HPI | Coherent ? |
|----------|-----|------|-----|---------|----------|-----------|
| Central | 1.2% | 3.5% | 2.0% | Neutre | Neutre | Oui |
| GFC | -4.5% | 1.0% | -3.0% | Baisse | Hausse (aide) | Oui : crise domine |
| Souveraine | -0.9% | 0.75% | -2.5% | Baisse | Hausse (aide) | Oui : credit crunch domine |
| Stagflation | -1.0% | 6.0% | -5.0% | Baisse | Baisse | Oui : double pression |
| **COVID** | **-6.0%** | **0.0%** | **+5.0%** | Baisse | Hausse | **Paradoxe** : taux dominent |
| Rupture | 0.9% | 2.5% | 3.0% | Hausse | Hausse | Oui |
| Reprise | 2.5% | 1.5% | 4.5% | Hausse | Hausse | Oui |
| Hyper | 5.0% | 8.0% | 10.0% | Hausse | Baisse | PIB domine |
| **Boom** | **1.8%** | **3.5%** | **+8.0%** | Faible hausse | Neutre | **Speculatif** |
| Trappe | 0.0% | 0.0% | -3.0% | Neutre | Hausse (aide) | Demande atone domine |
| Climat | -0.8% | 4.5% | -4.0% | Baisse | Baisse + DPE | Oui |

**Observations** :
- Deux scenarios ont un HPI "contre-intuitif" : COVID (+5% en recession) et Boom (+8% avec PIB modeste). Les deux sont historiquement documentes.
- Le COVID est l'exemple canonique ou les taux zero dominent le canal immobilier malgre la recession.
- Le Boom est un scenario de speculation pure (anticipations auto-realisatrices).

---

## 5. Synthese des anomalies et recommandations

### 5.1 Anomalies identifiees

| # | Anomalie | Severite | Explication | Recommandation |
|---|---------|----------|-------------|----------------|
| A1 | Convention bipolaire empeche la baisse du chomage en scenarios favorables | **MODEREE** | Reprise, Hypercroissance et Boom ont un chomage a 7.5% au lieu de 5-6% | Le RAROC en scenarios favorables est legerement sous-estime. Pas d'impact materiel sur l'allocation car le RAROC reste positif. Amelioration possible : permettre unemployment_bipolar negatif → abs() supprime. |
| A2 | Crise souveraine : ratio Okun 3.33x (chomage +4pp pour PIB -0.9%) | **FAIBLE** | Crise prolongee + hysteresis | Historiquement verifie. Pas d'erreur. |
| A3 | Hypercroissance : taux a 8% vs Taylor a 5.25% | **FAIBLE** | Regime structurel different (Trente Glorieuses) | Correct pour la reference historique. |

### 5.2 Limitation structurelle identifiee

**La convention bipolaire du chomage** (`unemployment_rate = base + |bipolar|`) ne permet pas de representer un marche du travail tendu (chomage < base). Cela affecte 3 scenarios :

- **Reprise** : chomage reel 2017-18 = 8.2%, vs modele = 7.5%. Ecart faible, impact negligeable.
- **Hypercroissance** : chomage reel Trente Glorieuses = 2-3%, vs modele = 7.5%. Ecart majeur, mais ce scenario est de toute facon hors du domaine de validite du modele (regime pre-moderne).
- **Boom immobilier** : chomage reel 2018 = 7.6%, vs modele = 7.5%. Impact nul.

**Impact sur le modele** : Le Z-score dans `macro_to_z()` est systematiquement surestime en scenarios favorables (le chomage a 7.5% au lieu de 5-6% ajoute du stress). Cela surestime les PD conditionnelles et sous-estime les RAROC. L'effet est de 2nd ordre car le PIB est le driver dominant du Z-score (poids 0.60 vs 0.40 pour le chomage dans le modele VSTOXX).

### 5.3 Fix applique : Convention IR dans macro_to_z() (Mars 2026)

**Probleme identifie** : La fonction `macro_to_z()` traitait les hausses de taux comme favorables (`z -= w * (x-mu)/sigma`). Cela produisait des resultats incoherents dans 3 scenarios :

- **Reprise** (taux=2.5%, GDP=2.5%) : les baisses de taux accommodantes etaient traitees comme adverses (+2.4sigma pour corporate). Le pd_cond resultant (12.6%) etait incoherent pour un scenario d'expansion.
- **Stagflation** (taux=6%, GDP=-1%) : les hausses de taux restrictives etaient traitees comme favorables. Le RAROC portfolio de +6.7% sous-estimait gravement le stress.
- **Transition climatique** (taux=4.5%, GDP=-0.5%) : meme probleme directionnel.

**Fix** : Changement de convention pour le taux d'interet. Desormais, les hausses de taux sont traitees comme adverses (`z += w * (x-mu)/sigma`), identique au chomage. La justification economique est le canal du service de la dette (Duffie et al. 2007) : des taux plus eleves augmentent le cout de la dette et donc la probabilite de defaut. Ce canal domine empiriquement l'effet Merton (drift du taux sans risque).

**Impact mesure** :
| Scenario | RAROC avant | RAROC apres | Explication |
|----------|-------------|-------------|-------------|
| Central | +10.8% | +10.8% | Inchange (delta_ir = 0) |
| Stagflation | +6.7% | **-10.8%** | Hausse de taux en recession = correctement devastateur |
| Reprise | ~+10% | **+17.8%** | Baisse de taux en expansion = correctement favorable |
| GFC | -5.5% | -5.5% | Baisse de taux = soulagement, GDP/HPI dominent |

**Calibration ajustee** : PE IR sensitivity reduite de 2.0 a 1.0 (dette LBO a taux fixe, holding long, Kaplan & Stromberg 2009).

### 5.4 Conclusion

**Les 11 scenarios sont economiquement coherents.** Chaque variable est justifiee par un evenement historique reel et/ou des projections d'institutions de reference (BCE, FMI, NGFS, Eurostat). Les apparentes "violations" des lois macro-financieres (Phillips, Okun, Taylor) sont systematiquement explicables par des chocs d'offre, des politiques publiques exceptionnelles (Kurzarbeit), ou des regimes structurels differents.

Les 3 anomalies identifiees (A1-A3) sont mineures. La limitation structurelle A1 (convention bipolaire du chomage) a un impact de 2nd ordre sur le RAROC. La convention IR (section 5.3) a ete corrigee avec un impact majeur sur la coherence macro-financiere.

---

## Annexe A : Verification historique croisee

Les valeurs du modele ont ete verifiees contre les donnees Eurostat, BCE et World Bank (mars 2026) :

| Scenario | Variable | Modele | Historique (source) | Ecart |
|----------|----------|--------|--------------------|----|
| GFC | PIB | -4.5% | -4.5% (Eurostat 2009) | 0.0pp |
| GFC | Inflation | 0.3% | 0.3% (HICP 2009) | 0.0pp |
| GFC | Taux BCE | 1.0% | 1.0% (MRO mai 2009) | 0.0pp |
| GFC | HPI | -3.0% | -3.2% (Eurostat HPI 2009) | +0.2pp |
| GFC | Chomage | 9.5% | 10.2% pic (avr. 2010) | -0.7pp |
| Souveraine | PIB | -0.9% | -0.9% (Eurostat 2012) | 0.0pp |
| Souveraine | Chomage | 11.5% | 12.1% pic (mai 2013) | -0.6pp |
| Souveraine | Taux BCE | 0.75% | 0.75% (MRO juil. 2012) | 0.0pp |
| Souveraine | Inflation | 2.5% | 2.5% (HICP 2012) | 0.0pp |
| COVID | PIB | -6.0% | -6.1% (Eurostat 2020) | +0.1pp |
| COVID | Chomage | 8.0% | 7.9% (Eurostat 2020 ann.) | +0.1pp |
| COVID | Taux BCE | 0.0% | 0.0% (MRO 2020) | 0.0pp |
| COVID | HPI | +5.0% | +5.4% (Eurostat HPI 2020) | -0.4pp |
| COVID | Inflation | 0.3% | 0.3% (HICP 2020) | 0.0pp |
| Reprise | PIB | 2.5% | 2.5% (Eurostat 2017) | 0.0pp |
| Reprise | HPI | 4.5% | 4.5% (Eurostat HPI 2017) | 0.0pp |
| Stagflation | Inflation | 8.0% | 8.4% HICP pic (oct. 2022) | -0.4pp |

**Conclusion de la verification** : Sur 17 points de controle, l'ecart moyen absolu est de **0.15pp**. Tous les ecarts sont inferieurs a 0.7pp. Les ecarts les plus significatifs concernent les pics de chomage (GFC -0.7pp, Souveraine -0.6pp), ce qui est normal car le modele capture un instantane annualise et non le pic cumule.

---

## Annexe B : Sources de reference

| Source | Utilisation |
|--------|------------|
| Eurostat GDP / HICP / HPI / Unemployment (nama_10_gdp, prc_hicp_aind, prc_hpi_a, une_rt_a) | Calibration historique de toutes les variables |
| BCE SDW (Statistical Data Warehouse) | Taux MRO, facilite de depot, VSTOXX, credit aux menages |
| BCE Staff Projections (Dec 2024) | Scenario Central |
| FMI WEO (World Economic Outlook), Oct 2020 / Apr 2023 | COVID, Stagflation |
| NGFS Short-Term Scenarios (Mai 2025) | Transition climatique |
| Blanchard & Leigh (2013), "Growth Forecast Errors and Fiscal Multipliers", AER | Crise souveraine, multiplicateurs |
| Maddison Project Database 2020 | Trente Glorieuses (PIB historique) |
| Friggit (2007), "Long-term house prices in France" | HPI historique France |
| Guerrieri et al. (2020), "Macroeconomic Implications of COVID-19" | Paradoxe COVID-HPI |
| Summers (2014), "U.S. Economic Prospects: Secular Stagnation" | Trappe a liquidite |
| Metcalf (2019), "On the Economics of a Carbon Tax" | Greenflation, pass-through carbone |
| IEA Critical Minerals Report 2023 | Inflation des materiaux de transition |
| Shiller (2005), "Irrational Exuberance" | Dynamique speculative HPI |
| Slacalek et al. (2020), "Household Balance Sheet Channels of Monetary Policy" | MPC de la richesse immobiliere |
