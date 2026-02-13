"""Generateur de journal d'audit LaTeX pour le pipeline IFRS 9.

Produit un document .tex compilable (pdflatex) qui repertorie
chronologiquement tous les calculs effectues par le pipeline,
avec formules mathematiques, valeurs numeriques et explications.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


# ── Helpers LaTeX ──────────────────────────────────────────


def _esc(text: str) -> str:
    """Echappe les caracteres speciaux LaTeX."""
    for ch, repl in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]:
        text = text.replace(ch, repl)
    return text


def _fmt_euro(val: float) -> str:
    """Formate un montant en EUR lisible LaTeX."""
    if abs(val) >= 1e9:
        return f"{val / 1e9:,.2f}~Md~EUR"
    if abs(val) >= 1e6:
        return f"{val / 1e6:,.2f}~M~EUR"
    return f"{val:,.0f}~EUR"


def _fmt_pct(val: float, decimals: int = 2) -> str:
    """Formate un ratio en pourcentage."""
    return f"{val * 100:.{decimals}f}\\%"


def _fmt_num(val: float, decimals: int = 2) -> str:
    """Formate un nombre."""
    return f"{val:.{decimals}f}"


def _safe_get(df: Optional[pd.DataFrame], col: str, agg: str = "mean") -> float:
    """Recupere une valeur agregee depuis un DataFrame, 0 si absent."""
    if df is None or col not in df.columns:
        return 0.0
    s = df[col]
    if agg == "sum":
        return float(s.sum())
    if agg == "mean":
        return float(s.mean())
    return 0.0


# ── Sections du document ──────────────────────────────────


def _preamble() -> str:
    return r"""\documentclass[a4paper,11pt]{article}

% Encodage et langue
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[french]{babel}

% Mise en page
\usepackage[margin=2.5cm]{geometry}
\usepackage{fancyhdr}
\usepackage{lastpage}

% Mathematiques
\usepackage{amsmath}
\usepackage{amssymb}

% Tableaux
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}

% Couleurs et liens
\usepackage[dvipsnames]{xcolor}
\usepackage[colorlinks=true,linkcolor=MidnightBlue,urlcolor=MidnightBlue]{hyperref}

% En-tete / pied de page
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\textsc{IFRS 9 Risk Cockpit -- Journal d'Audit}}
\fancyhead[R]{\small\thepage/\pageref{LastPage}}
\fancyfoot[C]{\small\textit{Document g\'en\'er\'e automatiquement -- ne pas modifier}}
\renewcommand{\headrulewidth}{0.4pt}

% Commandes utilitaires
\newcommand{\formule}[1]{\begin{equation}#1\end{equation}}
\newcommand{\objectif}[1]{\paragraph{Objectif.}#1}
\newcommand{\interpretation}[1]{\paragraph{Interpr\'etation.}#1}

"""


def _title_page(macro_params: dict, selected_model: str) -> str:
    date_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    scenario_name = macro_params.get("scenario_name", "Custom")

    lines = [
        r"\begin{document}",
        "",
        r"\begin{titlepage}",
        r"\centering",
        r"\vspace*{3cm}",
        r"{\Huge\bfseries Journal d'Audit des Calculs\\[0.5em]",
        r"IFRS 9 Risk Cockpit}\\[2cm]",
        r"{\Large Pipeline complet -- toutes \'etapes document\'ees}\\[1cm]",
        f"\\large Date de g\\'en\\'eration : {date_str}\\\\[0.5cm]",
        f"\\large Sc\\'enario macro : \\textbf{{{_esc(scenario_name)}}}\\\\[0.3cm]",
        f"\\large Mod\\`ele PD : \\textbf{{{_esc(selected_model)}}}\\\\[2cm]",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"\textbf{Variable macro} & \textbf{Valeur} \\",
        r"\midrule",
    ]

    macro_labels = {
        "unemployment_rate": ("Ch\\^omage", "\\%"),
        "gdp_growth": ("Croissance PIB", "\\%"),
        "interest_rate": ("Taux BCE", "\\%"),
        "hpi_growth": ("Prix immobiliers", "\\%"),
        "inflation_rate": ("Inflation", "\\%"),
    }
    for key, (label, unit) in macro_labels.items():
        val = macro_params.get(key, 0.0)
        lines.append(f"{label} & {val:.2f}{unit} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vfill",
        r"{\small Ce document retrace chronologiquement l'ensemble des calculs",
        r"effectu\'es par le pipeline IFRS~9, avec les formules math\'ematiques,",
        r"les valeurs num\'eriques r\'eelles et une explication p\'edagogique",
        r"\`a chaque \'etape.}",
        r"\end{titlepage}",
        "",
        r"\tableofcontents",
        r"\newpage",
        "",
    ]
    return "\n".join(lines)


def _section_parametres(macro_params: dict, selected_model: str) -> str:
    lines = [
        r"\section{Param\`etres d'Entr\'ee}",
        "",
        r"\objectif{Cette section r\'ecapitule les hypoth\`eses macro\'economiques",
        r"et le mod\`ele PD choisi. Tous les calculs en aval d\'ependent de ces param\`etres.}",
        "",
        r"\subsection{Variables macro\'economiques}",
        "",
        r"Les 5 variables macro alimentent simultan\'ement le canal cr\'edit (PD, LGD)",
        r"et le canal PE (NAV, multiples). Elles sont converties en chocs relatifs par rapport",
        r"au sc\'enario de base (SCENARIO\_BASE) avant injection dans le pipeline.",
        "",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Param\`etres macro et sc\'enario de base}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Variable} & \textbf{Valeur stress\'ee} & \textbf{Base} & \textbf{\'Ecart} \\",
        r"\midrule",
    ]

    from ifrs9_cockpit.config import SCENARIO_BASE
    base_vals = {
        "unemployment_rate": SCENARIO_BASE.unemployment_rate,
        "gdp_growth": SCENARIO_BASE.gdp_growth,
        "interest_rate": SCENARIO_BASE.interest_rate,
        "hpi_growth": SCENARIO_BASE.hpi_growth,
        "inflation_rate": SCENARIO_BASE.inflation_rate,
    }
    labels = {
        "unemployment_rate": "Ch\\^omage (\\%)",
        "gdp_growth": "PIB (\\%)",
        "interest_rate": "Taux BCE (\\%)",
        "hpi_growth": "HPI (\\%)",
        "inflation_rate": "Inflation (\\%)",
    }
    for key in base_vals:
        val = macro_params.get(key, base_vals[key])
        base = base_vals[key]
        delta = val - base
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"{labels[key]} & {val:.2f} & {base:.2f} & {sign}{delta:.2f} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\subsection{Mod\`ele PD s\'electionn\'e}",
        "",
        f"Mod\\`ele utilis\\'e : \\textbf{{{_esc(selected_model)}}}.",
        r"La PD 12 mois est produite par ce mod\`ele, puis stress\'ee via la transformation",
        r"logit--Vasicek (cf.\ section suivante).",
        "",
    ]
    return "\n".join(lines)


def _section_staging(result_stressed: pd.DataFrame) -> str:
    lines = [
        r"\section{Staging IFRS 9}",
        "",
        r"\objectif{Le staging IFRS~9 (norme \S5.5.1 -- \S5.5.20) classifie chaque exposition",
        r"en trois \'etapes selon la d\'et\'erioration significative du risque de cr\'edit (SICR).}",
        "",
        r"\subsection{Score SICR multi-facteurs (B5.5.17)}",
        "",
        r"Le passage en Stage~2 repose sur un score composite int\'egrant quatre facteurs :",
        "",
        r"\formule{",
        r"\text{SICR\_score} = w_1 \cdot \left(\frac{PD_{\text{current}}}{PD_{\text{origination}}} - 1\right)",
        r"+ w_2 \cdot \max\!\left(0,\; PD_{\text{current}} - PD_{\text{origination}}\right)",
        r"+ w_3 \cdot \frac{DPD}{30}",
        r"+ w_4 \cdot z_{\text{macro}}",
        r"}",
        "",
    ]

    from ifrs9_cockpit.config import SICR_CONFIG, IFRS9_CONFIG
    lines += [
        r"\noindent Pond\'erations calibr\'ees :",
        f"$w_1 = {SICR_CONFIG.w_pd_ratio}$ (ratio PD),",
        f"$w_2 = {SICR_CONFIG.w_pd_delta}$ (delta PD),",
        f"$w_3 = {SICR_CONFIG.w_dpd}$ (DPD),",
        f"$w_4 = {SICR_CONFIG.w_macro}$ (macro).\\\\",
        f"Seuil SICR : $\\tau = {SICR_CONFIG.threshold}$.\\\\",
        f"Stage~3 : DPD $\\geq {IFRS9_CONFIG.stage3_dpd_threshold}$j "
        f"ou PD $\\geq {_fmt_pct(IFRS9_CONFIG.stage3_pd_threshold, 0)}$.",
        "",
    ]

    # Distribution des stages
    if "stage" in result_stressed.columns:
        total = len(result_stressed)
        stage_counts = result_stressed["stage"].value_counts().sort_index()
        lines += [
            r"\subsection{Distribution des stages (apr\`es stress)}",
            "",
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{R\'epartition du portefeuille par stage}",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"\textbf{Stage} & \textbf{Nombre} & \textbf{Part (\%)} & \textbf{EAD (EUR)} \\",
            r"\midrule",
        ]
        for stage in [1, 2, 3]:
            n = int(stage_counts.get(stage, 0))
            pct = n / total * 100 if total > 0 else 0
            ead_stage = 0.0
            if "ead" in result_stressed.columns:
                ead_stage = float(
                    result_stressed.loc[result_stressed["stage"] == stage, "ead"].sum()
                )
            lines.append(
                f"Stage {stage} & {n:,} & {pct:.1f}\\% & {_fmt_euro(ead_stage)} \\\\"
            )
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]

    lines += [
        r"\interpretation{La norme IFRS~9 impose une d\'egradation progressive des provisions :",
        r"Stage~1 = 12 mois de pertes attendues, Stage~2 = lifetime, Stage~3 = d\'efaut av\'er\'e.",
        r"Un score SICR \'elev\'e traduit une d\'et\'erioration significative multi-crit\`eres,",
        r"allant au-del\`a du simple doublement de PD.}",
        "",
    ]
    return "\n".join(lines)


def _section_ecl_credit(
    result_stressed: pd.DataFrame,
    result_base: pd.DataFrame,
) -> str:
    lines = [
        r"\section{Calcul ECL Cr\'edit}",
        "",
        r"\objectif{L'Expected Credit Loss (ECL) est la mesure centrale d'IFRS~9.",
        r"Elle combine PD, LGD, EAD et facteur d'actualisation, pond\'er\'es sur 3 sc\'enarios",
        r"macro (50\%/25\%/25\%) conform\'ement \`a IFRS~9 \S5.5.17.}",
        "",
    ]

    # 3.1 PD stress logit-space
    lines += [
        r"\subsection{PD stress\'ee (logit-space, Merton--Vasicek)}",
        "",
        r"La PD point-in-time est obtenue en translatant la PD TTC dans l'espace logit :",
        "",
        r"\formule{",
        r"PD_{\text{stressed}} = \text{expit}\!\left(",
        r"\text{logit}(PD_{\text{current}}) + \text{shock}_{\text{composite}} \times A",
        r"\right)}",
        "",
    ]
    from ifrs9_cockpit.config import LOGIT_AMPLITUDE
    lines += [
        f"o\\`u $A = {LOGIT_AMPLITUDE}$ (amplitude logit calibr\\'ee EBA).",
        r"Le choc composite agr\`ege les 5 variables macro pond\'er\'ees par les sensibilit\'es sectorielles.",
        "",
    ]

    # 3.2 PD lifetime hazard rate
    lines += [
        r"\subsection{PD lifetime (taux de hasard, correction C1)}",
        "",
        r"La PD cumulative sur $T$ ann\'ees est calcul\'ee via le taux de hasard instantan\'e",
        r"(et non la formule de Bernoulli $(1-PD)^T$ qui sous-estime le risque) :",
        "",
        r"\formule{",
        r"h = -\ln(1 - PD_{12m}), \qquad PD_{\text{lifetime}} = 1 - e^{-h \times T}",
        r"}",
        "",
        r"Cette approche est coh\'erente avec les mod\`eles de survie et converge vers le discret",
        r"quand $PD \to 0$.",
        "",
    ]

    # 3.3 LGD Downturn
    from ifrs9_cockpit.config import SECTORS
    lines += [
        r"\subsection{LGD Downturn (corr\'elation cycle, correction H3)}",
        "",
        r"La LGD Through-the-Cycle est ajust\'ee en downturn via la corr\'elation LGD--cycle",
        r"(EBA GL/2019/03) :",
        "",
        r"\formule{",
        r"LGD_{\text{DT}} = LGD_{\text{TTC}} \times \left(1 + \rho_{\text{LGD}} \times |Z_{\text{stress}}|\right)",
        r"}",
        "",
        r"\noindent Valeurs de $\rho_{\text{LGD}}$ par secteur :",
        r"\begin{itemize}",
    ]
    for sec in SECTORS:
        lines.append(f"  \\item {_esc(sec.name)} : $\\rho = {sec.rho_lgd_cycle}$")
    lines += [
        r"\end{itemize}",
        "",
    ]

    # 3.4 Discount factor PD-weighted
    from ifrs9_cockpit.config import IFRS9_CONFIG
    lines += [
        r"\subsection{Facteur d'actualisation PD-pond\'er\'e (correction C2)}",
        "",
        r"Le facteur d'actualisation int\`egre la probabilit\'e de survie pour \'eviter",
        r"de suractualiser les flux non re\c{c}us en cas de d\'efaut :",
        "",
        r"\formule{",
        r"DF = \frac{1}{T} \sum_{t=1}^{T} \frac{1 - PD_{\text{cum}}(t)}{(1+r)^t}",
        r"}",
        "",
        f"avec $r = {_fmt_pct(IFRS9_CONFIG.discount_rate)}$ (EIR proxy) "
        f"et $T = {IFRS9_CONFIG.lifetime_horizon_years}$ ans.",
        "",
    ]

    # 3.5 ECL formula
    lines += [
        r"\subsection{ECL pond\'er\'e (3 sc\'enarios)}",
        "",
        r"L'ECL final est la somme pond\'er\'ee sur trois sc\'enarios macro :",
        "",
        r"\formule{",
        r"ECL = \sum_{s \in \{B, A, F\}} w_s \times PD_s \times LGD_s \times EAD \times DF_s",
        r"}",
        "",
        r"avec $w_B = 50\%$, $w_A = 25\%$, $w_F = 25\%$ (Base, Adverse, Favorable).",
        "",
    ]

    # 3.6 Credit spread Merton
    from ifrs9_cockpit.config import BASEL_CONFIG
    lines += [
        r"\subsection{Spread Merton (correction H4)}",
        "",
        r"Le spread de cr\'edit th\'eorique est d\'eriv\'e du mod\`ele structurel de Merton :",
        "",
        r"\formule{",
        r"s = -\frac{\ln(1 - PD \times LGD)}{T} + \lambda_{\text{liq}}",
        r"}",
        "",
        f"avec $\\lambda_{{\\text{{liq}}}} = "
        f"{BASEL_CONFIG.liquidity_premium_bps}$~bps (prime de liquidit\\'e).",
        "",
    ]

    # Tableau recapitulatif par secteur
    if "sector" in result_stressed.columns:
        lines += [
            r"\subsection{R\'ecapitulatif par secteur}",
            "",
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{M\'etriques ECL agr\'eg\'ees par secteur}",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"\textbf{Secteur} & \textbf{EAD} & \textbf{PD moy.} & \textbf{LGD moy.}",
            r"& \textbf{ECL} & \textbf{Spread} & \textbf{Stage 2+3} \\",
            r"\midrule",
        ]
        for sec_name, grp in result_stressed.groupby("sector"):
            ead_s = grp["ead"].sum() if "ead" in grp.columns else 0
            pd_m = grp["pd_12m"].mean() if "pd_12m" in grp.columns else 0
            lgd_m = grp["lgd"].mean() if "lgd" in grp.columns else 0
            ecl_s = grp["ecl_weighted"].sum() if "ecl_weighted" in grp.columns else 0
            spread = grp["credit_spread"].mean() if "credit_spread" in grp.columns else 0
            s23 = 0
            if "stage" in grp.columns:
                s23 = (grp["stage"] >= 2).sum() / len(grp) if len(grp) > 0 else 0
            lines.append(
                f"{_esc(str(sec_name))} & {_fmt_euro(ead_s)} & {_fmt_pct(pd_m)} "
                f"& {_fmt_pct(lgd_m)} & {_fmt_euro(ecl_s)} & {_fmt_num(spread * 10000, 0)}bps "
                f"& {_fmt_pct(s23)} \\\\"
            )

        # Total
        ead_t = result_stressed["ead"].sum() if "ead" in result_stressed.columns else 0
        ecl_t = result_stressed["ecl_weighted"].sum() if "ecl_weighted" in result_stressed.columns else 0
        pd_t = result_stressed["pd_12m"].mean() if "pd_12m" in result_stressed.columns else 0
        lgd_t = result_stressed["lgd"].mean() if "lgd" in result_stressed.columns else 0
        lines += [
            r"\midrule",
            f"\\textbf{{Total}} & {_fmt_euro(ead_t)} & {_fmt_pct(pd_t)} "
            f"& {_fmt_pct(lgd_t)} & {_fmt_euro(ecl_t)} & -- & -- \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]

    lines += [
        r"\interpretation{L'ECL pond\'er\'e capture l'asym\'etrie entre sc\'enarios",
        r"favorables et adverses. La transformation logit garantit que les PD restent",
        r"dans $[0, 1]$ m\^eme sous stress extr\^eme, et le taux de hasard \'evite",
        r"la sous-estimation de la PD lifetime pour les horizons longs.}",
        "",
    ]
    return "\n".join(lines)


def _section_pe(result_pe: pd.DataFrame) -> str:
    lines = [
        r"\section{Valorisation PE (IFRS 13)}",
        "",
        r"\objectif{Les participations Private Equity sont valoris\'ees en juste valeur",
        r"(IFRS~13) selon les m\'ethodes IPEV. Le pipeline calcule la NAV stress\'ee,",
        r"la probabilit\'e de distress (logit-space) et la perte attendue PE.}",
        "",
    ]

    # NAV
    lines += [
        r"\subsection{NAV stress\'ee (m\'ethodes IPEV)}",
        "",
        r"La NAV de chaque position est stress\'ee via une fonction exponentielle :",
        "",
        r"\formule{",
        r"NAV_{\text{stressed}} = NAV_{\text{base}} \times \exp\!\left(",
        r"-\text{shock}_{\text{composite}} \times \text{scale}",
        r"\right)}",
        "",
        r"La m\'ethode IPEV d\'epend du secteur (EV/Revenue, EV/EBITDA, Cap Rate/NOI).",
        "",
    ]

    # MOIC/IRR
    lines += [
        r"\subsection{MOIC et IRR}",
        "",
        r"\formule{",
        r"MOIC = \frac{NAV}{Capital\_invest\hspace{-0.5pt}i}, \qquad",
        r"IRR = MOIC^{1/T} - 1",
        r"}",
        "",
    ]

    # P(distress)
    from ifrs9_cockpit.config import PE_DISTRESS_LOGIT_SCALE
    lines += [
        r"\subsection{P(distress) logit-space}",
        "",
        r"La probabilit\'e de distress est calcul\'ee dans l'espace logit :",
        "",
        r"\formule{",
        r"P(\text{distress}) = \text{expit}\!\left(",
        r"-2 \ln(MOIC) + \text{stress}_{\text{PE}} \times S",
        r"\right)}",
        "",
        f"avec $S = {PE_DISTRESS_LOGIT_SCALE}$ (\\'echelle logit PE).",
        "",
    ]

    # Expected Loss PE
    from ifrs9_cockpit.config import PE_CLASSIFICATION_CONFIG
    lines += [
        r"\subsection{Expected Loss PE}",
        "",
        r"\formule{",
        r"EL_{\text{PE}} = P(\text{distress}) \times LGD_{\text{equity}} \times NAV",
        r"}",
        "",
        f"avec $LGD_{{\\text{{equity}}}} = {_fmt_pct(PE_CLASSIFICATION_CONFIG.lgd_equity)}$.",
        "",
    ]

    # CRR3 composite
    lines += [
        r"\subsection{Risk Weight CRR3 composite (correction H8)}",
        "",
        r"Le Risk Weight PE est d\'etermin\'e par un score composite CRR3",
        r"(CRR3 Art.~133/155), et non par un seuil unique de P(distress) :",
        "",
        r"\formule{",
        r"\text{Score} = 0.4 \times P(\text{distress}) + 0.3 \times (1 - MOIC_{\text{norm}})",
        r"+ 0.2 \times \text{drawdown} + 0.1 \times \text{illiquidit\'e}",
        r"}",
        "",
        r"Grille CRR3 : Score $< 0.25 \Rightarrow 190\%$, "
        r"Score $< 0.50 \Rightarrow 250\%$, "
        r"Score $\geq 0.50 \Rightarrow 400\%$.",
        "",
    ]

    # Tableau par secteur
    if "sector" in result_pe.columns:
        lines += [
            r"\subsection{R\'ecapitulatif PE par secteur}",
            "",
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{M\'etriques PE agr\'eg\'ees par secteur}",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"\textbf{Secteur} & \textbf{NAV} & \textbf{MOIC} & \textbf{P(dist.)}",
            r"& \textbf{EL PE} & \textbf{RWA PE} & \textbf{Cat\'eg.} \\",
            r"\midrule",
        ]
        for sec_name, grp in result_pe.groupby("sector"):
            nav_s = grp["nav"].sum() if "nav" in grp.columns else 0
            moic_m = grp["moic"].mean() if "moic" in grp.columns else 0
            dist_m = grp["distress_prob"].mean() if "distress_prob" in grp.columns else 0
            el_s = grp["expected_loss_pe"].sum() if "expected_loss_pe" in grp.columns else 0
            rwa_s = grp["rwa_pe"].sum() if "rwa_pe" in grp.columns else 0
            # Most frequent category
            cat = "N/A"
            if "risk_category" in grp.columns:
                cat = grp["risk_category"].mode().iloc[0] if len(grp) > 0 else "N/A"
            lines.append(
                f"{_esc(str(sec_name))} & {_fmt_euro(nav_s)} & {_fmt_num(moic_m)} "
                f"& {_fmt_pct(dist_m)} & {_fmt_euro(el_s)} & {_fmt_euro(rwa_s)} "
                f"& {_esc(str(cat))} \\\\"
            )

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]

    lines += [
        r"\interpretation{La valorisation PE en logit-space capture la non-lin\'earit\'e",
        r"entre MOIC et risque de distress. Le score CRR3 composite remplace",
        r"l'ancien seuillage binaire par une \'evaluation multi-crit\`eres",
        r"conforme aux exigences CRR3 de janvier 2025.}",
        "",
    ]
    return "\n".join(lines)


def _section_metriques_avancees(
    advanced_metrics: pd.DataFrame,
    hhi_cross: dict,
    raroc_eva: pd.DataFrame,
    asymmetry_matrix: pd.DataFrame,
) -> str:
    lines = [
        r"\section{M\'etriques Avanc\'ees}",
        "",
        r"\objectif{Cette section pr\'esente les indicateurs de risque avanc\'es",
        r"qui compl\`etent l'ECL : ratios NPL, co\^ut du risque, concentration (HHI),",
        r"RAROC/EVA et matrice d'asym\'etrie cr\'edit/PE.}",
        "",
    ]

    # NPL, Cost of Risk, Texas
    lines += [
        r"\subsection{Ratios prudentiels (NPL, Cost of Risk, Texas)}",
        "",
        r"\formule{",
        r"\text{NPL} = \frac{EAD_{\text{Stage 3}}}{EAD_{\text{total}}}, \qquad",
        r"\text{CoR} = \frac{ECL}{EAD \times T} \times 10\,000 \;\text{(bps)}",
        r"}",
        "",
        r"Le Texas Ratio (correction C3) est restreint au Stage~3 :",
        "",
        r"\formule{",
        r"\text{Texas} = \frac{EAD_{\text{Stage 3}}}{ECL_{\text{Stage 3}} + \text{Capital}_{\text{share}}}",
        r"}",
        "",
    ]

    if advanced_metrics is not None and len(advanced_metrics) > 0:
        lines += [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Ratios prudentiels par secteur}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"\textbf{Secteur} & \textbf{NPL} & \textbf{CoR (bps)} & \textbf{Coverage}",
            r"& \textbf{Texas} \\",
            r"\midrule",
        ]
        for _, row in advanced_metrics.iterrows():
            sec = str(row.get("sector", ""))
            npl = row.get("npl_ratio", 0)
            cor = row.get("cost_of_risk_bps", 0)
            cov = row.get("coverage_ratio", 0)
            tex = row.get("texas_ratio_synth", row.get("texas_ratio", 0))
            lines.append(
                f"{_esc(sec)} & {_fmt_pct(npl)} & {_fmt_num(cor, 1)} "
                f"& {_fmt_pct(cov)} & {_fmt_num(tex)} \\\\"
            )
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]

    # HHI cross-cell
    hhi_val = hhi_cross.get("hhi_crosscell", 0) if hhi_cross else 0
    lines += [
        r"\subsection{HHI cross-cell (concentration)}",
        "",
        r"L'indice HHI mesure la concentration du portefeuille sur 10 cellules (5 secteurs $\times$ 2 canaux) :",
        "",
        r"\formule{",
        r"HHI = \sum_{i=1}^{10} \left(\frac{EAD_i}{EAD_{\text{total}}} \times 100\right)^2",
        r"}",
        "",
        f"\\textbf{{HHI calcul\\'e}} : {_fmt_num(hhi_val, 0)}.",
        r"Seuil r\'eglementaire : $< 1500$ (diversifi\'e), $1500$--$2500$ (mod\'er\'e), $> 2500$ (concentr\'e).",
        "",
    ]

    # RAROC / EVA
    from ifrs9_cockpit.config import BASEL_CONFIG
    lines += [
        r"\subsection{RAROC / EVA (correction H5)}",
        "",
        r"Le RAROC complet int\`egre le CIR et l'imp\^ot sur les soci\'et\'es :",
        "",
        r"\formule{",
        r"RAROC = \frac{(\text{Revenue} \times (1 - CIR) - EL) \times (1 - \tau)}",
        r"{\text{Capital}\_\text{allou\'e}}",
        r"}",
        "",
        f"avec $CIR = {_fmt_pct(BASEL_CONFIG.cir)}$ et $\\tau = {_fmt_pct(BASEL_CONFIG.tax_rate)}$.",
        "",
        r"\formule{",
        r"EVA = (\text{RAROC} - k_e) \times \text{Capital}\_\text{allou\'e}",
        r"}",
        "",
    ]

    if raroc_eva is not None and len(raroc_eva) > 0:
        lines += [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{RAROC et EVA par cellule}",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"\textbf{Secteur} & \textbf{Canal} & \textbf{Rev. net} & \textbf{EL}",
            r"& \textbf{RAROC} & \textbf{EVA} \\",
            r"\midrule",
        ]
        for _, row in raroc_eva.iterrows():
            sec = str(row.get("sector", ""))
            canal = str(row.get("canal", ""))
            rev_net = row.get("revenue_net", 0)
            el = row.get("expected_loss", 0)
            raroc = row.get("raroc", 0)
            eva = row.get("eva", 0)
            lines.append(
                f"{_esc(sec)} & {_esc(canal)} & {_fmt_euro(rev_net)} "
                f"& {_fmt_euro(el)} & {_fmt_pct(raroc)} & {_fmt_euro(eva)} \\\\"
            )
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]

    # Asymmetry matrix
    if asymmetry_matrix is not None and len(asymmetry_matrix) > 0:
        lines += [
            r"\subsection{Matrice d'asym\'etrie cr\'edit / PE}",
            "",
            r"La matrice d'asym\'etrie compare les profils de risque cr\'edit et PE par secteur.",
            r"Un ratio $> 1$ signale que le canal PE est plus expos\'e que le cr\'edit.",
            "",
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Asym\'etrie par secteur (ECL vs EL PE, RWA, RAROC)}",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"\textbf{Secteur} & \textbf{ECL cr\'edit} & \textbf{EL PE}",
            r"& \textbf{Loss ratio} & \textbf{RAROC cr.} & \textbf{RAROC PE}",
            r"& \textbf{$\Delta$ RAROC} \\",
            r"\midrule",
        ]
        for _, row in asymmetry_matrix.iterrows():
            sec = str(row.get("sector", ""))
            ecl_c = row.get("ecl_credit", 0)
            el_pe = row.get("el_pe", 0)
            lr = row.get("loss_ratio", 0)
            rar_c = row.get("raroc_credit", 0)
            rar_pe = row.get("raroc_pe", 0)
            delta = row.get("raroc_delta", 0)
            lines.append(
                f"{_esc(sec)} & {_fmt_euro(ecl_c)} & {_fmt_euro(el_pe)} "
                f"& {_fmt_num(lr)} & {_fmt_pct(rar_c)} & {_fmt_pct(rar_pe)} "
                f"& {_fmt_pct(delta)} \\\\"
            )
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]

    lines += [
        r"\interpretation{Le HHI cross-cell capture la concentration jointe cr\'edit/PE.",
        r"Le RAROC int\`egre CIR et fiscalit\'e pour une mesure de rentabilit\'e",
        r"ajust\'ee du risque comparable entre canaux et secteurs.}",
        "",
    ]
    return "\n".join(lines)


def _section_optimisation(
    optimization: dict,
    crr3_sensitivity: pd.DataFrame,
) -> str:
    lines = [
        r"\section{Optimisation \& CRR3}",
        "",
        r"\objectif{Le module d'optimisation d\'etermine l'allocation optimale",
        r"cr\'edit/PE sous contraintes prudentielles CRR3, en maximisant le RAROC",
        r"sous budget RWA et ratio CET1.}",
        "",
    ]

    # Softmax allocation
    lines += [
        r"\subsection{Allocation softmax adaptative (correction M7)}",
        "",
        r"Les poids sectoriels sont calcul\'es via softmax avec \'echelle adaptative :",
        "",
        r"\formule{",
        r"w_i = \frac{\exp(\beta \times RAROC_i)}{\sum_j \exp(\beta \times RAROC_j)},",
        r"\qquad \beta = \text{clip}\!\left(\frac{1}{\text{Var}(RAROC)},\; 2,\; 50\right)",
        r"}",
        "",
        r"L'\'echelle $\beta$ s'adapte \`a la dispersion des RAROC : forte dispersion $\to$",
        r"$\beta$ faible (diversification), faible dispersion $\to$ $\beta$ \'elev\'e (concentration).",
        "",
    ]

    if optimization:
        credit_alloc = optimization.get("credit_allocation", 0)
        pe_alloc = optimization.get("pe_allocation", 0)
        rwa_w = optimization.get("rwa_weighted", 0)
        cet1 = optimization.get("cet1_ratio", 0)
        headroom = optimization.get("cet1_headroom", 0)
        lines += [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Allocation optimale cr\'edit / PE}",
            r"\begin{tabular}{lr}",
            r"\toprule",
            r"\textbf{M\'etrique} & \textbf{Valeur} \\",
            r"\midrule",
            f"Allocation cr\\'edit & {_fmt_pct(credit_alloc)} \\\\",
            f"Allocation PE & {_fmt_pct(pe_alloc)} \\\\",
            f"RWA pond\\'er\\'e & {_fmt_euro(rwa_w)} \\\\",
            f"Ratio CET1 & {_fmt_pct(cet1)} \\\\",
            f"Headroom CET1 & {_fmt_pct(headroom)} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]

    # CRR3 sensitivity
    lines += [
        r"\subsection{Sensibilit\'e CRR3 (3 options RW PE)}",
        "",
        r"La CRR3 pr\'evoit trois classifications pour le Risk Weight PE :",
        r"190\% (IRB diversifi\'e), 250\% (equity g\'en\'eral), 400\% (sp\'eculatif).",
        "",
    ]

    if crr3_sensitivity is not None and len(crr3_sensitivity) > 0:
        lines += [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Sensibilit\'e CRR3 : impact du RW PE sur le ratio CET1}",
            r"\begin{tabular}{rrrr}",
            r"\toprule",
            r"\textbf{RW PE (\%)} & \textbf{RWA total} & \textbf{CET1} & \textbf{Headroom} \\",
            r"\midrule",
        ]
        for _, row in crr3_sensitivity.iterrows():
            rw = row.get("rw_pe", 0)
            rwa_t = row.get("rwa_total", 0)
            c1 = row.get("cet1_ratio", 0)
            hr = row.get("headroom", 0)
            lines.append(
                f"{_fmt_num(rw, 0)}\\% & {_fmt_euro(rwa_t)} "
                f"& {_fmt_pct(c1)} & {_fmt_pct(hr)} \\\\"
            )
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]

    lines += [
        r"\interpretation{L'optimisation softmax adaptative \'evite l'hyper-concentration",
        r"sur un seul secteur rentable en ajustant l'\'echelle $\beta$ \`a la dispersion",
        r"des RAROC. La sensibilit\'e CRR3 montre l'impact de la classification PE",
        r"sur le ratio CET1 et le headroom prudentiel.}",
        "",
    ]
    return "\n".join(lines)


def _section_ai_analyst(
    analytics_state,
    trans_matrix: pd.DataFrame,
) -> str:
    lines = [
        r"\section{AI Analyst (2 passes)}",
        "",
        r"\objectif{Le moteur AI Analyst ex\'ecute une analyse CRO \`a 5 couches",
        r"(Croisement, Allocation Proportionnelle, Tipping Points + RST, R\'egime, Prospective)",
        r"en 2 passes it\'eratives pour affiner les recommandations.}",
        "",
    ]

    # Regime
    regime_name = "Non d\\'etect\\'e"
    regime_probs = {}
    if analytics_state and analytics_state.regime:
        regime_name = _esc(analytics_state.regime.detected_regime)
        regime_probs = analytics_state.regime.probabilities or {}

    lines += [
        r"\subsection{R\'egime macro d\'etect\'e (Couche 4)}",
        "",
        r"La classification de r\'egime utilise un scoring composite sur les 5 variables macro.",
        r"Les probabilit\'es softmax identifient le r\'egime dominant :",
        "",
        f"\\textbf{{R\\'egime d\\'etect\\'e : {regime_name}}}",
        "",
    ]
    if regime_probs:
        lines += [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Probabilit\'es de r\'egime}",
            r"\begin{tabular}{lr}",
            r"\toprule",
            r"\textbf{R\'egime} & \textbf{Probabilit\'e} \\",
            r"\midrule",
        ]
        for reg, prob in sorted(regime_probs.items(), key=lambda x: -x[1]):
            lines.append(f"{_esc(reg)} & {_fmt_pct(prob)} \\\\")
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]

    # Allocation proportionnelle
    lines += [
        r"\subsection{Allocation Proportionnelle (Couche 2)}",
        "",
        r"L'allocation proportionnelle attribue l'ECL total aux 10 cellules",
        r"(5 secteurs $\times$ 2 canaux) :",
        "",
        r"\formule{",
        r"C_i = \frac{EL_i}{\sum_j EL_j} \times ECL_{\text{total}}",
        r"}",
        "",
    ]

    if analytics_state and analytics_state.euler_contributions is not None:
        euler = analytics_state.euler_contributions
        if len(euler) > 0:
            lines += [
                r"\begin{table}[h]",
                r"\centering",
                r"\caption{Contributions par cellule (Allocation Proportionnelle)}",
                r"\begin{tabular}{lrr}",
                r"\toprule",
                r"\textbf{Cellule} & \textbf{Contribution} & \textbf{Part (\%)} \\",
                r"\midrule",
            ]
            # Try to find the right columns
            contrib_col = None
            pct_col = None
            cell_col = None
            for c in euler.columns:
                cl = c.lower()
                if "contrib" in cl and "pct" not in cl and "share" not in cl:
                    contrib_col = c
                if "pct" in cl or "share" in cl:
                    pct_col = c
                if "cell" in cl or "sector" in cl or "name" in cl:
                    cell_col = c
            if cell_col is None and euler.index.name:
                cell_col = "__index__"

            for idx, row in euler.iterrows():
                cell_name = str(row[cell_col]) if cell_col and cell_col != "__index__" else str(idx)
                contrib = float(row[contrib_col]) if contrib_col else 0.0
                pct = float(row[pct_col]) * 100 if pct_col else 0.0
                lines.append(
                    f"{_esc(cell_name)} & {_fmt_euro(contrib)} & {_fmt_num(pct, 1)}\\% \\\\"
                )
            lines += [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]

    # RST + Tipping points
    rst_dist = 0.0
    if analytics_state:
        rst_dist = analytics_state.rst_distance

    lines += [
        r"\subsection{Tipping Points \& RST Mahalanobis (Couches 3, correction H6)}",
        "",
        r"Le reverse stress test identifie le sc\'enario minimal qui ferait d\'epasser",
        r"le seuil ECL cible. La distance est mesur\'ee en sigma de Mahalanobis :",
        "",
        r"\formule{",
        r"d = \sqrt{\mathbf{x}^\top \Sigma^{-1} \mathbf{x}}",
        r"}",
        "",
        r"o\`u $\mathbf{x}$ est le vecteur de chocs macro et $\Sigma$ la matrice de covariance historique.",
        "",
        f"\\textbf{{Distance RST}} : {_fmt_num(rst_dist)} $\\sigma$.",
        "",
    ]

    if analytics_state and analytics_state.tipping_points is not None:
        tp = analytics_state.tipping_points
        if len(tp) > 0:
            lines += [
                r"\begin{table}[h]",
                r"\centering",
                r"\caption{Tipping points par variable/secteur}",
                r"\begin{tabular}{" + "l" * min(len(tp.columns), 6) + r"}",
                r"\toprule",
            ]
            # Header
            cols_to_show = list(tp.columns)[:6]
            header = " & ".join(f"\\textbf{{{_esc(c)}}}" for c in cols_to_show)
            lines.append(f"{header} \\\\")
            lines.append(r"\midrule")
            for _, row in tp.head(10).iterrows():
                vals = " & ".join(
                    _esc(str(round(row[c], 3)) if isinstance(row[c], float) else str(row[c]))
                    for c in cols_to_show
                )
                lines.append(f"{vals} \\\\")
            lines += [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]

    # Trajectories OU
    lines += [
        r"\subsection{Trajectoires Ornstein--Uhlenbeck (correction H10)}",
        "",
        r"Les projections macro \`a T+3/6/9/12 mois utilisent un processus OU",
        r"\`a retour vers la moyenne :",
        "",
        r"\formule{",
        r"x_{t+1} = x_t + \kappa \left(\theta - x_t\right) \Delta t",
        r"+ \sigma \sqrt{\Delta t}\; \varepsilon_t",
        r"}",
        "",
        r"avec $\kappa$ = vitesse de retour, $\theta$ = \'equilibre long terme,",
        r"$\sigma$ = volatilit\'e annualis\'ee.",
        "",
    ]

    # Risk Appetite
    if analytics_state and analytics_state.risk_appetite_matrix is not None:
        ram = analytics_state.risk_appetite_matrix
        if len(ram) > 0:
            lines += [
                r"\subsection{Risk Appetite (feux tricolores)}",
                "",
                r"Le cadre Risk Appetite (FSB 2013) classe chaque m\'etrique par cellule",
                r"en vert (acceptable), ambre (vigilance) ou rouge (breach).",
                "",
                r"\begin{table}[h]",
                r"\centering",
                r"\caption{Matrice Risk Appetite}",
                r"\begin{tabular}{" + "l" * min(len(ram.columns), 6) + r"}",
                r"\toprule",
            ]
            cols_to_show = list(ram.columns)[:6]
            header = " & ".join(f"\\textbf{{{_esc(c)}}}" for c in cols_to_show)
            lines.append(f"{header} \\\\")
            lines.append(r"\midrule")
            for _, row in ram.head(12).iterrows():
                vals = " & ".join(
                    _esc(str(round(row[c], 3)) if isinstance(row[c], float) else str(row[c]))
                    for c in cols_to_show
                )
                lines.append(f"{vals} \\\\")
            lines += [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]

    # Transition matrix
    if trans_matrix is not None and len(trans_matrix) > 0:
        lines += [
            r"\subsection{Matrice de transition des stages}",
            "",
            r"La matrice de transition capture les mouvements entre stages",
            r"(base $\to$ stress\'e) :",
            "",
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Matrice de transition (lignes = stage initial, colonnes = stage final)}",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"& \textbf{Stage 1} & \textbf{Stage 2} & \textbf{Stage 3} \\",
            r"\midrule",
        ]
        for idx in trans_matrix.index:
            vals = []
            for col in trans_matrix.columns:
                v = trans_matrix.loc[idx, col]
                vals.append(_fmt_pct(v) if isinstance(v, float) else str(v))
            line_str = f"\\textbf{{Stage {idx}}} & " + " & ".join(vals)
            lines.append(f"{line_str} \\\\")
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]

    # Recommendations
    if analytics_state and analytics_state.recommendations:
        lines += [
            r"\subsection{Recommandations CRO}",
            "",
        ]
        for i, rec in enumerate(analytics_state.recommendations, 1):
            lines += [
                f"\\textbf{{R{i}.}} {_esc(rec.action)}\\\\",
                f"R\\'egime : {_esc(rec.regime)}, "
                f"D\\'eclencheur : {_esc(rec.trigger)}, "
                f"Confiance : {_esc(rec.confidence)}.\\\\",
                f"Distance RST : {_fmt_num(rec.rst_distance)} $\\sigma$, "
                f"Risk Appetite : {_esc(rec.risk_appetite_status)}.\\\\[0.3em]",
            ]
        lines.append("")

    lines += [
        r"\interpretation{L'analyse CRO \`a 5 couches combine des m\'ethodes quantitatives",
        r"(Mahalanobis, Ornstein--Uhlenbeck, softmax) avec un cadre d\'ecisionnel",
        r"(Risk Appetite, feux tricolores) pour produire des recommandations actionables.",
        r"Les 2 passes it\'eratives permettent d'affiner le diagnostic en int\'egrant",
        r"le r\'egime d\'etect\'e.}",
        "",
    ]
    return "\n".join(lines)


def _closing() -> str:
    date_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    return "\n".join([
        r"\newpage",
        r"\section*{Annexe -- M\'etadonn\'ees}",
        "",
        r"\begin{description}",
        f"\\item[Date de g\\'en\\'eration] {date_str}",
        r"\item[Pipeline] IFRS 9 Risk Cockpit v1.0",
        r"\item[Norme] IFRS 9 (ECL), IFRS 13 (juste valeur PE)",
        r"\item[Cadre prudentiel] B\^ale III / CRR3 (janvier 2025)",
        r"\item[Corrections] 24 corrections de rigueur math\'ematique (Phase C)",
        r"\end{description}",
        "",
        r"\end{document}",
    ])


# ── Fonction principale ───────────────────────────────────


def generate_audit_latex(
    macro_params: dict,
    selected_model: str,
    result_base: pd.DataFrame,
    result_stressed: pd.DataFrame,
    result_pe: pd.DataFrame,
    advanced_metrics: pd.DataFrame,
    hhi_cross: dict,
    raroc_eva: pd.DataFrame,
    asymmetry_matrix: pd.DataFrame,
    optimization: dict,
    crr3_sensitivity: pd.DataFrame,
    analytics_state,
    trans_matrix: pd.DataFrame,
) -> str:
    """Retourne le contenu LaTeX complet en une chaine.

    Le document est organisé en 7 sections chronologiques correspondant
    aux étapes du pipeline IFRS 9, avec formules, valeurs et explications.
    """
    parts = [
        _preamble(),
        _title_page(macro_params, selected_model),
        _section_parametres(macro_params, selected_model),
        _section_staging(result_stressed),
        _section_ecl_credit(result_stressed, result_base),
        _section_pe(result_pe),
        _section_metriques_avancees(
            advanced_metrics, hhi_cross, raroc_eva, asymmetry_matrix,
        ),
        _section_optimisation(optimization, crr3_sensitivity),
        _section_ai_analyst(analytics_state, trans_matrix),
        _closing(),
    ]
    return "\n".join(parts)
