"""Point d'entree standalone : python -m ifrs9_cockpit.config."""
from ifrs9_cockpit.config import (
    RANDOM_SEED, N_CLIENTS, N_MONTHS,
    SECTORS, ECL_SCENARIOS, PREDEFINED_SCENARIOS,
    BASEL_CONFIG, RISK_APPETITE_CONFIG,
    PE_CLASSIFICATION_CONFIG, DASHBOARD_CONFIG,
    REQUIRED_CREDIT_COLS, REQUIRED_PE_COLS,
    REQUIRED_CREDIT_RESULT_COLS, REQUIRED_PE_RESULT_COLS,
    MACRO_INCOHERENCE_RULES,
)


def main() -> None:
    print("=" * 60)
    print("IFRS 9 Risk Cockpit — Configuration")
    print("=" * 60)

    print(f"\nSeed : {RANDOM_SEED}")
    print(f"Clients : {N_CLIENTS:,}")
    print(f"Mois historique : {N_MONTHS}")

    print(f"\n--- {len(SECTORS)} Secteurs ---")
    for s in SECTORS:
        print(
            f"  {s.name:15s} | prop={s.proportion:.0%} | PD base={s.base_default_rate:.0%} "
            f"| IPEV={s.valuation_method:15s} | multiple={s.exit_multiple_base:.1f}x"
        )
    total = sum(s.proportion for s in SECTORS)
    print(f"  Total proportions : {total:.4f}")

    print(f"\n--- {len(ECL_SCENARIOS)} Scenarios ECL ---")
    for sc in ECL_SCENARIOS:
        print(f"  {sc.name:12s} | poids={sc.weight:.0%}")
    total_w = sum(sc.weight for sc in ECL_SCENARIOS)
    print(f"  Total poids : {total_w:.4f}")

    print(f"\n--- {len(PREDEFINED_SCENARIOS)} Scenarios predefinis (dropdown) ---")
    for name, params in PREDEFINED_SCENARIOS.items():
        print(f"  {name:20s} | {params}")

    print(f"\n--- Basel III / CRR3 ---")
    print(f"  CET1 cible : {BASEL_CONFIG.cet1_target:.1%}")
    print(f"  RW PE : {BASEL_CONFIG.rw_pe_options} (defaut={BASEL_CONFIG.rw_pe_default}%)")
    print(f"  PE max allocation : {BASEL_CONFIG.pe_max_allocation:.0%}")
    print(f"  HHI max : {BASEL_CONFIG.hhi_max}")

    print(f"\n--- Risk Appetite ---")
    print(f"  ECL/EAD : vert < {RISK_APPETITE_CONFIG.ecl_ead_green:.1%} "
          f"| ambre < {RISK_APPETITE_CONFIG.ecl_ead_amber:.1%} | rouge")
    print(f"  RAROC   : vert > {RISK_APPETITE_CONFIG.raroc_green:.1%} "
          f"| ambre > {RISK_APPETITE_CONFIG.raroc_amber:.1%} | rouge")
    print(f"  HHI     : vert < {RISK_APPETITE_CONFIG.hhi_green} "
          f"| ambre < {RISK_APPETITE_CONFIG.hhi_amber} | rouge")

    print(f"\n--- Contrats DataFrame ---")
    print(f"  Credit cols : {len(REQUIRED_CREDIT_COLS)} colonnes")
    print(f"  PE cols     : {len(REQUIRED_PE_COLS)} colonnes")
    print(f"  Credit result : {len(REQUIRED_CREDIT_RESULT_COLS)} colonnes")
    print(f"  PE result     : {len(REQUIRED_PE_RESULT_COLS)} colonnes")

    print(f"\n--- Regles d'incoherence macro : {len(MACRO_INCOHERENCE_RULES)} ---")
    for rule in MACRO_INCOHERENCE_RULES:
        print(f"  {rule.name:25s} | {rule.description}")

    print(f"\n--- Classification PE ---")
    print(f"  Performing : P(distress) < {PE_CLASSIFICATION_CONFIG.distress_threshold_performing:.0%}")
    print(f"  Watchlist  : P(distress) < {PE_CLASSIFICATION_CONFIG.distress_threshold_watchlist:.0%}")
    print(f"  Distressed : P(distress) >= {PE_CLASSIFICATION_CONFIG.distress_threshold_watchlist:.0%}")
    print(f"  Secondary discount : {PE_CLASSIFICATION_CONFIG.secondary_discount:.0%}")

    print(f"\n--- Palette ---")
    print(f"  Primary  : {DASHBOARD_CONFIG.theme_primary}")
    print(f"  Bg dark  : {DASHBOARD_CONFIG.theme_bg_dark}")

    print("\nConfiguration valide.")


if __name__ == "__main__":
    main()
