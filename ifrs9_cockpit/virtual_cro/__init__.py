"""Virtual CRO — Systeme Multi-Agents Neuro-Symbolique (NeSy MAS).

Architecture en 4 couches :
    1. Agents neuronaux (MLP) : Macro, Quant, PE, Contrarian
    2. Fusion Dempster-Shafer : Jousselme + bBPA + regime-switching
    3. QBAF NeSy : arguments symboliques IFRS 9 + arguments neuronaux
    4. PMA Engine : Post-Model Adjustment automatise (ECL_legal - ECL_committee)

References :
    - Jousselme et al. (2001) : distance entre masses de croyance
    - Dung (1995), Baroni et al. (2019) : QBAF argumentation
    - Hurlin et al. (2026) : reverse stress testing
    - MASCA (ACL 2025) : multi-agent credit assessment
    - PMADS (arXiv 2510.17108) : post-model adjustments
"""

from ifrs9_cockpit.virtual_cro.engine import VirtualCROEngine, VirtualCROResult
from ifrs9_cockpit.virtual_cro.agents import MacroAgent, QuantAgent, PEAgent, ContrarianAgent
from ifrs9_cockpit.virtual_cro.fusion import DSFusion
from ifrs9_cockpit.virtual_cro.qbaf import NeSyQBAF
from ifrs9_cockpit.virtual_cro.pma import PMAEngine

__all__ = [
    "VirtualCROEngine",
    "VirtualCROResult",
    "MacroAgent",
    "QuantAgent",
    "PEAgent",
    "ContrarianAgent",
    "DSFusion",
    "NeSyQBAF",
    "PMAEngine",
]
