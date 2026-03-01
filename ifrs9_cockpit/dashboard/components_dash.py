"""Backward compatibility -- re-exports from components package.

All component builders have been moved to ``ifrs9_cockpit.dashboard.components``
sub-modules (cards, narrative, sidebar, arbitrage, helpers).

This shim ensures that existing imports like::

    from ifrs9_cockpit.dashboard.components_dash import build_header

continue to work without modification.
"""

from ifrs9_cockpit.dashboard.components import *  # noqa: F401,F403
