"""Configuration du logging structure pour le projet IFRS 9.

Fournit un logger structure (structlog) pour remplacer les print()
disperses dans le projet. Chaque evenement de log est enrichi
automatiquement du timestamp, module, et contexte metier.

Usage :
    from ifrs9_cockpit.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("ecl_calculated", ecl_total=1_234_567.89, n_clients=30_000)
"""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Configure structlog pour le projet.

    Args:
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR).
        json_output: Si True, sortie JSON (pour production). Sinon, console lisible.
    """
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(
            colors=sys.stderr.isatty(),
        )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Reduire le bruit des librairies tierces
    for lib in ("urllib3", "matplotlib", "PIL", "torch", "xgboost"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Retourne un logger structure pour le module donne.

    Args:
        name: Nom du module (typiquement __name__).

    Returns:
        Logger structure avec contexte enrichi.
    """
    return structlog.get_logger(name)


# Auto-setup au premier import (mode console dev)
setup_logging(level="INFO", json_output=False)
