"""Package d'entrainement offline des modeles PD.

Usage :
    python -m ifrs9_cockpit.training.train                         # synthetique 1M
    python -m ifrs9_cockpit.training.train --data portefeuille.csv # CSV utilisateur
    python -m ifrs9_cockpit.training.train --n-clients 500000      # taille custom
"""
