"""constantes pour mutualisé les params des runs"""

BATCH = 128
P=0.1 #taux de dropout, semble etre le taux "de base" dans le papier mais a adapté éventuellement
EARLY_GRID = [0, 0.1, 0.2, 0.3, 0.5, 1]    # durée de l'early en pct du nombre d'epochs arbitraire pour l'instant
# T_EARLY sera donc = EPOCHS * early_rate for early_rate in EARLY_GRID
SEEDS = [0, 1, 2]