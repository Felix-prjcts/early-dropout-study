"""constantes pour mutualisé les params des runs"""

BATCH = 128
P = 0.1   # taux de dropout, semble etre le taux "de base" dans le papier
EARLY_GRID = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1]   # durée de l'early en fraction du nb d'epochs
# 0, 0.05, 0.1, 0.3, 1 = grille obligatoire du protocole commun ; 0.2 et 0.5 en plus
# T_EARLY = round(EPOCHS * frac)  (attention : round(2.5) = 2 en python, donc 0.05 sur 50 epochs = 2 epochs)
SEEDS = [0, 1, 2, 3, 4]
