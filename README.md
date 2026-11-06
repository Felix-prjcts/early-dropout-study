# Projet DL - early dropout, V1 test

On bosse sur le papier "Dropout Reduces Underfitting" (Liu et al. 2023). En gros ils disent que si on met du dropout seulement au début de l'entraînement puis qu'on l'enlève (early dropout), ça aide les modèles qui sous-apprennent. Ils le montrent que sur ImageNet avec des gros modèles, et ils expliquent pas trop ce que "début" veut dire (50 epochs par défaut, "ça marche de 5 à 300").

Nous on veut voir si ça tient sur des modèles du cours (MLP, CNN, ResNet sur Fashion-MNIST / CIFAR-10), avec le même protocole partout, et surtout combien de temps il faut garder le dropout.

## Protocole actuel (sert comme base pour commencer les test mais surement pas opti)

- Optimiseur : Adam, lr = 1e-3,
- Schedule : cosinus jusqu'à 0 sur tout l'entraînement. Pas de ReduceLROnPlateau parce que sinon le lr change pas au même moment selon qu'il y a du dropout ou pas, et on saurait plus ce qui joue
- Batch : 128
- Dropout : p = 0.1 (proba de couper un neurone), mis après chaque couche cachée
- Early dropout : dropout à 0.1 (a changer éventuellement) pendant les T premières epochs, puis 0. T est une fraction du nb d'epochs : `EARLY_GRID = [0, 0.1, 0.2, 0.3, 0.5, 1]`, avec 0 = pas de dropout et 1 = dropout classique tout le long
- Données : on garde 10 000 images du train pour la validation (split fixe, seed 0), normalisation calculée sur le train seulement
- Seeds : 3 (5 dans la version fashionmnist mais ca risque d'etre trop). Pour une même seed on a la même init des poids, donc on compare early vs sans dropout seed par seed
- Ce qu'on mesure à chaque epoch, en mode eval (sans dropout) : loss train, loss val, accuracy val, accuracy test
- Valeur finale d'un run = moyenne des 5 dernières epochs (ça lisse le bruit)
- On décide tout sur la validation, le test c'est juste pour le chiffre final

Ce qui change d'un cadre à l'autre : seulement la taille du modèle et le nb d'epochs, trouvés à l'étape 1.

## La démarche

1. Calibration : trouver un régime de sous-apprentissage  
On fait une grille largeur × nb d'epochs, en comparant juste sans dropout et dropout classique. On regarde delta = acc val (dropout classique) - acc val (sans dropout) :
- delta > 0 : le dropout aide, donc le modèle surapprend => pas bon pour nous
- delta < 0 : le dropout fait perdre, donc le modèle sous-apprend => c'est ce qu'on veut


2. Expérience principale : est-ce que l'early dropout aide et quelle durée ?  
Dans le réglage trouvé, on lance toute la grille `EARLY_GRID` sur les seeds. Pour chaque durée on calcule le gain par rapport à sans dropout (seed par seed), On regarde :
- est-ce que la loss d'entraînement finale baisse (le "fit mieux" du papier)  
- est-ce que l'accuracy val/test monte (la généralisation)
- quelle durée marche le mieux, et si c'est large comme dit le papier ou pas

Choses a tester en plus :   
Si le meilleur point est à la plus petite durée testée, on rajoute des durées plus courtes (1-2 epochs) pour voir si l'optimum est encore en dessous.

Epochs ou itérations :
On refait l'expérience 2 avec d'autres tailles de batch (32 et 512). Si la bonne durée reste la même en epochs, c'est la quantité de données vue qui compte. Si elle reste la même en nb d'itérations, c'est le nb de pas de gradient. Pour l'instant prévu que sur le MLP (c'est le moins cher).


## Les fichiers

- `config.py` : constantes communes (batch, p, grille des durées, seeds)
- `models.py` : les archis + l'optimiseur, comme ça tout le monde a les mêmes
- `mlp_fmnist.ipynb` : cadre MLP / Fashion-MNIST (calibration, expériences, graphes)
- `resultats/` : un csv par run, nommé `cadre_wLARGEUR_eEPOCHS_fracFRAC_sSEED.csv`. Si on relance, les runs déjà faits sont sautés
