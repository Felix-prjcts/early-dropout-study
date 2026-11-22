# L’early dropout réduit-il le sous-apprentissage à petite échelle ?

**Étude expérimentale — Deep Learning, M2 ISF, Université Paris Dauphine–PSL**  
**Auteurs :** Nibir Hossain, Félix Bouvier, Oscar Hedges et Lyes Bouchoucha

## Présentation

Le **dropout** est couramment utilisé pour limiter le surapprentissage, mais l'appliquer pendant tout l'entraînement peut dégrader l'ajustement. Dans [*Dropout Reduces Underfitting*](https://proceedings.mlr.press/v202/liu23aq.html) (ICML 2023), Liu et al. proposent l'**early dropout** : appliquer le dropout seulement au début de l'entraînement, puis le désactiver. Sur ImageNet, cette stratégie abaisse la perte d'entraînement et améliore l'exactitude de modèles qui sous-apprennent.

Nous examinons si ces observations se retrouvent sur des modèles 100 à 1 000 fois plus petits, sur **CIFAR-10**, **Fashion-MNIST** et **California Housing**, en séparant la qualité de l'ajustement (perte d'entraînement) des performances de généralisation (validation et test).

## Questions étudiées

1. **L'early dropout marche-t-il à notre échelle, et sur quels modèles ?** Améliore-t-il l'ajustement, et ce gain se traduit-il en généralisation ?
2. **Combien de temps faut-il garder le dropout ?** Que veut dire « early » (10 %, 30 %, 50 % de l'entraînement ?), et cette durée se compte-t-elle en époques quand la taille de batch varie ?

Deux analyses complètent ces questions : la pénalité sous **dropout permanent** (le critère de sous-apprentissage de l'article) prédit-elle l'intérêt de l'early dropout ? Et, sur CIFAR-10, les mesures de direction des gradients confirment-elles l'explication proposée par Liu et al. ?

## Protocole expérimental

Pour chaque configuration, nous comparons trois stratégies :

- **Sans dropout** : modèle de référence ;
- **Dropout permanent** : régularisation active pendant tout l'entraînement ;
- **Early dropout** : régularisation active pendant les `k` premières époques, puis désactivée.

Le taux principal est **`p = 0,1`**, constant pendant la fenêtre. Dans le ResNet de CIFAR-10, nous utilisons la **stochastic depth**, qui désactive des branches résiduelles entières.

| Jeu de données | Modèles principaux | Budget principal | Fenêtres initiales étudiées | Graines | Métriques |
|---|---|---|---|---|---|
| **CIFAR-10** | MLP, CNN et ResNet | 10 époques (MLP/CNN), 20 (ResNet) | 10, 30, 50 % ; fenêtre tardive de 30 % | 3 en sélection, 5 nouvelles ensuite | Entropie croisée d'entraînement, exactitude validation/test |
| **Fashion-MNIST** | MLP (largeur 16) et CNN (largeur 4) | 50 époques | 1, 2, 5, 10, 15, 25 époques | 5 (10 pour le CNN), 5 nouvelles pour confirmer le régime | Entropie croisée, exactitude validation/test |
| **California Housing** | MLP à deux couches cachées (largeur 32) | 50 époques | 2, 5, 10, 15, 25 époques | 10 en calibration, dont 5 pour les fenêtres | MSE d'entraînement, R² validation/test |

Les configurations principales utilisent des **batchs de 128**. Une étude complémentaire fait varier la taille de batch : 32 et 512 sur Fashion-MNIST (plus 200 époques au batch 512, pour comparer à nombre de mises à jour égal), 8 à 512 sur California Housing.

Les stratégies sont comparées de façon **appariée** (mêmes graines, même découpage des données). Les écarts au modèle sans dropout sont résumés par leur moyenne et un **intervalle de confiance à 95 %** (loi de Student sur les différences entre graines). La **correction de Holm** s'applique aux deux familles de tests fixées à l'avance sur la perte d'entraînement : les trois architectures de CIFAR-10 (fenêtre de 30 %) et les douze comparaisons de la grille Fashion-MNIST. Les autres comparaisons (test, ablations, taille de batch) sont **exploratoires**.

## Principaux résultats

**Ajustement et généralisation**

- **CIFAR-10, ResNet :** six époques de stochastic depth sur vingt abaissent l'entropie croisée d'entraînement de **0,066** [−0,102 ; −0,029] (Holm p = 0,015) et portent l'exactitude de test de 77,88 % à 78,90 %, soit **+1,03 point** [+0,20 ; +1,86] (comparaison exploratoire). Le gain apparaît après la coupure du masque.
- **CIFAR-10, MLP et CNN :** l'early dropout augmente la perte d'entraînement du MLP (+0,031, Holm p = 0,003) et ne change rien pour le CNN.
- **Fashion-MNIST :** sur le MLP, deux ou cinq époques de dropout sur cinquante abaissent la perte d'entraînement (−0,013 et −0,011, résistent à Holm), sans gain d'exactitude établi. Sur un budget de 20 époques, le gain de test atteint **+0,37 point** [+0,06 ; +0,68]. Sur un MLP qui surapprend (largeur 64), la perte d'entraînement baisse quand même (−0,008) : l'effet porte d'abord sur l'optimisation.
- **California Housing :** le dropout permanent pénalise le modèle (−1,40 point de R² de validation), mais aucune fenêtre initiale ne l'améliore ; les fenêtres longues le dégradent (−0,66 point à 25 époques).

**Durée de la fenêtre**

- La fenêtre utile est **courte** : 2 à 5 époques sur 50 pour le MLP de Fashion-MNIST (25 époques deviennent défavorables), 10 à 30 % pour le ResNet. Prolonger à 50 % coûte 0,32 point de validation au ResNet par rapport à 30 %.
- Elle **dépend du modèle et de la dose** : le CNN de Fashion-MNIST réagit plus tard (baisse la plus nette à 10 époques), et un taux de 0,3 raccourcit la fenêtre.
- Le **moment** compte : six époques au début font mieux que six époques à la fin (+1,22 point de validation pour le ResNet).
- La durée **ne se compte pas en époques** : sur Fashion-MNIST, au batch 512, quinze époques restent favorables (−0,012) alors qu'elles sont neutres au batch 128, et aucune fenêtre n'aide au batch 32. Le batch 512 fait apparaître un gain d'exactitude (CNN : +1,46 point de validation, +1,33 de test), mais ce gain **disparaît à nombre de mises à jour égal** (MLP au batch 512 sur 200 époques : au mieux +0,11 point de test). Sur California Housing, les grands batchs rendent au contraire le dropout coûteux plus tôt.

**Mécanisme**

- Les mesures de *gradient direction variance* (GDV) et de *gradient direction error* (GDE) sur CIFAR-10 ne montrent pas la réduction attendue (tous les intervalles contiennent zéro), y compris pour le ResNet : elles ne confirment pas le mécanisme proposé, sans le réfuter.

**Enseignement principal :** l'early dropout fait surtout prendre de l'avance à l'optimisation. Cette avance ne devient un gain d'exactitude que si l'entraînement s'arrête avant la convergence (ResNet sur 20 époques, CNN de Fashion-MNIST au batch 512), et la pénalité sous dropout permanent ne suffit pas à prédire où l'effet apparaît.

## Contenu du dépôt

| Chemin | Contenu |
|---|---|
| `cifar10.ipynb` | Sélection des configurations, comparaisons MLP/CNN/ResNet, durée et position des fenêtres, analyse des gradients |
| `fashion_mnist.ipynb` | Calibration et confirmation du régime, grille des durées, budget, contrôle en surapprentissage, taux et position du dropout, taille de batch (32, 128, 512 ; 200 époques au batch 512) |
| `california_housing.ipynb` | Calibration, fenêtres initiales en régression, taille de batch (8 à 512) |
| `resultats/` | Un CSV par entraînement pour Fashion-MNIST et California Housing (sous-dossiers `batch32/`, `batch512/` et `p03/` pour les études complémentaires) |
| `resultats_cifar10/` | Résultats de l'étude CIFAR-10 (traces JSON, tableaux CSV) |
| `early_dropout.pdf` | Rapport final |

Les poids des modèles (`*.pt`) ne sont pas versionnés, à cause de leur taille. Les jeux de données sont téléchargés automatiquement dans `data/` au premier lancement (Fashion-MNIST, CIFAR-10) ou par scikit-learn (California Housing) ; ce dossier n'est pas versionné.

## Lire et reproduire les expériences

Chaque notebook est autonome : toutes les fonctions (modèles, entraînement, statistiques) y sont définies, sans script externe. Dépendances : `torch`, `torchvision`, `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn` (California Housing) et `tqdm` (CIFAR-10).

```bash
pip install torch torchvision numpy pandas scipy matplotlib scikit-learn tqdm
jupyter lab
```

- **Fashion-MNIST et California Housing :** chaque entraînement est enregistré dans un CSV de `resultats/`, et les entraînements déjà présents sont sautés. Exécuter le notebook du début à la fin relit donc les résultats en quelques secondes et reproduit tous les tableaux et figures. Supprimer un CSV relance l'entraînement correspondant.
- **CIFAR-10 :** le notebook lit ses résultats dans `resultats_cifar10/`. Sans ces fichiers, il réentraîne les modèles, ce qui demande un GPU.

### Lire les résultats

- Un écart de **perte** négatif est favorable ; un écart d'**exactitude** ou de **R²** positif est favorable.
- Les intervalles de confiance portent sur les **différences appariées entre graines** : ils ne mesurent pas l'incertitude liée à d'autres découpages des données.
- Une baisse de la perte d'entraînement ne garantit pas une meilleure généralisation.
- Sur CIFAR-10, l'architecture, le mécanisme de régularisation et le budget changent ensemble entre le ResNet et les autres modèles.

## Utilisation de l'intelligence artificielle

Des outils d'IA générative ont servi d'assistance pour écrire et nettoyer le code des notebooks, déboguer, vérifier des calculs statistiques, produire des figures et rédiger ou mettre en forme certaines parties du rapport. Les auteurs ont défini la problématique, choisi et validé les expériences, vérifié les résultats et en assument l'interprétation.

## Références

- Liu, Z., Xu, Z., Jin, J., Shen, Z. et Darrell, T. (2023). [*Dropout Reduces Underfitting*](https://proceedings.mlr.press/v202/liu23aq.html). *Proceedings of the 40th International Conference on Machine Learning*, PMLR 202, 22233–22248.
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. et Salakhutdinov, R. (2014). [*Dropout: A Simple Way to Prevent Neural Networks from Overfitting*](https://jmlr.org/papers/v15/srivastava14a.html). *Journal of Machine Learning Research*, 15, 1929–1958.
- Huang, G., Sun, Y., Liu, Z., Sedra, D. et Weinberger, K. Q. (2016). [*Deep Networks with Stochastic Depth*](https://doi.org/10.1007/978-3-319-46493-0_39). *ECCV 2016*, 646–661.
- Holm, S. (1979). [*A Simple Sequentially Rejective Multiple Test Procedure*](https://doi.org/10.2307/4615733). *Scandinavian Journal of Statistics*, 6(2), 65–70.
