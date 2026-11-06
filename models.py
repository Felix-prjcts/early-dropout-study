"""Mettre en commun les archi. Pour l'instant on peut tester chacun dans notre notebook mais l'idéal
serait d'avoir les memes archi/hyperparams"""

import torch
from torch import nn

def mlp(input_shape, output_shape, couche_cache = 64):
    """Pour l'instant juste l'archi et l'opti,
    plus tard éventuellement la loss ou tout le training"""
    model = nn.Sequential(
        nn.Linear(input_shape, couche_cache), nn.ReLU(), nn.Dropout(0.0),
        nn.Linear(couche_cache, couche_cache), nn.ReLU(), nn.Dropout(0.0),
        nn.Linear(couche_cache, output_shape),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, opt


        