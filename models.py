"""Les architectures communes. Le dropout est créé à 0 : train.py règle son taux à chaque epoch."""

import torch
from torch import nn


def mlp(input_shape, output_shape, couche_cache=64):
    """MLP à 2 couches cachées de couche_cache neurones, un dropout après chacune."""
    model = nn.Sequential(
        nn.Linear(input_shape, couche_cache), nn.ReLU(), nn.Dropout(0.0),
        nn.Linear(couche_cache, couche_cache), nn.ReLU(), nn.Dropout(0.0),
        nn.Linear(couche_cache, output_shape),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, opt


def cnn(in_channels, output_shape, largeur=16, n_blocs=3):
    """CNN type VGG : n_blocs blocs [conv-relu-conv-relu-maxpool-dropout], le nombre de canaux double à chaque bloc.
    Le pooling global à la fin permet d'utiliser le même modèle quelle que soit la taille des images."""
    couches, c_in = [], in_channels
    for k in range(n_blocs):
        c_out = largeur * 2 ** k
        couches += [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv2d(c_out, c_out, kernel_size=3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2), nn.Dropout(0.0)]
        c_in = c_out
    couches += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c_in, output_shape)]
    model = nn.Sequential(*couches)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, opt
