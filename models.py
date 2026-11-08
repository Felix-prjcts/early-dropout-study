"""Mettre en commun les archi. Pour l'instant on peut tester chacun dans notre notebook mais l'idéal
serait d'avoir les memes archi/hyperparams"""

import torch
from torch import nn
from torchvision.ops import StochasticDepth


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

def cnn(in_channels, output_shape, largeur=16, n_blocs=3):
    """CNN type VGG : n_blocs blocs [conv-relu-conv-relu-maxpool-dropout]
    Le pooling global à la fin permet d'utiliser la même archi quelle que soit la taille des images."""
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

class BlocRes(nn.Module):
    """Bloc résiduel : x + F(x). Le stochastic depth saute la branche F(x), jamais le raccourci."""
    def __init__(self, c_in, c_out, stride=1, sd_rate=0.0):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out),
        )
        # raccourci : identité, ou conv 1x1 si la taille/le nombre de canaux change
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))
        else:
            self.shortcut = nn.Identity()
        self.sd = StochasticDepth(sd_rate, mode="row")  # saute la branche par échantillon
        self.sd.base_p = sd_rate                         # taux de base, pour pouvoir le moduler ensuite

    def forward(self, x):
        return torch.relu(self.shortcut(x) + self.sd(self.f(x)))


def resnet(output_shape=200, widths=(32, 64, 128, 256), blocs=(1, 1, 1, 1), sd_max=0.0, p_head=0.0, in_channels=3):
    """Petit ResNet pour images 64x64 (Tiny ImageNet). ~1.2M paramètres avec les valeurs par défaut.
    sd_max : taux de stochastic depth du dernier bloc (croît linéairement avec la profondeur).
    p_head : dropout avant le classifieur final."""
    n_total = sum(blocs)
    rates = [sd_max * i / max(n_total - 1, 1) for i in range(n_total)]

    layers = [nn.Conv2d(in_channels, widths[0], 3, 1, 1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU()]
    c_in, k = widths[0], 0
    for i, (c, n) in enumerate(zip(widths, blocs)):
        for j in range(n):
            stride = 2 if (j == 0 and i > 0) else 1   # on réduit la résolution au début de chaque stage (sauf le 1er)
            layers.append(BlocRes(c_in, c, stride, rates[k]))
            c_in, k = c, k + 1

    model = nn.Sequential(
        *layers,
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Dropout(p_head),
        nn.Linear(widths[-1], output_shape),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, opt
