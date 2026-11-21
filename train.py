"""Entraînement avec early dropout : dropout de taux p pendant les t_early premières epochs, puis 0.
t_early = 0 donne un run sans dropout, t_early = epochs un run avec dropout classique.
Le learning rate suit un cosinus jusqu'à 0 sur tout l'entraînement."""

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn


def regler_dropout(model, taux):
    for couche in model.modules():
        if isinstance(couche, nn.Dropout):
            couche.p = taux


def une_epoch(model, opt, X, y, perte, batch):
    model.train()
    ordre = torch.randperm(len(X), device=X.device)
    for i in range(0, len(X), batch):
        idx = ordre[i:i + batch]
        loss = perte(model(X[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()


def sous_ensemble(X, y, n):
    """Toujours les mêmes n exemples du train, pour mesurer la loss d'entraînement sans tout évaluer."""
    idx = torch.randperm(len(X), generator=torch.Generator().manual_seed(1234))[:n].to(X.device)
    return X[idx], y[idx]


def entrainer(model, opt, data, epochs, t_early, p=0.1, batch=128, n_eval=10000):
    """Classification. Renvoie un DataFrame avec une ligne par epoch (mesures faites sans dropout)."""
    Xtr, ytr = data["Xtr"], data["ytr"]
    Xsub, ysub = sous_ensemble(Xtr, ytr, n_eval)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    @torch.no_grad()
    def evaluer(X, y, bs=1000):
        # loss et accuracy sans dropout, par paquets de 1000 pour ne pas saturer la mémoire
        model.eval()
        loss, bons = 0.0, 0
        for i in range(0, len(X), bs):
            sortie = model(X[i:i + bs])
            loss += F.cross_entropy(sortie, y[i:i + bs], reduction="sum").item()
            bons += (sortie.argmax(1) == y[i:i + bs]).sum().item()
        return loss / len(X), bons / len(X)

    lignes = []
    for epoch in range(epochs):
        print(f"\rEpoch {epoch + 1}/{epochs}", end="", flush=True)
        taux = p if epoch < t_early else 0.0
        regler_dropout(model, taux)
        une_epoch(model, opt, Xtr, ytr, F.cross_entropy, batch)
        sched.step()

        train_loss, train_acc = evaluer(Xsub, ysub)
        val_loss, val_acc = evaluer(data["Xva"], data["yva"])
        ligne = {"epoch": epoch, "rate": taux, "p": p, "t_early": t_early, "batch": batch,
                 "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}
        if "Xte" in data:
            ligne["test_acc"] = evaluer(data["Xte"], data["yte"])[1]
        lignes.append(ligne)
    print()
    return pd.DataFrame(lignes)


def entrainer_regression(model, opt, data, epochs, t_early, p=0.1, batch=128, n_eval=10000):
    """Régression (le modèle sort une valeur par exemple). Loss : MSE. Métrique : R²
    (1 = parfait, 0 = pas mieux que prédire la moyenne)."""
    Xtr, ytr = data["Xtr"], data["ytr"].float().view(-1)
    Xsub, ysub = sous_ensemble(Xtr, ytr, n_eval)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def perte(sortie, y):
        return F.mse_loss(sortie.view(-1), y)

    @torch.no_grad()
    def evaluer(X, y, bs=1000):
        # MSE et R² sans dropout, par paquets
        model.eval()
        y = y.float().view(-1)
        pred = torch.cat([model(X[i:i + bs]).view(-1) for i in range(0, len(X), bs)])
        mse = F.mse_loss(pred, y).item()
        return mse, 1 - mse / y.var(unbiased=False).item()

    lignes = []
    for epoch in range(epochs):
        taux = p if epoch < t_early else 0.0
        regler_dropout(model, taux)
        une_epoch(model, opt, Xtr, ytr, perte, batch)
        sched.step()

        train_loss, train_r2 = evaluer(Xsub, ysub)
        val_loss, val_r2 = evaluer(data["Xva"], data["yva"])
        ligne = {"epoch": epoch, "rate": taux, "p": p, "t_early": t_early, "batch": batch,
                 "train_loss": train_loss, "train_r2": train_r2, "val_loss": val_loss, "val_r2": val_r2}
        if "Xte" in data:
            ligne["test_r2"] = evaluer(data["Xte"], data["yte"])[1]
        lignes.append(ligne)
    return pd.DataFrame(lignes)
