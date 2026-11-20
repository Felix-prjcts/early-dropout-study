import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import StochasticDepth


def entrainer(model, opt, data, epochs, t_early, p=0.1, batch=128, lineaire=False, n_eval=10000):
    """Entraîne le modèle avec early dropout : dropout p pendant les t_early premières epochs, puis 0.
    Pareil pour stochasticDepth avec un taux fixé à la création du modèle. 
    t_early = 0 : sans dropout, t_early = epochs : dropout classique.
    lineaire=True : le taux descend de p (ou sd_max pour sd) à 0 pendant les t_early epochs au lieu de rester constant
    """
    prep = data.get("prep", lambda x: x)
    Xtr, ytr = data["Xtr"], data["ytr"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rows = []

    # Pour calculer la loss/accuracy sur un sous-ensemble de l'entraînement, plus rapide.
    g = torch.Generator().manual_seed(1234)
    sub = torch.randperm(len(Xtr), generator=g)[:min(n_eval, len(Xtr))].to(Xtr.device)
    Xsub, ysub = Xtr[sub], ytr[sub]

    # évaluation par paquets pour ne pas saturer la RAM
    @torch.no_grad()
    def evaluer(X, y, bs=1000):
        model.eval()
        loss, correct = 0.0, 0
        for i in range(0, len(X), bs):
            out = model(prep(X[i:i + bs]))
            yb = y[i:i + bs]
            loss += F.cross_entropy(out, yb, reduction="sum").item()
            correct += (out.argmax(1) == yb).sum().item()
        return loss / len(X), correct / len(X)

    for epoch in range(epochs):
        print(f"\rEpoch {epoch + 1}/{epochs}",end="", flush=True)
        # facteur d'intensité de cette epoch : 1 -> 0
        if epoch >= t_early:
            scale = 0.0
        elif lineaire:
            scale = 1 - epoch / t_early
        else:
            scale = 1.0
        rate = p * scale

        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = rate
            elif isinstance(m, StochasticDepth):
                m.p = m.base_p * scale

        # une epoch d'entraînement
        model.train()
        perm = torch.randperm(len(Xtr), device=Xtr.device)
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            xb, yb = prep(Xtr[idx]), ytr[idx]
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        # évaluation sans dropout, par paquets
        tr_loss, tr_acc = evaluer(Xsub, ysub)
        va_loss, va_acc = evaluer(data["Xva"], data["yva"])
        row = {"epoch": epoch, "rate": rate, "scale": scale, "p": p, "t_early": t_early, "batch": batch,
               "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc}
        if "Xte" in data:
            row["test_acc"] = evaluer(data["Xte"], data["yte"])[1]
        rows.append(row)

    print()
    return pd.DataFrame(rows)

def entrainer_regression(model, opt, data, epochs, t_early, p=0.1, batch=128, lineaire=False, n_eval=10000):
    """Même principe que entrainer (early dropout / stochastic depth), pour une tâche de régression.
    Loss : MSE. Métrique : R² (1 = parfait, 0 = pas mieux que prédire la moyenne, peut être négatif).
    Le modèle doit sortir 1 valeur par exemple (output_shape=1), et ytr / yva (/ yte) sont des valeurs
    continues, idéalement standardisées avec les stats du train.
    """
    prep = data.get("prep", lambda x: x)
    Xtr, ytr = data["Xtr"], data["ytr"].float().view(-1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rows = []

    # sous-ensemble fixe du train pour l'évaluation
    g = torch.Generator().manual_seed(1234)
    sub = torch.randperm(len(Xtr), generator=g)[:min(n_eval, len(Xtr))].to(Xtr.device)
    Xsub, ysub = Xtr[sub], ytr[sub]

    def sortie(xb):
        """Sortie du modèle au format (B,), pour correspondre aux cibles."""
        return model(xb).view(-1)

    @torch.no_grad()
    def evaluer(X, y, bs=1000):
        """Renvoie (MSE, R²), calculés par paquets."""
        model.eval()
        y = y.float().view(-1)
        sse, s_y, s_y2 = 0.0, 0.0, 0.0
        for i in range(0, len(X), bs):
            out, yb = sortie(prep(X[i:i + bs])), y[i:i + bs]
            sse += F.mse_loss(out, yb, reduction="sum").item()
            s_y += yb.sum().item()
            s_y2 += (yb ** 2).sum().item()
        n = len(X)
        sst = s_y2 - s_y ** 2 / n          # somme des carrés totale (variance des cibles × n)
        return sse / n, 1 - sse / sst

    for epoch in range(epochs):
        # facteur d'intensité de cette epoch : 1 -> 0
        if epoch >= t_early:
            scale = 0.0
        elif lineaire:
            scale = 1 - epoch / t_early
        else:
            scale = 1.0
        rate = p * scale

        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = rate
            elif isinstance(m, StochasticDepth):
                m.p = m.base_p * scale

        # une epoch d'entraînement
        model.train()
        perm = torch.randperm(len(Xtr), device=Xtr.device)
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            loss = F.mse_loss(sortie(prep(Xtr[idx])), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        # évaluation sans dropout
        tr_loss, tr_r2 = evaluer(Xsub, ysub)
        va_loss, va_r2 = evaluer(data["Xva"], data["yva"])
        row = {"epoch": epoch, "rate": rate, "scale": scale, "p": p, "t_early": t_early, "batch": batch,
               "train_loss": tr_loss, "train_r2": tr_r2, "val_loss": va_loss, "val_r2": va_r2}
        if "Xte" in data:
            row["test_r2"] = evaluer(data["Xte"], data["yte"])[1]
        rows.append(row)
    return pd.DataFrame(rows)