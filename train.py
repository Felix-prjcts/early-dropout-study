import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import StochasticDepth


def entrainer(model, opt, data, epochs, t_early, p=0.1, batch=128, lineaire=False):
    """Entraîne le modèle avec early dropout : dropout p pendant les t_early premières epochs, puis 0.
    Pareil pour stochasticDepth avec un taux fixé à la création du modèle. 
    t_early = 0 : sans dropout, t_early = epochs : dropout classique.
    lineaire=True : le taux descend de p (ou sd_max pour sd) à 0 pendant les t_early epochs au lieu de rester constant
    """
    prep = data.get("prep", lambda x: x)
    Xtr, ytr = data["Xtr"], data["ytr"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rows = []

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
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            xb, yb = prep(Xtr[idx]), ytr[idx]
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        # évaluation sans dropout, par paquets
        tr_loss, tr_acc = evaluer(Xtr, ytr)
        va_loss, va_acc = evaluer(data["Xva"], data["yva"])
        row = {"epoch": epoch, "rate": rate, "scale": scale, "p": p, "t_early": t_early, "batch": batch,
               "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc}
        if "Xte" in data:
            row["test_acc"] = evaluer(data["Xte"], data["yte"])[1]
        rows.append(row)
        
    return pd.DataFrame(rows)
