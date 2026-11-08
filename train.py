import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn


def entrainer(model, opt, data, epochs, t_early, p=0.1, batch=128, lineaire=False):
    """Entraîne le modèle avec early dropout : dropout p pendant les t_early premières epochs, puis 0.
    t_early = 0 : sans dropout, t_early = epochs : dropout classique.
    lineaire=True : le taux descend de p à 0 pendant les t_early epochs au lieu de rester constant
    """
    Xtr, ytr = data["Xtr"], data["ytr"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rows = []

    for epoch in range(epochs):
        # taux de dropout de cette epoch
        if epoch >= t_early:
            rate = 0.0
        elif lineaire:
            rate = p * (1 - epoch / t_early)
        else:
            rate = p
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = rate

        # une epoch d'entraînement
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        # évaluation sans dropout
        model.eval()
        with torch.no_grad():
            out_tr, out_va, out_te = model(Xtr), model(data["Xva"]), model(data["Xte"])
            rows.append({
                "epoch": epoch, "rate": rate, "p": p, "t_early": t_early, "batch": batch,
                "train_loss": F.cross_entropy(out_tr, ytr).item(),
                "val_loss": F.cross_entropy(out_va, data["yva"]).item(),
                "val_acc": (out_va.argmax(1) == data["yva"]).float().mean().item(),
                "test_acc": (out_te.argmax(1) == data["yte"]).float().mean().item(),
            })

    return pd.DataFrame(rows)
