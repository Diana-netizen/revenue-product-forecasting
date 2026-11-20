"""Direct multi-horizon GRU with a publication-day-masked pinball loss."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..constants import HORIZON, LOOKBACK, QUANTILES, STEPS_PER_EPOCH, TARGETS
from ..folds import gather, to_quantiles, window_origins
from ._torch import DEVICE, seed_all

TUNE_TARGETS = TARGETS
QT = torch.tensor(QUANTILES, dtype=torch.float32, device=DEVICE)


class DirectMultiHorizonGRU(nn.Module):
    def __init__(self, n_hist, n_fut, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.encoder = nn.GRU(n_hist, hidden, num_layers=layers, batch_first=True,
                              dropout=dropout if layers > 1 else 0.0)
        self.decoder = nn.Sequential(nn.Linear(hidden + n_fut, hidden), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, 6))

    def forward(self, xh, xf):
        _, h = self.encoder(xh)
        ctx = h[-1].unsqueeze(1).expand(-1, xf.size(1), -1)
        out = self.decoder(torch.cat([ctx, xf], dim=-1))
        return out.view(out.size(0), out.size(1), 2, 3)


def masked_pinball(pred, target, weight):
    """pred (B,H,2,3), target (B,H,2), weight (B,H): publication-day mask."""
    err = target.unsqueeze(-1) - pred
    loss = torch.maximum(QT * err, (QT - 1) * err)
    w = weight[:, :, None, None]
    return (loss * w).sum() / (w.sum() * 6).clamp_min(1.0)


def _eval_tensors(fd, with_targets):
    n = fd.n_train
    t = lambda a: torch.as_tensor(np.stack(a), device=DEVICE)
    Hv = t([fd.hist[p][n[p] - LOOKBACK:n[p]] for p in fd.products])
    Fv = t([fd.fut[p][n[p]:n[p] + HORIZON] for p in fd.products])
    Wv = t([fd.eval_pub[p] for p in fd.products])
    Yv = t([fd.y_eval[p] for p in fd.products]) if with_targets else None
    return Hv, Fv, Yv, Wv


def fit(fd, params, steps, track=True, targets=None, save_dir=None):
    seed_all()
    H, Fu, Y, W = (torch.as_tensor(a, device=DEVICE) for a in gather(fd, window_origins(fd)))
    model = DirectMultiHorizonGRU(H.shape[-1], Fu.shape[-1], params["hidden"], params["layers"],
                                  params["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    if track:
        Hv, Fv, Yv, Wv = _eval_tensors(fd, True)
    curve = []
    for _ in range(steps):
        model.train()
        for _ in range(STEPS_PER_EPOCH):
            idx = torch.randint(0, len(H), (params["batch_size"],), device=DEVICE)
            opt.zero_grad(set_to_none=True)
            masked_pinball(model(H[idx], Fu[idx]), Y[idx], W[idx]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if track:
            model.eval()
            with torch.no_grad():
                curve.append(masked_pinball(model(Hv, Fv), Yv, Wv).item())
    return model, curve


def predict(model, fd, steps=None):
    Hv, Fv, _, _ = _eval_tensors(fd, False)
    model.eval()
    with torch.no_grad():
        out = model(Hv, Fv).cpu().numpy()
    return {p: to_quantiles(out[k], fd, p) for k, p in enumerate(fd.products)}


def save(model, directory, params=None):
    d = Path(directory); d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "gru_state.pt")
    (d / "gru_params.json").write_text(json.dumps(params or {}))
    # architecture inputs, needed to rebuild the network when serving
    (d / "gru_shape.json").write_text(json.dumps(dict(n_hist=model.encoder.input_size,
                                                      n_fut=model.decoder[0].in_features - model.encoder.hidden_size)))


DEFAULT = dict(hidden=64, layers=2, dropout=0.1, lr=1e-3, batch_size=32, weight_decay=0.0)


def sample(rng):
    return dict(hidden=int(rng.choice([32, 64, 128])), layers=int(rng.choice([1, 2])),
                dropout=float(rng.choice([0.0, 0.1, 0.2, 0.3])),
                lr=float(np.exp(rng.uniform(np.log(3e-4), np.log(3e-3)))),
                batch_size=int(rng.choice([16, 32, 64])), weight_decay=float(rng.choice([0.0, 1e-5, 1e-4])))