"""Training and conservative ensemble scoring for R2.4.3.1."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from fgsp_ch.models.m63433524r2431_target_tail_preview import TargetTailPreviewHead


def train_target_tail_member(features, exit_event, at_risk, severity, *, seed, hidden_channels, epochs, learning_rate, severity_weight, device):
    x, event, risk, sev = (np.asarray(features, dtype=np.float32), np.asarray(exit_event, dtype=np.float32), np.asarray(at_risk, dtype=bool), np.asarray(severity, dtype=np.float32))
    if x.ndim != 2 or x.shape[1] != 84 or event.shape != (len(x), 4) or risk.shape != event.shape or sev.shape != event.shape:
        raise ValueError("invalid R2.4.3.1 arrays")
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
    torch.manual_seed(int(seed)); model = TargetTailPreviewHead(hidden_channels=int(hidden_channels)).to(device)
    tx, te, tr = (torch.as_tensor((x-mean)/scale, device=device), torch.as_tensor(event, device=device), torch.as_tensor(risk, device=device))
    ts = torch.as_tensor(np.log1p(np.maximum(sev, 0.0)), device=device)
    positive, negative = max(float(event[risk].sum()), 1.0), max(float(risk.sum()-event[risk].sum()), 1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits, _, predicted_severity = model(tx)
        raw = nn.functional.binary_cross_entropy_with_logits(logits, te, pos_weight=torch.as_tensor(negative/positive, device=device), reduction="none")
        loss = (raw * tr.float()).sum() / tr.float().sum().clamp_min(1.0)
        loss = loss + float(severity_weight) * nn.functional.smooth_l1_loss(predicted_severity, ts)
        loss.backward(); optimizer.step()
    return model.eval(), mean.astype(np.float64), scale.astype(np.float64)


@torch.no_grad()
def ensemble_target_tail_score(members, features, *, upper_std_multiplier, device):
    x = np.asarray(features, dtype=np.float32); values = []
    for model, mean, scale in members:
        _, cumulative, _ = model(torch.as_tensor((x-mean)/scale, dtype=torch.float32, device=device))
        values.append(cumulative.cpu().numpy())
    stacked = np.stack(values); mean = stacked.mean(0)
    upper = np.maximum.accumulate(np.clip(mean + float(upper_std_multiplier)*stacked.std(0), 0.0, 1.0), axis=1)
    return mean, upper, stacked
