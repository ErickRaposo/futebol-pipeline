# -*- coding: utf-8 -*-
"""Funções de perda e componentes de otimização PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def ordinal_margin_loss(probs: torch.Tensor, y: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    """Loss auxiliar ordinal por margin ranking (vetorizada, sem loop sobre o batch).

    Para cada exemplo, a classe verdadeira `y` e a classe "mais distante" no
    espectro ordinal Home(0)-Draw(1)-Away(2) têm distância |i-j|; a mais distante
    é o anti-palpite máximo (ex.: y=Home -> mais distante Away). Penaliza via
    hinge loss quando P(classe verdadeira) não está pelo menos `margin` acima de
    P(classe mais distante):

        loss_i = relu(margin - (p_true - p_farthest))

    Ao somar (pesada por LAMBDA_ORDINAL) à CrossEntropy, força o modelo a manter
    a classe real com probabilidade mínima acima da classe do lado errado.
    """
    farthest = torch.where(y < 2 - y, torch.full_like(y, 2), torch.zeros_like(y))

    p_true = probs.gather(1, y.unsqueeze(1)).squeeze(1)
    p_farthest = probs.gather(1, farthest.unsqueeze(1)).squeeze(1)

    hinge = torch.clamp(margin - (p_true - p_farthest), min=0.0)
    return hinge.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - target_probs) ** self.gamma
        loss = -focal_weight * target_log_probs

        if self.alpha is not None:
            alpha_weights = self.alpha[targets]
            loss = loss * alpha_weights

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
