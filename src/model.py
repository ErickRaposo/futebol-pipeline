# -*- coding: utf-8 -*-
"""Módulos PyTorch da rede neural SimpleFootballTransformer e Calibrador.

Preserva 100% da arquitetura original com Positional Encoding aprendida,
Siamese Dual Sequence Attention Pooling, MatchFeatureEncoder com SE-Block
e suporte a Temperature Scaling para calibração.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sanitize_mask_for_encoder(mask):
    if mask is None:
        return None
    all_padded = mask.all(dim=1, keepdim=True)
    return mask & (~all_padded)


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super(AttentionPooling, self).__init__()
        self.attn_linear = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        x = torch.nan_to_num(x, nan=0.0)
        scores = self.attn_linear(x)
        
        if mask is not None:
            all_padded = mask.all(dim=1, keepdim=True)
            effective_mask = mask.unsqueeze(-1) & (~all_padded.unsqueeze(-1))
            scores = scores.masked_fill(effective_mask, float('-inf'))
            weights = torch.softmax(scores, dim=1)
            weights = weights.masked_fill(all_padded.unsqueeze(-1), 0.0)
        else:
            weights = torch.softmax(scores, dim=1)
        
        weights = torch.nan_to_num(weights, nan=0.0)
        context = torch.sum(weights * x, dim=1)
        context = torch.nan_to_num(context, nan=0.0)
        return context, weights


class MatchFeatureEncoder(nn.Module):
    def __init__(self, num_match_features: int = 7, hidden_dim: int = 64,
                 embed_dim: int = 32, dropout: float = 0.1):
        super(MatchFeatureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_match_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU()
        )
        
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, match_features):
        emb = self.encoder(match_features)
        gate_weights = self.gate(emb)
        scaled_emb = emb * gate_weights
        return scaled_emb


class SimpleFootballTransformer(nn.Module):
    def __init__(self, max_team_id: int = 500, max_comp_id: int = 50,
                 num_seq_features: int = 11, num_match_features: int = 7,
                 d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 max_seq_len: int = 15, max_mando_len: int = 8,
                 match_embed_dim: int = 32, odds_sequencia: bool = True,
                 odds_skip_connection: bool = False):
        super(SimpleFootballTransformer, self).__init__()
        
        self.team_emb_dim = 16
        self.comp_emb_dim = 16
        self.match_embed_dim = match_embed_dim
        self.odds_sequencia = odds_sequencia
        self.odds_skip_connection = odds_skip_connection

        self.team_emb = nn.Embedding(max_team_id + 1, embedding_dim=self.team_emb_dim)
        self.comp_emb = nn.Embedding(max_comp_id + 1, embedding_dim=self.comp_emb_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim=d_model)
        self.pos_mando_emb = nn.Embedding(max_mando_len, embedding_dim=d_model)

        self.input_proj = nn.Linear(num_seq_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, activation="gelu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attn_pooling = AttentionPooling(d_model)

        self.match_encoder = MatchFeatureEncoder(
            num_match_features=num_match_features,
            hidden_dim=64,
            embed_dim=match_embed_dim,
            dropout=0.1
        )

        self.odds_skip = nn.Linear(3, 3)

        concat_dim = (8 * d_model) + (2 * self.team_emb_dim) + self.comp_emb_dim + match_embed_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 3)
        )

    def forward(self, x_home, x_away, x_home_mando, x_away_mando,
                home_team_id, away_team_id, comp_id, match_features,
                mask_home=None, mask_away=None, mask_home_mando=None, mask_away_mando=None):
        seq_len = x_home.size(1)
        positions = torch.arange(seq_len, device=x_home.device).unsqueeze(0)
        pos_embedding = self.pos_emb(positions)

        enc_mask_home = sanitize_mask_for_encoder(mask_home)
        enc_mask_away = sanitize_mask_for_encoder(mask_away)
        enc_mask_hm = sanitize_mask_for_encoder(mask_home_mando)
        enc_mask_am = sanitize_mask_for_encoder(mask_away_mando)

        proj_home = self.input_proj(x_home) + pos_embedding
        out_home = self.transformer_encoder(proj_home, src_key_padding_mask=enc_mask_home)
        h_home, weights_home = self.attn_pooling(out_home, mask=mask_home)

        proj_away = self.input_proj(x_away) + pos_embedding
        out_away = self.transformer_encoder(proj_away, src_key_padding_mask=enc_mask_away)
        h_away, weights_away = self.attn_pooling(out_away, mask=mask_away)

        diff = h_home - h_away
        diff_abs = torch.abs(diff)
        prod = h_home * h_away

        seq_mando_len = x_home_mando.size(1)
        pos_mando = self.pos_mando_emb(torch.arange(seq_mando_len, device=x_home_mando.device).unsqueeze(0))

        proj_h_mando = self.input_proj(x_home_mando) + pos_mando
        out_h_mando = self.transformer_encoder(proj_h_mando, src_key_padding_mask=enc_mask_hm)
        h_home_mando, _ = self.attn_pooling(out_h_mando, mask=mask_home_mando)

        proj_a_mando = self.input_proj(x_away_mando) + pos_mando
        out_a_mando = self.transformer_encoder(proj_a_mando, src_key_padding_mask=enc_mask_am)
        h_away_mando, _ = self.attn_pooling(out_a_mando, mask=mask_away_mando)

        diff_mando = h_home_mando - h_away_mando

        emb_h = self.team_emb(home_team_id)
        emb_a = self.team_emb(away_team_id)
        emb_c = self.comp_emb(comp_id)

        match_emb = self.match_encoder(match_features)

        concat = torch.cat([
            h_home, h_away, diff, diff_abs, prod,
            h_home_mando, h_away_mando, diff_mando,
            emb_h, emb_a, emb_c, match_emb
        ], dim=1)

        logits = self.classifier(concat)

        if self.odds_skip_connection and not self.odds_sequencia:
            odds_direct = match_features[:, -3:]
            logits = logits + self.odds_skip(odds_direct)

        return logits, weights_home, weights_away


class TemperatureScaler(nn.Module):
    """Calibrador de probabilidades por Temperature Scaling.
    
    Ajusta um único parâmetro escalar `temperature` > 0 utilizando EXCLUSIVAMENTE
    o conjunto de validação do fold para minimizar a NLL (CrossEntropyLoss).
    """
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor):
        """Ajusta a temperatura otimizando CrossEntropyLoss no conjunto de validação."""
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        criterion = nn.CrossEntropyLoss()

        def eval_step():
            optimizer.zero_grad()
            loss = criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        # Garante temperatura estritamente positiva
        with torch.no_grad():
            self.temperature.clamp_(min=0.01)
        return float(self.temperature.item())
