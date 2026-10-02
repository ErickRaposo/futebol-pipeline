# -*- coding: utf-8 -*-
"""Engine de Treinamento e Avaliação por Fold com Suporte a Múltiplas Seeds.

Garante isolamento absoluto entre folds:
- Scaler fitado apenas com treino.
- Checkpoint selecionado pela menor loss de validação.
- Calibração de probabilidade fitada apenas na validação.
- Limpeza de memória CUDA e destruição do modelo ao término de cada fold.
"""

import copy
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.model import SimpleFootballTransformer
from src.loss import ordinal_margin_loss
from src.calibration import calibrate_and_predict
from src.metrics import calcular_metricas_completas, calcular_baseline_odds_mercado
from src.leakage_checks import check_fold_boundaries, check_scaler_isolation, check_test_isolation


class FastDataset(Dataset):
    def __init__(self, data_dict, indices, match_features_tensor):
        self.x_home = data_dict['x_home'][indices]
        self.x_away = data_dict['x_away'][indices]
        self.x_home_mando = data_dict['x_home_mando'][indices]
        self.x_away_mando = data_dict['x_away_mando'][indices]
        self.mask_home = data_dict['mask_home'][indices]
        self.mask_away = data_dict['mask_away'][indices]
        self.mask_home_mando = data_dict['mask_home_mando'][indices]
        self.mask_away_mando = data_dict['mask_away_mando'][indices]
        self.h_team = data_dict['home_team_id'][indices]
        self.a_team = data_dict['away_team_id'][indices]
        self.comp = data_dict['comp_id'][indices]
        self.match_features = match_features_tensor[indices]
        self.y = data_dict['y'][indices]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            'x_home': self.x_home[idx],
            'x_away': self.x_away[idx],
            'x_home_mando': self.x_home_mando[idx],
            'x_away_mando': self.x_away_mando[idx],
            'mask_home': self.mask_home[idx],
            'mask_away': self.mask_away[idx],
            'mask_home_mando': self.mask_home_mando[idx],
            'mask_away_mando': self.mask_away_mando[idx],
            'home_team_id': self.h_team[idx],
            'away_team_id': self.a_team[idx],
            'comp_id': self.comp[idx],
            'match_features': self.match_features[idx],
            'y': self.y[idx]
        }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_fold_training(fold_info: dict, data_dict: dict, cfg) -> dict:
    """Executa o pipeline de treino, validação, calibração e teste para um único fold.
    
    Roda N_SEEDS para o fold e calcula médias e desvios padrão.
    """
    train_idx = fold_info['train_indices']
    val_idx = fold_info['val_indices']
    test_idx = fold_info['test_indices']

    dates = data_dict['dates']
    train_dates = dates[train_idx]
    val_dates = dates[val_idx]
    test_dates = dates[test_idx]

    # 1. Verifica os limites temporais e não sobreposição do fold
    check_fold_boundaries(train_dates, val_dates, test_dates, train_idx, val_idx, test_idx)

    # 2. Scaler do elo_diff fitado SOMENTE no treino do fold
    raw_elo_diffs = data_dict['match_features'][:, 0].numpy()
    scaler_mean = float(np.mean(raw_elo_diffs[train_idx]))
    scaler_std = float(np.std(raw_elo_diffs[train_idx])) + 1e-6

    # Verifica isolamento do scaler
    check_scaler_isolation(train_idx, val_idx, scaler_mean, scaler_std, raw_elo_diffs)

    # Aplica a transformação de escala no tensor de match_features
    match_features_scaled = data_dict['match_features'].clone()
    match_features_scaled[:, 0] = (match_features_scaled[:, 0] - scaler_mean) / scaler_std

    # 3. Métricas da Baseline do Mercado nos mesmos jogos de teste
    test_y = data_dict['y'][test_idx].numpy()
    test_market_odds = data_dict['market_odds'][test_idx]
    market_baseline_metrics = calcular_baseline_odds_mercado(test_y, test_market_odds)

    seed_results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for s_idx in range(cfg.N_SEEDS):
        current_seed = cfg.SEED + s_idx
        set_seed(current_seed)

        train_ds = FastDataset(data_dict, train_idx, match_features_scaled)
        val_ds = FastDataset(data_dict, val_idx, match_features_scaled)
        test_ds = FastDataset(data_dict, test_idx, match_features_scaled)

        gen = torch.Generator().manual_seed(current_seed)
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE_TRAIN, shuffle=True, generator=gen)
        val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE_EVAL, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE_EVAL, shuffle=False)

        # Rebalanceamento de classes no treino
        train_y = data_dict['y'][train_idx].numpy()
        counts = np.bincount(train_y, minlength=3)
        total_samples = len(train_y)
        weights = (total_samples / (3.0 * counts)) ** cfg.EXPOENTE
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

        max_team = max(data_dict['home_team_id'].max().item(), data_dict['away_team_id'].max().item())
        max_comp = data_dict['comp_id'].max().item()
        num_match_feats = match_features_scaled.shape[1]
        num_seq_feats = data_dict['x_home'].shape[2]

        model = SimpleFootballTransformer(
            max_team_id=max_team,
            max_comp_id=max_comp,
            num_seq_features=num_seq_feats,
            num_match_features=num_match_feats,
            d_model=cfg.D_MODEL,
            nhead=cfg.NHEAD,
            num_layers=cfg.NUM_LAYERS,
            max_seq_len=cfg.MAX_SEQ_LEN,
            max_mando_len=cfg.MAX_MANDO_LEN,
            match_embed_dim=cfg.MATCH_EMBED_DIM,
            odds_sequencia=cfg.ODDS_SEQUENCIA,
            odds_skip_connection=cfg.ODDS_SKIP_CONNECTION
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-5)

        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        patience_counter = 0
        val_losses = []

        for epoch in range(1, cfg.EPOCHS + 1):
            model.train()
            for batch in train_loader:
                x_h, x_a = batch['x_home'].to(device), batch['x_away'].to(device)
                x_hm, x_am = batch['x_home_mando'].to(device), batch['x_away_mando'].to(device)
                m_h, m_a = batch['mask_home'].to(device), batch['mask_away'].to(device)
                m_hm, m_am = batch['mask_home_mando'].to(device), batch['mask_away_mando'].to(device)
                h_team, a_team = batch['home_team_id'].to(device), batch['away_team_id'].to(device)
                comp_id, match_feats, y = batch['comp_id'].to(device), batch['match_features'].to(device), batch['y'].to(device)

                optimizer.zero_grad()
                outputs, _, _ = model(x_h, x_a, x_hm, x_am, h_team, a_team, comp_id, match_feats,
                                      mask_home=m_h, mask_away=m_a, mask_home_mando=m_hm, mask_away_mando=m_am)
                probs = F.softmax(outputs, dim=1)
                loss = criterion(outputs, y) + cfg.LAMBDA_ORDINAL * ordinal_margin_loss(probs, y)
                loss.backward()
                optimizer.step()

            scheduler.step()

            # Avaliação na Validação
            model.eval()
            val_loss = 0.0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    x_h, x_a = batch['x_home'].to(device), batch['x_away'].to(device)
                    x_hm, x_am = batch['x_home_mando'].to(device), batch['x_away_mando'].to(device)
                    m_h, m_a = batch['mask_home'].to(device), batch['mask_away'].to(device)
                    m_hm, m_am = batch['mask_home_mando'].to(device), batch['mask_away_mando'].to(device)
                    h_team, a_team = batch['home_team_id'].to(device), batch['away_team_id'].to(device)
                    comp_id, match_feats, y = batch['comp_id'].to(device), batch['match_features'].to(device), batch['y'].to(device)

                    outputs, _, _ = model(x_h, x_a, x_hm, x_am, h_team, a_team, comp_id, match_feats,
                                          mask_home=m_h, mask_away=m_a, mask_home_mando=m_hm, mask_away_mando=m_am)
                    v_loss = criterion(outputs, y)
                    val_loss += v_loss.item() * len(y)
                    val_total += len(y)

            val_loss /= val_total
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.PATIENCE:
                    break

        print(f"      Seed {current_seed} | Melhor Época: {best_epoch}/{cfg.EPOCHS} | Best Val Loss: {best_val_loss:.4f}", flush=True)

        # Carrega o melhor checkpoint e verifica isolamento
        model.load_state_dict(best_model_state)
        check_test_isolation(best_epoch, val_losses)

        # Calibração e Predição Probabilística no Teste
        _, test_probs_calibrated, fitted_temp = calibrate_and_predict(model, val_loader, test_loader, device)

        # Cálculo das Métricas no Conjunto de Teste Out-of-Sample
        test_metrics = calcular_metricas_completas(test_y, test_probs_calibrated)
        test_metrics['fitted_temperature'] = fitted_temp
        test_metrics['best_epoch'] = best_epoch
        seed_results.append(test_metrics)

        print(f"      Teste Out-of-Sample | Acc: {test_metrics['accuracy']:.2f}% | LogLoss: {test_metrics['log_loss']:.4f} | RPS: {test_metrics['rps']:.4f} | Temp: {fitted_temp:.3f}", flush=True)

        # Libera memória PyTorch / CUDA
        del model, optimizer, train_loader, val_loader, test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        'fold': fold_info['fold'],
        'train_period': fold_info['train_period'],
        'val_period': fold_info['val_period'],
        'test_period': fold_info['test_period'],
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx),
        'model_metrics': seed_results,
        'market_baseline_metrics': market_baseline_metrics
    }
