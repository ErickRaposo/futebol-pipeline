# -*- coding: utf-8 -*-
"""Calibração de probabilidades pós-processamento com Temperature Scaling."""

import torch
import torch.nn.functional as F
import numpy as np

from src.model import TemperatureScaler


def calibrate_and_predict(model, val_loader, test_loader, device):
    """Ajusta o calibrador de temperatura usando o conjunto de VALIDAÇÃO do fold
    e gera previsões calibradas para o conjunto de TESTE out-of-sample.
    """
    model.eval()

    # 1. Extração dos logits e rótulos da VALIDAÇÃO
    val_logits_list, val_labels_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            x_h, x_a = batch['x_home'].to(device), batch['x_away'].to(device)
            x_hm, x_am = batch['x_home_mando'].to(device), batch['x_away_mando'].to(device)
            m_h, m_a = batch['mask_home'].to(device), batch['mask_away'].to(device)
            m_hm, m_am = batch['mask_home_mando'].to(device), batch['mask_away_mando'].to(device)
            h_team, a_team = batch['home_team_id'].to(device), batch['away_team_id'].to(device)
            comp_id, match_feats, y = batch['comp_id'].to(device), batch['match_features'].to(device), batch['y'].to(device)

            logits, _, _ = model(x_h, x_a, x_hm, x_am, h_team, a_team, comp_id, match_feats,
                                 mask_home=m_h, mask_away=m_a, mask_home_mando=m_hm, mask_away_mando=m_am)
            val_logits_list.append(logits)
            val_labels_list.append(y)

    val_logits_tensor = torch.cat(val_logits_list, dim=0)
    val_labels_tensor = torch.cat(val_labels_list, dim=0)

    # 2. Ajuste do TemperatureScaler APENAS na VALIDAÇÃO
    scaler = TemperatureScaler().to(device)
    fitted_temperature = scaler.fit(val_logits_tensor, val_labels_tensor)

    # 3. Predição calibrada no conjunto de TESTE out-of-sample
    test_logits_list = []
    test_probs_calibrated_list = []

    with torch.no_grad():
        for batch in test_loader:
            x_h, x_a = batch['x_home'].to(device), batch['x_away'].to(device)
            x_hm, x_am = batch['x_home_mando'].to(device), batch['x_away_mando'].to(device)
            m_h, m_a = batch['mask_home'].to(device), batch['mask_away'].to(device)
            m_hm, m_am = batch['mask_home_mando'].to(device), batch['mask_away_mando'].to(device)
            h_team, a_team = batch['home_team_id'].to(device), batch['away_team_id'].to(device)
            comp_id, match_feats = batch['comp_id'].to(device), batch['match_features'].to(device)

            logits, _, _ = model(x_h, x_a, x_hm, x_am, h_team, a_team, comp_id, match_feats,
                                 mask_home=m_h, mask_away=m_a, mask_home_mando=m_hm, mask_away_mando=m_am)
            
            calibrated_logits = scaler(logits)
            probs = F.softmax(calibrated_logits, dim=1)

            test_logits_list.append(logits.cpu().numpy())
            test_probs_calibrated_list.append(probs.cpu().numpy())

    test_logits = np.concatenate(test_logits_list, axis=0)
    test_probs_calibrated = np.concatenate(test_probs_calibrated_list, axis=0)

    return test_logits, test_probs_calibrated, fitted_temperature
