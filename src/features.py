# -*- coding: utf-8 -*-
"""Pré-computação causal de features em passada única cronológica.

Atende ao Princípio de Computação:
- Features causais por construção (Elo, sequências, H2H, rolling stats, descanso)
  são pré-computadas UMA única vez numa passada cronológica sobre o histórico.
- Para cada partida, as features são calculadas estritamente com o estado anterior
  e o estado só é atualizado após a montagem do vetor da partida.
- O período de burn-in/aquecimento (jogos anteriores a `start_date`, ex.: 2012)
  serve para acumular Elo, H2H e histórico real dos times antes da geração dos
  tensores de treino, garantindo que os primeiros folds não comecem com estados vazios.
"""

import numpy as np
import pandas as pd
import torch


BASE_RATE_ODDS = np.array([0.446588951990549, 0.2687988385498954, 0.2846122094595556])


def build_causal_features(df: pd.DataFrame, start_date: str = '2013-01-01',
                         max_seq_len: int = 15, max_mando_len: int = 8,
                         odds_sequencia: bool = True) -> dict:
    """Constrói tensores PyTorch e dicionário de dados em passada única causal.
    
    Parâmetros:
      - df: DataFrame de partidas ordenado por `date ASC, match_id ASC`.
      - start_date: Data limite inferior para inclusão dos tensores no dataset final.
        Partidas com data < start_date servem para aquecimento (burn-in) do Elo/histórico.
      - max_seq_len: Tamanho da sequência do histórico geral dos times.
      - max_mando_len: Tamanho da sequência do histórico por mando (casa/fora).
      - odds_sequencia: Se True, inclui as odds no histórico da sequência;
        se False, inclui as odds no vetor da partida atual.
    
    Retorna:
      Dicionário com tensores PyTorch, datas, probabilidades implícitas do mercado
      e metadados para indexação nos folds.
    """
    res_map = {'H': 0, 'D': 1, 'A': 2}
    
    # Estados acumulados causais
    team_history = {}
    team_home_history = {}
    team_away_history = {}
    team_raw_history = {}
    h2h_history = {}
    team_elo = {}

    num_seq_features = 14 if odds_sequencia else 11

    # Listas de tensores para partidas a partir de start_date
    x_home_list, x_away_list = [], []
    x_home_mando_list, x_away_mando_list = [], []
    mask_home_list, mask_away_list = [], []
    mask_home_mando_list, mask_away_mando_list = [], []
    h_team_list, a_team_list, comp_list = [], [], []
    match_features_list, y_list, date_list = [], [], []
    market_odds_list = []
    match_id_list = []

    HOME_ADV = 60.0
    K = 30.0

    for row in df.itertuples(index=False):
        m_date = str(row.date)
        if pd.isna(row.home_team_id) or pd.isna(row.away_team_id) or pd.isna(row.competition_id):
            continue
            
        h_id, a_id = int(row.home_team_id), int(row.away_team_id)
        c_id = int(row.competition_id)
        m_id = int(row.match_id)

        try:
            h_g = int(float(row.home_goals)) if pd.notna(row.home_goals) else 0
        except (ValueError, TypeError):
            h_g = 0

        try:
            a_g = int(float(row.away_goals)) if pd.notna(row.away_goals) else 0
        except (ValueError, TypeError):
            a_g = 0

        res = str(row.result).strip() if pd.notna(row.result) else ""

        try:
            h_rest = float(row.home_rest_days) if pd.notna(row.home_rest_days) else 4.0
        except (ValueError, TypeError):
            h_rest = 4.0

        try:
            a_rest = float(row.away_rest_days) if pd.notna(row.away_rest_days) else 4.0
        except (ValueError, TypeError):
            a_rest = 4.0

        # Probabilidades implícitas do mercado (1/odds) normalizadas
        try:
            o_h = float(row.home_odds) if pd.notna(row.home_odds) else None
            o_d = float(row.draw_odds) if pd.notna(row.draw_odds) else None
            o_a = float(row.away_odds) if pd.notna(row.away_odds) else None
        except (ValueError, TypeError):
            o_h = o_d = o_a = None

        if o_h and o_d and o_a and o_h > 0 and o_d > 0 and o_a > 0:
            inv = np.array([1.0 / o_h, 1.0 / o_d, 1.0 / o_a])
            odds_probs = inv / inv.sum()
        else:
            odds_probs = BASE_RATE_ODDS.copy()

        if res not in res_map:
            continue

        # 1. Leitura do Estado ANTERIOR (Pré-Jogo)
        h_elo = team_elo.get(h_id, 1500.0)
        a_elo = team_elo.get(a_id, 1500.0)
        elo_diff = (h_elo - a_elo)
        elo_match_balance = 1.0 / (1.0 + abs(elo_diff))

        h2h_key = (min(h_id, a_id), max(h_id, a_id))
        h2h_list = h2h_history.get(h2h_key, [])
        h2h_draw_rate = (sum(1 for r in h2h_list[-5:] if r == 'D') / float(len(h2h_list[-5:]))) if len(h2h_list) > 0 else 0.25

        h_raw = team_raw_history.get(h_id, [])
        a_raw = team_raw_history.get(a_id, [])
        h_draw_rate = (sum(1 for r in h_raw[-10:] if r['is_draw']) / float(len(h_raw[-10:]))) if len(h_raw) > 0 else 0.25
        a_draw_rate = (sum(1 for r in a_raw[-10:] if r['is_draw']) / float(len(a_raw[-10:]))) if len(a_raw) > 0 else 0.25

        draw_similarity = 1.0 - abs(h_draw_rate - a_draw_rate)

        h_diffs = [r['diff'] for r in h_raw[-10:]]
        h_vol_norm = min(float(np.var(h_diffs)) / 4.0, 1.0) if len(h_diffs) > 1 else 0.0
        a_diffs = [r['diff'] for r in a_raw[-10:]]
        a_vol_norm = min(float(np.var(a_diffs)) / 4.0, 1.0) if len(a_diffs) > 1 else 0.0

        volatility_diff = abs(h_vol_norm - a_vol_norm)

        rest_diff = (h_rest - a_rest) / 14.0
        rest_sum = (h_rest + a_rest) / 28.0

        match_feature_vector = [
            elo_diff,
            elo_match_balance,
            h2h_draw_rate,
            rest_diff,
            rest_sum,
            draw_similarity,
            volatility_diff
        ]

        if not odds_sequencia:
            match_feature_vector.extend(list(odds_probs))

        # 2. Se a partida está na janela ativa (>= start_date), salva nos tensores
        if m_date >= start_date:
            hist_h = team_history.get(h_id, [])
            num_real_h = min(len(hist_h), max_seq_len)
            pad_h = max_seq_len - num_real_h
            seq_h = list(hist_h)[-max_seq_len:]
            while len(seq_h) < max_seq_len:
                seq_h.insert(0, [0.0] * num_seq_features)
            mask_h = [True] * pad_h + [False] * num_real_h

            hist_a = team_history.get(a_id, [])
            num_real_a = min(len(hist_a), max_seq_len)
            pad_a = max_seq_len - num_real_a
            seq_a = list(hist_a)[-max_seq_len:]
            while len(seq_a) < max_seq_len:
                seq_a.insert(0, [0.0] * num_seq_features)
            mask_a = [True] * pad_a + [False] * num_real_a

            hist_hm = team_home_history.get(h_id, [])
            num_real_hm = min(len(hist_hm), max_mando_len)
            pad_hm = max_mando_len - num_real_hm
            seq_hm = list(hist_hm)[-max_mando_len:]
            while len(seq_hm) < max_mando_len:
                seq_hm.insert(0, [0.0] * num_seq_features)
            mask_hm = [True] * pad_hm + [False] * num_real_hm

            hist_am = team_away_history.get(a_id, [])
            num_real_am = min(len(hist_am), max_mando_len)
            pad_am = max_mando_len - num_real_am
            seq_am = list(hist_am)[-max_mando_len:]
            while len(seq_am) < max_mando_len:
                seq_am.insert(0, [0.0] * num_seq_features)
            mask_am = [True] * pad_am + [False] * num_real_am

            x_home_list.append(seq_h)
            x_away_list.append(seq_a)
            x_home_mando_list.append(seq_hm)
            x_away_mando_list.append(seq_am)
            mask_home_list.append(mask_h)
            mask_away_list.append(mask_a)
            mask_home_mando_list.append(mask_hm)
            mask_away_mando_list.append(mask_am)

            h_team_list.append(h_id)
            a_team_list.append(a_id)
            comp_list.append(c_id)
            match_features_list.append(match_feature_vector)
            y_list.append(res_map[res])
            date_list.append(m_date)
            market_odds_list.append(odds_probs)
            match_id_list.append(m_id)

        # 3. ATUALIZAÇÃO DO ESTADO APÓS PROCESSAR A PARTIDA (Causalidade Estrita)
        e_h = 1.0 / (1.0 + 10.0 ** (-((h_elo + HOME_ADV) - a_elo) / 400.0))
        s_h = 1.0 if res == 'H' else (0.5 if res == 'D' else 0.0)
        
        g_diff = abs(h_g - a_g)
        mov = 1.0 if g_diff <= 1 else (1.5 if g_diff == 2 else (11.0 + g_diff) / 8.0)

        team_elo[h_id] = h_elo + K * mov * (s_h - e_h)
        team_elo[a_id] = a_elo + K * mov * ((1.0 - s_h) - (1.0 - e_h))

        h_pts = 3.0 if res == 'H' else (1.0 if res == 'D' else 0.0)
        a_pts = 3.0 if res == 'A' else (1.0 if res == 'D' else 0.0)

        h_diff = (h_g - a_g) / 5.0
        a_diff = (a_g - h_g) / 5.0
        h_clean = 1.0 if a_g == 0 else 0.0
        a_clean = 1.0 if h_g == 0 else 0.0
        h_fts = 1.0 if h_g == 0 else 0.0
        a_fts = 1.0 if a_g == 0 else 0.0
        is_d = 1.0 if res == 'D' else 0.0

        vec_h = [h_g / 5.0, a_g / 5.0, h_pts / 3.0, 1.0, min(h_rest, 14.0) / 14.0, h_diff, h_clean, h_fts, is_d, h_draw_rate, h_vol_norm]
        vec_a = [a_g / 5.0, h_g / 5.0, a_pts / 3.0, 0.0, min(a_rest, 14.0) / 14.0, a_diff, a_clean, a_fts, is_d, a_draw_rate, a_vol_norm]

        if odds_sequencia:
            vec_h.extend(list(odds_probs))
            vec_a.extend(list(odds_probs[::-1]))

        if h_id not in team_history: team_history[h_id] = []
        if a_id not in team_history: team_history[a_id] = []
        if h_id not in team_home_history: team_home_history[h_id] = []
        if a_id not in team_away_history: team_away_history[a_id] = []
        if h_id not in team_raw_history: team_raw_history[h_id] = []
        if a_id not in team_raw_history: team_raw_history[a_id] = []
        if h2h_key not in h2h_history: h2h_history[h2h_key] = []

        team_history[h_id].append(vec_h)
        team_history[a_id].append(vec_a)
        team_home_history[h_id].append(vec_h)
        team_away_history[a_id].append(vec_a)
        team_raw_history[h_id].append({'is_draw': is_d, 'diff': h_diff})
        team_raw_history[a_id].append({'is_draw': is_d, 'diff': a_diff})
        h2h_history[h2h_key].append(res)

    return {
        'x_home': torch.tensor(x_home_list, dtype=torch.float32),
        'x_away': torch.tensor(x_away_list, dtype=torch.float32),
        'x_home_mando': torch.tensor(x_home_mando_list, dtype=torch.float32),
        'x_away_mando': torch.tensor(x_away_mando_list, dtype=torch.float32),
        'mask_home': torch.tensor(mask_home_list, dtype=torch.bool),
        'mask_away': torch.tensor(mask_away_list, dtype=torch.bool),
        'mask_home_mando': torch.tensor(mask_home_mando_list, dtype=torch.bool),
        'mask_away_mando': torch.tensor(mask_away_mando_list, dtype=torch.bool),
        'home_team_id': torch.tensor(h_team_list, dtype=torch.long),
        'away_team_id': torch.tensor(a_team_list, dtype=torch.long),
        'comp_id': torch.tensor(comp_list, dtype=torch.long),
        'match_features': torch.tensor(match_features_list, dtype=torch.float32),
        'y': torch.tensor(y_list, dtype=torch.long),
        'dates': np.array(date_list),
        'market_odds': np.array(market_odds_list, dtype=np.float32),
        'match_ids': np.array(match_id_list, dtype=np.int64)
    }
