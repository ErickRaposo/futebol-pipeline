# -*- coding: utf-8 -*-
"""Suíte de Testes Automatizados e Teste Canário de Anti-Leakage Causal."""

import unittest
import numpy as np
import torch
import pandas as pd

from src.folds import generate_walk_forward_folds
from src.leakage_checks import check_fold_boundaries, check_scaler_isolation, check_test_isolation
from src.features import build_causal_features
from src.metrics import calcular_metricas_completas, calcular_rps, calcular_ece


def run_all_tests():
    print("=" * 70)
    print("SUÍTE DE TESTES AUTOMATIZADOS - VALIDAÇÃO WALK-FORWARD E CAUSALIDADE")
    print("=" * 70)

    # Teste 1: Geração de Folds e Limites Temporais
    print("\n-> Testando Geração de Folds e Limites Temporais...")
    dates_mock = np.array([
        f"{y:04d}-{m:02d}-15"
        for y in range(2013, 2027)
        for m in range(1, 13)
    ])
    folds = generate_walk_forward_folds(
        dates_mock, window_type="rolling", train_window=4, val_window=1, test_window=1, step=1
    )
    assert len(folds) == 9, f"Esperado 9 folds, obtido {len(folds)}"

    for f in folds:
        tr_dates = dates_mock[f['train_indices']]
        val_dates = dates_mock[f['val_indices']]
        te_dates = dates_mock[f['test_indices']]
        check_fold_boundaries(tr_dates, val_dates, te_dates, f['train_indices'], f['val_indices'], f['test_indices'])
    print("   [OK] Geração de folds e limites temporais validados com sucesso.")

    # Teste 2: Teste Canário de Causalidade (Imutabilidade do Passado diante do Futuro)
    print("\n-> Executando TESTE CANÁRIO de Causalidade (Imutabilidade de Features de Passado)...")
    data_original = []
    for i in range(1, 101):
        d_str = f"2018-01-{(i % 28) + 1:02d}"
        data_original.append({
            'match_id': i,
            'date': d_str,
            'season': '2018',
            'competition_id': 1,
            'round': i % 38,
            'home_team_id': (i % 10) + 1,
            'away_team_id': ((i + 1) % 10) + 1,
            'home_goals': (i % 3),
            'away_goals': ((i + 1) % 3),
            'result': 'H' if (i % 3) > ((i + 1) % 3) else ('D' if (i % 3) == ((i + 1) % 3) else 'A'),
            'home_rest_days': 4.0,
            'away_rest_days': 4.0,
            'home_odds': 2.0,
            'draw_odds': 3.2,
            'away_odds': 3.5,
            'odds_open': None
        })

    df_orig = pd.DataFrame(data_original)
    feats_orig = build_causal_features(df_orig, start_date='2018-01-01', max_seq_len=5, max_mando_len=3)

    # Mutação drástica em dados de partidas FUTURAS (após partida 50)
    data_mutated = [dict(r) for r in data_original]
    for r in data_mutated[50:]:
        r['home_goals'] = 99
        r['away_goals'] = 88
        r['result'] = 'A'
        r['home_odds'] = 99.0

    df_mutated = pd.DataFrame(data_mutated)
    feats_mutated = build_causal_features(df_mutated, start_date='2018-01-01', max_seq_len=5, max_mando_len=3)

    # Valida se as primeiras 50 partidas permaneceram 100% IDÊNTICAS bit-a-bit
    diff_x_home = (feats_orig['x_home'][:50] - feats_mutated['x_home'][:50]).abs().max().item()
    diff_match_f = (feats_orig['match_features'][:50] - feats_mutated['match_features'][:50]).abs().max().item()

    assert diff_x_home == 0.0, f"Violação de Causalidade! diff_x_home={diff_x_home}"
    assert diff_match_f == 0.0, f"Violação de Causalidade! diff_match_f={diff_match_f}"
    print("   [OK] TESTE CANÁRIO PASSOU: Alterações em datas futuras NÃO afetam features passadas!")

    # Teste 3: Checagem de Isolamento do Scaler e Checkpoint
    print("\n-> Testando Isolamento de Scaler e Checkpoint...")
    raw_elo = np.random.randn(100)
    tr_idx = np.arange(0, 70)
    val_idx = np.arange(70, 100)

    scaler_m = float(np.mean(raw_elo[tr_idx]))
    scaler_s = float(np.std(raw_elo[tr_idx]))
    check_scaler_isolation(tr_idx, val_idx, scaler_m, scaler_s, raw_elo)

    val_losses = [1.2, 1.1, 0.95, 0.98, 1.05]
    check_test_isolation(chosen_epoch=3, val_losses=val_losses)
    print("   [OK] Checagens de isolamento de scaler e checkpoint validadas.")

    # Teste 4: Métricas Probabilísticas e RPS
    print("\n-> Testando Cálculo de Métricas (RPS, ECE, Brier, MCC)...")
    y_true_mock = np.array([0, 1, 2, 0, 1, 2])
    y_probs_mock = np.array([
        [0.7, 0.2, 0.1],
        [0.2, 0.6, 0.2],
        [0.1, 0.3, 0.6],
        [0.8, 0.1, 0.1],
        [0.2, 0.5, 0.3],
        [0.2, 0.2, 0.6]
    ])
    metrics = calcular_metricas_completas(y_true_mock, y_probs_mock)
    assert metrics['accuracy'] == 100.0
    assert metrics['rps'] < 0.1
    print("   [OK] Funções de métricas validadas com sucesso.")

    print("\n" + "=" * 70)
    print("TODOS OS TESTES PASSARAM COM SUCESSO! A INFRAESTRUTURA ESTÁ VALIDADA.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
