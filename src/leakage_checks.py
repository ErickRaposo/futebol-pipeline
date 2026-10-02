# -*- coding: utf-8 -*-
"""Checks automáticos anti-leakage com falha explícita (raise) caso violação ocorra."""

import numpy as np


def check_fold_boundaries(train_dates: np.ndarray, val_dates: np.ndarray, test_dates: np.ndarray,
                          train_indices: np.ndarray, val_indices: np.ndarray, test_indices: np.ndarray):
    """Verifica rigorosamente os limites temporais e a ausência de sobreposição nos folds."""
    max_train_date = str(sorted(train_dates)[-1])
    min_val_date = str(sorted(val_dates)[0])
    max_val_date = str(sorted(val_dates)[-1])
    min_test_date = str(sorted(test_dates)[0])

    if not (max_train_date < min_val_date):
        raise ValueError(
            f"[VAZAMENTO TEMPORAL DETECTADO] Data máxima de treino ({max_train_date}) "
            f"não é estritamente menor que a data mínima de validação ({min_val_date})."
        )

    if not (max_val_date < min_test_date):
        raise ValueError(
            f"[VAZAMENTO TEMPORAL DETECTADO] Data máxima de validação ({max_val_date}) "
            f"não é estritamente menor que a data mínima de teste ({min_test_date})."
        )

    s_train = set(train_indices)
    s_val = set(val_indices)
    s_test = set(test_indices)

    if len(s_train & s_val) > 0:
        raise ValueError("[SOBREPOSIÇÃO DE FOLD] Há índices compartilhados entre treino e validação.")

    if len(s_val & s_test) > 0:
        raise ValueError("[SOBREPOSIÇÃO DE FOLD] Há índices compartilhados entre validação e teste.")

    if len(s_train & s_test) > 0:
        raise ValueError("[SOBREPOSIÇÃO DE FOLD] Há índices compartilhados entre treino e teste.")


def check_scaler_isolation(train_indices: np.ndarray, val_indices: np.ndarray,
                           scaler_mean: float, scaler_std: float,
                           raw_elo_diffs: np.ndarray):
    """Verifica se o scaler foi fitado exclusivamente com os dados do conjunto de treino."""
    train_mean = float(np.mean(raw_elo_diffs[train_indices]))
    train_std = float(np.std(raw_elo_diffs[train_indices]))

    if abs(scaler_mean - train_mean) > 1e-5 or abs(scaler_std - train_std) > 1e-5:
        raise ValueError(
            f"[VAZAMENTO DE SCALER] Os parâmetros do scaler (média={scaler_mean:.4f}, std={scaler_std:.4f}) "
            f"diferem da média e desvio padrão reais do conjunto de treino (média={train_mean:.4f}, std={train_std:.4f})."
        )


def check_test_isolation(chosen_epoch: int, val_losses: list):
    """Verifica se o checkpoint selecionado é o de menor loss de validação."""
    best_epoch_expected = int(np.argmin(val_losses)) + 1
    if chosen_epoch != best_epoch_expected:
        raise ValueError(
            f"[VIOLAÇÃO DE CHECKPOINT] Época selecionada ({chosen_epoch}) diverge da época de menor "
            f"loss de validação ({best_epoch_expected})."
        )
