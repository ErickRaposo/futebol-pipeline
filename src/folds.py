# -*- coding: utf-8 -*-
"""Gerador automático de folds para Rolling Walk-Forward Cross-Validation.

Atende às exigências de isolamento temporal:
- Janela de treino de tamanho fixo (rolling) que desliza a cada passo.
- Sem datas hardcoded; os limites são derivados dinamicamente das datas dos dados.
- Garante separação estrita entre treino, validação e teste.
"""

from typing import List, Dict, Any
import numpy as np


def generate_walk_forward_folds(
    dates_array: np.ndarray,
    window_type: str = "rolling",
    train_window: int = 4,
    val_window: int = 1,
    test_window: int = 1,
    step: int = 1,
    window_unit: str = "CALENDAR_YEAR"
) -> List[Dict[str, Any]]:
    """Gera automaticamente a lista de folds para a validação walk-forward.
    
    Parâmetros:
      - dates_array: Array com as datas das partidas em formato 'YYYY-MM-DD'.
      - window_type: "rolling" (tamanho fixo de treino) ou "expanding" (janela expansiva).
      - train_window: Tamanho da janela de treino em unidades de tempo.
      - val_window: Tamanho da janela de validação em unidades de tempo.
      - test_window: Tamanho da janela de teste em unidades de tempo.
      - step: Passo de avanço da janela.
      - window_unit: "CALENDAR_YEAR" (ano civil) ou "SEASON".
      
    Retorna:
      Lista de dicionários contendo metadados e índices do dataset por fold.
    """
    if len(dates_array) == 0:
        raise ValueError("O array de datas está vazio.")

    # Extrai os anos civis do array de datas
    years = np.array([int(d.split('-')[0]) for d in dates_array])
    min_year = int(years.min())
    max_year = int(years.max())

    folds = []
    fold_counter = 1

    # Inicia a partir do primeiro ano com histórico completo pós burn-in
    curr_start_year = min_year

    while True:
        if window_type == "rolling":
            train_start_year = curr_start_year
        else:
            # Mode expanding: treino sempre inicia no ano mínimo
            train_start_year = min_year

        train_end_year = curr_start_year + train_window - 1
        val_start_year = train_end_year + 1
        val_end_year = val_start_year + val_window - 1
        test_start_year = val_end_year + 1
        test_end_year = test_start_year + test_window - 1

        # Critério de parada: se a janela de teste ultrapassar o último ano disponível
        if test_start_year > max_year:
            break

        if test_end_year > max_year:
            test_end_year = max_year

        train_start_str = f"{train_start_year:04d}-01-01"
        train_end_str = f"{train_end_year:04d}-12-31"
        val_start_str = f"{val_start_year:04d}-01-01"
        val_end_str = f"{val_end_year:04d}-12-31"
        test_start_str = f"{test_start_year:04d}-01-01"
        test_end_str = f"{test_end_year:04d}-12-31"

        train_indices = np.where((dates_array >= train_start_str) & (dates_array <= train_end_str))[0]
        val_indices = np.where((dates_array >= val_start_str) & (dates_array <= val_end_str))[0]
        test_indices = np.where((dates_array >= test_start_str) & (dates_array <= test_end_str))[0]

        # Inclui o fold se houver amostras válidas nos três conjuntos
        if len(train_indices) > 0 and len(val_indices) > 0 and len(test_indices) > 0:
            folds.append({
                'fold': fold_counter,
                'train_period': (train_start_str, train_end_str),
                'val_period': (val_start_str, val_end_str),
                'test_period': (test_start_str, test_end_str),
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices
            })
            fold_counter += 1

        curr_start_year += step

    return folds
