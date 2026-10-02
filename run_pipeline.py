# -*- coding: utf-8 -*-
"""Ponto de Entrada Principal para Execução do Pipeline Rolling Walk-Forward.

Atende a todos os Requisitos FASE 2:
- Executa a validação Walk-Forward com janela móvel (rolling).
- Utiliza a pré-computação causal única de features.
- Valida o isolamento de scalers, checkpoints e calibração por fold.
- Gera relatório completo com a baseline das Odds do Mercado e métricas probabilísticas.
- Suporta GPU automaticamente se disponível.
"""

import os
import time
import torch

# Paralelismo CPU de fallback
if not torch.cuda.is_available():
    num_cpus = os.cpu_count() or 4
    torch.set_num_threads(num_cpus)

import src.config as cfg
from src.data_loader import load_matches_df
from src.features import build_causal_features
from src.folds import generate_walk_forward_folds
from src.trainer import run_fold_training
from src.reports import print_and_save_report


def main():
    print("=" * 80, flush=True)
    print("INICIANDO PIPELINE DE PREVISÃO DE FUTEBOL - ROLLING WALK-FORWARD", flush=True)
    device_name = "GPU (" + torch.cuda.get_device_name(0) + ")" if torch.cuda.is_available() else f"CPU ({torch.get_num_threads()} threads)"
    print(f"DISPOSITIVO DE EXECUÇÃO: {device_name}", flush=True)
    print("=" * 80, flush=True)

    start_time = time.time()

    # 1. Carregamento dos dados brutos do SQLite
    print(f"\n[1/5] Carregando banco de dados de partidas de '{cfg.DB_PATH}'...", flush=True)
    df_matches = load_matches_df(cfg.DB_PATH)
    print(f"      Total de partidas encontradas: {len(df_matches)}", flush=True)

    # 2. Pré-computação causal única (Princípio de Computação)
    print("\n[2/5] Executando passada única causal de extração de features...", flush=True)
    data_dict = build_causal_features(
        df_matches,
        start_date=cfg.START_DATE,
        max_seq_len=cfg.MAX_SEQ_LEN,
        max_mando_len=cfg.MAX_MANDO_LEN,
        odds_sequencia=cfg.ODDS_SEQUENCIA
    )
    print(f"      Partidas no dataset ativo (>= {cfg.START_DATE}): {len(data_dict['y'])}", flush=True)

    # 3. Geração automática dos Folds do Walk-Forward
    print("\n[3/5] Gerando janelas e folds do Rolling Walk-Forward...", flush=True)
    folds = generate_walk_forward_folds(
        data_dict['dates'],
        window_type=cfg.WINDOW_TYPE,
        train_window=cfg.TRAIN_WINDOW,
        val_window=cfg.VALIDATION_WINDOW,
        test_window=cfg.TEST_WINDOW,
        step=cfg.STEP,
        window_unit=cfg.WINDOW_UNIT
    )
    print(f"      Total de folds gerados automaticamente: {len(folds)}", flush=True)

    for f in folds:
        print(f"      - Fold {f['fold']}: Treino {f['train_period']} ({len(f['train_indices'])} jogos) | "
              f"Val {f['val_period']} ({len(f['val_indices'])} jogos) | "
              f"Teste {f['test_period']} ({len(f['test_indices'])} jogos)", flush=True)

    # 4. Treinamento e Avaliação por Fold
    print("\n[4/5] Treinando modelo, calibrando e avaliando por fold...", flush=True)
    fold_results = []
    for f in folds:
        print(f"\n---> Executando FOLD {f['fold']}/{len(folds)} <---", flush=True)
        res = run_fold_training(f, data_dict, cfg)
        fold_results.append(res)

    # 5. Geração e Salvamento do Relatório
    print("\n[5/5] Gerando relatório consolidado do experimento...", flush=True)
    print_and_save_report(fold_results, cfg)

    elapsed = time.time() - start_time
    print(f"Pipeline concluído com sucesso em {elapsed:.2f} segundos!", flush=True)


if __name__ == "__main__":
    main()
