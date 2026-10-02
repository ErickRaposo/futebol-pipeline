# -*- coding: utf-8 -*-
"""Configurações globais e hiperparâmetros do projeto de previsão de futebol.

Todas as variáveis de execução, janelas de walk-forward, seeds e hiperparâmetros
estão centralizadas neste arquivo e podem ser alteradas sem modificar a lógica
dos módulos.
"""

import os

# ==============================================================================
# 1. CONFIGURAÇÕES DE VALIDAÇÃO WALK-FORWARD
# ==============================================================================
# Habilita o modo Rolling Walk-Forward Cross-Validation.
WALK_FORWARD_MODE = True

# Tipo de janela temporal: "rolling" (janela de treino de tamanho fixo deslizando)
WINDOW_TYPE = "rolling"

# Tamanho das janelas (medido na unidade configurada em WINDOW_UNIT)
TRAIN_WINDOW = 4          # Janela de treino (4 anos/temporadas)
VALIDATION_WINDOW = 1     # Janela de validação (1 ano/temporada)
TEST_WINDOW = 1           # Janela de teste out-of-sample (1 ano/temporada)
STEP = 1                  # Passo de deslizamento da janela (1 ano/temporada)

# Unidade temporal da janela: "CALENDAR_YEAR" ou "SEASON"
WINDOW_UNIT = "CALENDAR_YEAR"

# ==============================================================================
# 2. CONFIGURAÇÕES DE REPRODUTIBILIDADE E SEEDS
# ==============================================================================
SEED = 42
N_SEEDS = 1               # Quantidade de seeds a rodar por fold (reporta média/desvio)

# ==============================================================================
# 3. CONFIGURAÇÕES DE DADOS E ARMAZENAMENTO
# ==============================================================================
# Localização do banco SQLite: busca primeiro no diretório atual, depois no diretório pai
DB_PATH = "futebol.db" if os.path.exists("futebol.db") else ("../futebol.db" if os.path.exists("../futebol.db") else "futebol.db")
START_DATE = "2013-01-01"  # Data inicial para montagem de tensores de gradiente
                          # (jogos de 2012 servem como burn-in para Elo e históricos)

# ==============================================================================
# 4. FEATURE FLAGS E PIPELINE CAUSAL
# ==============================================================================
# Odds na sequência histórica dos times (True) ou no vetor da partida atual (False)
ODDS_SEQUENCIA = False

# Atalho direto (skip connection) das odds da partida para os logits finais
ODDS_SKIP_CONNECTION = False

# Comprimento máximo das sequências históricas
MAX_SEQ_LEN = 15          # Sequência geral (últimos N jogos)
MAX_MANDO_LEN = 8         # Sequência por mando (últimos N jogos casa/fora)

# ==============================================================================
# 5. HIPERPARÂMETROS DA REDE NEURAL (SIMPLE FOOTBALL TRANSFORMER)
# ==============================================================================
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
MATCH_EMBED_DIM = 32

# Loss composta
EXPOENTE = 0.3            # Expoente de rebalanceamento de classes (class_weight)
LAMBDA_ORDINAL = 0.2      # Peso da loss auxiliar ordinal por margin ranking

# Otimização
LEARNING_RATE = 0.0007
WEIGHT_DECAY = 1e-4
BATCH_SIZE_TRAIN = 128    # Tamanho de batch para treino (otimizado para GPU/CPU)
BATCH_SIZE_EVAL = 256     # Tamanho de batch para avaliação
EPOCHS = 30
PATIENCE = 5              # Patience do Early Stopping baseado no val_loss
