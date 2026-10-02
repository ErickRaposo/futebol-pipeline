# Football Result Prediction Transformer - Rolling Walk-Forward Pipeline

Pipeline modular de Deep Learning baseado em PyTorch e Transformer (`SimpleFootballTransformer`) para previsão probabilística de resultados de futebol (Vitória Mandante / Empate / Vitória Visitante: H/D/A), com validação rigorosa via **Rolling Walk-Forward Cross-Validation** e isolamento causal estrito.

---

## 🚀 Destaques da Arquitetura & Anti-Leakage

1. **Princípio de Computação Causal em Passada Única**:
   - As features causais por construção (Elo Rating dinâmico $K=30$, $HomeAdv=60$, histórico H2H, estatísticas móveis, descanso e sequências de mando) são calculadas **UMA única vez em ordem cronológica**.
   - Para cada partida $P$ na data $D$, apenas informações conhecidas **antes do início de $P$** entram no vetor de features. O estado só é atualizado após a montagem do vetor da partida.
   - Partidas do ano de 2012 servem de *burn-in / aquecimento* para acumulação inicial do Elo e históricos reais dos times antes do início do primeiro fold de treino.

2. **Rolling Walk-Forward Cross-Validation**:
   - Janela de treino móvel de tamanho fixo (ex: 4 anos de treino, 1 ano de validação, 1 ano de teste out-of-sample).
   - Folds gerados dinamicamente a partir das datas do banco de dados (sem datas hardcoded).
   - Isolamento total entre folds: Scalers ($Z$-score), Seleção de Checkpoint (Early Stopping na loss de validação) e Calibração de Probabilidades (Temperature Scaling) são ajustados **exclusivamente com o conjunto de treino/validação de cada fold**.

3. **Suporte Transparente a GPU & CPU**:
   - Utiliza automaticamente aceleração por GPU (`torch.cuda`) se disponível (NVIDIA CUDA / Colab / Kaggle).
   - Suporte a múltiplas threads de CPU caso rodando localmente sem GPU.

4. **Métricas Completas & Baseline das Odds de Mercado**:
   - Avalia Acurácia, Chance Dupla (Top-2), Log Loss, Brier Score, ECE (Expected Calibration Error), RPS (Ranked Probability Score), Macro F1, MCC e ROC-AUC.
   - Avalia a **Baseline do Mercado (Odds das casas de apostas)** nos exatos mesmos jogos de teste de cada fold para comparação direta e transparente.

---

## 📁 Estrutura de Pastas e Módulos

```
futebol_pipeline/
├── README.md                   # Documentação completa e instruções de execução
├── requirements.txt             # Dependências do projeto (PyTorch, Pandas, Scikit-Learn, etc.)
├── .gitignore                   # Arquivos ignorados pelo Git (bancos, checkpoints, caches)
├── run_tests.py                 # Suíte de testes automatizados e Teste Canário anti-leakage
├── run_pipeline.py              # Ponto de entrada principal do pipeline Walk-Forward
└── src/
    ├── __init__.py
    ├── config.py                # Hiperparâmetros centrais e variáveis editáveis
    ├── data_loader.py           # Carregamento do banco SQLite ordenado por data
    ├── features.py              # Extrator de features causais em passada única
    ├── folds.py                 # Gerador automático de janelas móveis (Walk-Forward)
    ├── model.py                 # Arquitetura PyTorch (SimpleFootballTransformer + TemperatureScaler)
    ├── loss.py                  # CrossEntropy rebalanceada + Loss Ordinal Auxiliar
    ├── metrics.py               # Métricas probabilísticas e Baseline das Odds
    ├── calibration.py           # Calibração via Temperature Scaling
    ├── leakage_checks.py        # Módulo de validação de isolamento estrito
    ├── trainer.py               # Engine de treino e avaliação por fold/seed
    └── reports.py               # Gerador de relatórios em Markdown e JSON
```

---

## 🛠️ Como Executar

### 1. Requisitos e Instalação

```bash
git clone https://github.com/SEU_USUARIO/futebol_pipeline.git
cd futebol_pipeline
pip install -r requirements.txt
```

> **Nota sobre o Banco de Dados (`futebol.db`)**:
> O pipeline procura por `futebol.db` no diretório atual ou no diretório pai (`../futebol.db`). Garanta que o banco de dados `futebol.db` esteja presente.

---

### 2. Executar Suíte de Testes Automatizados (com Teste Canário)

Para garantir que a causalidade temporal, a ausência de sobreposição nos folds e o isolamento de scalers estejam 100% operacionais, rode:

```bash
python run_tests.py
```

---

### 3. Executar o Pipeline Walk-Forward Completo

Para iniciar o treinamento e avaliação out-of-sample nos 9 folds (2013-2026):

```bash
python run_pipeline.py
```

No final da execução, os relatórios detalhados serão salvos em:
- `relatorio_walk_forward.md` (Relatório formatado em Markdown com tabelas)
- `experimento_walk_forward.json` (Dados estruturados JSON)

---

## 💻 Como Rodar na GPU (Google Colab ou Kaggle)

1. Faça upload da pasta `futebol_pipeline/` e do arquivo `futebol.db` para o seu Google Drive ou Kaggle Dataset.
2. Em um Notebook com GPU ativada (Runtime -> Change runtime type -> T4 GPU):

```python
!git clone https://github.com/SEU_USUARIO/futebol_pipeline.git
%cd futebol_pipeline

# Copie ou garanta o futebol.db na pasta
!cp /content/drive/MyDrive/futebol.db .

!pip install -r requirements.txt
!python run_tests.py
!python run_pipeline.py
```

---

## ⚙️ Configurações e Hiperparâmetros Editar (`src/config.py`)

No arquivo `src/config.py`, você pode ajustar facilmente:

- `TRAIN_WINDOW` / `VALIDATION_WINDOW` / `TEST_WINDOW`: Tamanho das janelas do Walk-Forward em anos.
- `BATCH_SIZE_TRAIN` / `BATCH_SIZE_EVAL`: Ajuste de batch para GPU (ex: 128 ou 256).
- `EPOCHS` / `PATIENCE`: Quantidade máxima de épocas e paciência para early stopping.
- `N_SEEDS`: Número de sementes aleatórias por fold para cálculo de desvio padrão.
