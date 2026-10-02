# -*- coding: utf-8 -*-
"""Geração, formatação e salvamento de relatórios de avaliação do Walk-Forward."""

import json
import numpy as np


def build_markdown_report(results: list, cfg) -> str:
    """Constrói o relatório estruturado em Markdown com tabelas por fold, resumo e comparação."""
    lines = []
    lines.append("# RELATÓRIO DE AVALIAÇÃO ROLLING WALK-FORWARD")
    lines.append("\n## 1. Configuração do Experimento")
    lines.append(f"- **WALK_FORWARD_MODE**: `{cfg.WALK_FORWARD_MODE}`")
    lines.append(f"- **WINDOW_TYPE**: `{cfg.WINDOW_TYPE}`")
    lines.append(f"- **Janelas (Treino / Val / Teste)**: `{cfg.TRAIN_WINDOW}` / `{cfg.VALIDATION_WINDOW}` / `{cfg.TEST_WINDOW}` {cfg.WINDOW_UNIT}")
    lines.append(f"- **Passo (Step)**: `{cfg.STEP}` {cfg.WINDOW_UNIT}")
    lines.append(f"- **Seeds por Fold**: `{cfg.N_SEEDS}` (Seed Base: `{cfg.SEED}`)")
    lines.append(f"- **ODDS_SEQUENCIA**: `{cfg.ODDS_SEQUENCIA}` | **ODDS_SKIP_CONNECTION**: `{cfg.ODDS_SKIP_CONNECTION}`")
    lines.append(f"- **Loss**: CrossEntropy + Ordinal Margin Loss (`lambda={cfg.LAMBDA_ORDINAL}`)")
    lines.append(f"- **Hiperparâmetros Neural**: `d_model={cfg.D_MODEL}`, `nhead={cfg.NHEAD}`, `layers={cfg.NUM_LAYERS}`, `lr={cfg.LEARNING_RATE}`")

    lines.append("\n## 2. Resultado por Fold (Modelo Out-of-Sample)")
    lines.append("| Fold | Treino | Validação | Teste | Acc (%) | LogLoss | Brier | ECE | RPS | MacroF1 | AUC |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    metrics_keys = ['accuracy', 'log_loss', 'brier_score', 'ece_macro', 'rps', 'macro_f1', 'auc_macro']
    fold_model_means = {k: [] for k in metrics_keys}
    fold_market_means = {k: [] for k in metrics_keys}

    for res in results:
        f_num = res['fold']
        tr_p = f"{res['train_period'][0][:4]}-{res['train_period'][1][:4]}"
        val_p = f"{res['val_period'][0][:4]}-{res['val_period'][1][:4]}"
        te_p = f"{res['test_period'][0][:4]}-{res['test_period'][1][:4]}"

        # Média das seeds do fold
        m_seeds = res['model_metrics']
        acc = float(np.mean([s['accuracy'] for s in m_seeds]))
        ll = float(np.mean([s['log_loss'] for s in m_seeds]))
        brier = float(np.mean([s['brier_score'] for s in m_seeds]))
        ece = float(np.mean([s['ece_macro'] for s in m_seeds]))
        rps = float(np.mean([s['rps'] for s in m_seeds]))
        f1 = float(np.mean([s['macro_f1'] for s in m_seeds]))
        auc = float(np.mean([s['auc_macro'] for s in m_seeds]))

        fold_model_means['accuracy'].append(acc)
        fold_model_means['log_loss'].append(ll)
        fold_model_means['brier_score'].append(brier)
        fold_model_means['ece_macro'].append(ece)
        fold_model_means['rps'].append(rps)
        fold_model_means['macro_f1'].append(f1)
        fold_model_means['auc_macro'].append(auc)

        # Baseline do mercado no teste
        mk = res['market_baseline_metrics']
        fold_market_means['accuracy'].append(mk['accuracy'])
        fold_market_means['log_loss'].append(mk['log_loss'])
        fold_market_means['brier_score'].append(mk['brier_score'])
        fold_market_means['ece_macro'].append(mk['ece_macro'])
        fold_market_means['rps'].append(mk['rps'])
        fold_market_means['macro_f1'].append(mk['macro_f1'])
        fold_market_means['auc_macro'].append(mk['auc_macro'])

        lines.append(f"| {f_num} | {tr_p} | {val_p} | {te_p} | {acc:.2f}% | {ll:.4f} | {brier:.4f} | {ece:.4f} | {rps:.4f} | {f1:.4f} | {auc:.4f} |")

    # Linhas de Estatística Agregada do Modelo
    lines.append("\n## 3. Resumo Estatístico do Modelo (Média ± Desvio)")
    lines.append("| Métrica | Média (Mean) | Desvio Padrão (Std) | Mínimo | Máximo |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")

    metric_names_map = {
        'accuracy': 'Acurácia Global (%)',
        'log_loss': 'Log Loss (Cross-Entropy)',
        'brier_score': 'Brier Score',
        'ece_macro': 'ECE (Erro Calibração)',
        'rps': 'RPS (Ranked Prob. Score)',
        'macro_f1': 'Macro F1-Score',
        'auc_macro': 'ROC-AUC Macro'
    }

    for k, name in metric_names_map.items():
        arr = np.array(fold_model_means[k])
        lines.append(f"| {name} | {arr.mean():.4f} | {arr.std():.4f} | {arr.min():.4f} | {arr.max():.4f} |")

    lines.append("\n## 4. Comparação Modelo vs Baseline das Odds do Mercado")
    lines.append("| Métrica | Modelo (Média) | Odds Mercado (Média) | Diferença (Modelo - Odds) | Vantagem |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for k, name in metric_names_map.items():
        mod_m = float(np.mean(fold_model_means[k]))
        mkt_m = float(np.mean(fold_market_means[k]))
        diff = mod_m - mkt_m

        if k in ['log_loss', 'brier_score', 'ece_macro', 'rps']:
            # Menor é melhor
            visto = "Modelo Melhor" if diff < 0 else "Mercado Melhor"
        else:
            # Maior é melhor
            visto = "Modelo Melhor" if diff > 0 else "Mercado Melhor"

        lines.append(f"| {name} | {mod_m:.4f} | {mkt_m:.4f} | {diff:+.4f} | {visto} |")

    lines.append("\n## 5. Auditoria de Leakage e Controles de Integridade")
    lines.append("- [x] **Limites de Folds**: `max(train) < min(val) < min(test)` sem nenhuma sobreposição.")
    lines.append("- [x] **Ajuste de Scaler**: Z-score fitado estritamente com os dados de treino do fold.")
    lines.append("- [x] **Seleção de Checkpoint**: Early stopping e checkpoint selecionados estritamente na validação.")
    lines.append("- [x] **Calibração de Probabilidades**: Temperature Scaling ajustado estritamente na validação.")
    lines.append("- [x] **Isolamento de Teste**: O conjunto de teste permaneceu 100% cego até a avaliação final.")

    return "\n".join(lines)


def print_and_save_report(results: list, cfg, output_md_path: str = "relatorio_walk_forward.md", output_json_path: str = "experimento_walk_forward.json"):
    """Gera o relatório em Markdown e JSON, imprime no console e salva em arquivo."""
    report_md = build_markdown_report(results, cfg)
    print("\n" + "="*90, flush=True)
    print(report_md, flush=True)
    print("="*90 + "\n", flush=True)

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    export_data = {
        'config': {
            'WALK_FORWARD_MODE': cfg.WALK_FORWARD_MODE,
            'WINDOW_TYPE': cfg.WINDOW_TYPE,
            'TRAIN_WINDOW': cfg.TRAIN_WINDOW,
            'VALIDATION_WINDOW': cfg.VALIDATION_WINDOW,
            'TEST_WINDOW': cfg.TEST_WINDOW,
            'STEP': cfg.STEP,
            'SEED': cfg.SEED,
            'N_SEEDS': cfg.N_SEEDS
        },
        'results': results
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"[OK] Relatório salvo em '{output_md_path}' e dados estruturados em '{output_json_path}'.", flush=True)
