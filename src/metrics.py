# -*- coding: utf-8 -*-
"""Cálculo completo de métricas de avaliação probabilísticas e de classificação,
incluindo a Baseline das Odds do Mercado e RPS (Ranked Probability Score).
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef
)


def calcular_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10):
    """Calcula o Expected Calibration Error (ECE) médio entre as 3 classes."""
    total_samples = len(y_true)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece_per_class = []

    for k in range(3):
        prob_k = y_probs[:, k]
        true_k = (y_true == k).astype(float)
        ece_k = 0.0

        for i in range(n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
            in_bin = (prob_k > bin_lower) & (prob_k <= bin_upper)
            num_in_bin = np.sum(in_bin)
            if num_in_bin > 0:
                accuracy_in_bin = np.mean(true_k[in_bin])
                avg_confidence_in_bin = np.mean(prob_k[in_bin])
                ece_k += np.abs(accuracy_in_bin - avg_confidence_in_bin) * (num_in_bin / total_samples)

        ece_per_class.append(float(ece_k))

    return float(np.mean(ece_per_class)), ece_per_class


def calcular_rps(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Calcula o Ranked Probability Score (RPS) multiclasse ordenado H(0) < D(1) < A(2).

    RPS = (1 / (K-1)) * sum_{k=1}^{K-1} (P_k - O_k)^2
    onde P_k e O_k são as probabilidades acumuladas e observadas acumuladas.
    """
    n_samples = len(y_true)
    rps_list = []

    for i in range(n_samples):
        p = y_probs[i]
        y = y_true[i]

        p_cum1 = p[0]
        p_cum2 = p[0] + p[1]

        o_cum1 = 1.0 if y <= 0 else 0.0
        o_cum2 = 1.0 if y <= 1 else 0.0

        rps_i = 0.5 * ((p_cum1 - o_cum1) ** 2 + (p_cum2 - o_cum2) ** 2)
        rps_list.append(rps_i)

    return float(np.mean(rps_list))


def calcular_top2_accuracy(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Calcula a acurácia de Chance Dupla (Top-2): verdadeiro resultado entre as 2 maiores probs."""
    top2 = np.argsort(y_probs, axis=1)[:, -2:]
    hits = np.array([y_true[i] in top2[i] for i in range(len(y_true))])
    return float(np.mean(hits) * 100)


def calcular_metricas_completas(y_true: np.ndarray, y_probs: np.ndarray) -> dict:
    """Calcula o conjunto completo de métricas probabilísticas e de classificação."""
    y_preds = np.argmax(y_probs, axis=1)

    acc = float(np.mean(y_preds == y_true) * 100)
    l_loss = float(log_loss(y_true, y_probs, labels=[0, 1, 2]))
    
    y_true_onehot = np.eye(3)[y_true]
    brier = float(np.mean(np.sum((y_probs - y_true_onehot) ** 2, axis=1)))

    ece_macro, ece_per_class = calcular_ece(y_true, y_probs)
    rps = calcular_rps(y_true, y_probs)
    top2_acc = calcular_top2_accuracy(y_true, y_probs)

    auc_macro = float(roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro', labels=[0, 1, 2]))
    auc_per_class = [float(a) for a in roc_auc_score(y_true, y_probs, multi_class='ovr', average=None, labels=[0, 1, 2])]

    report = classification_report(
        y_true, y_preds,
        target_names=['Mandante (Home)', 'Empate (Draw)', 'Visitante (Away)'],
        output_dict=True,
        zero_division=0
    )

    macro_f1 = float(report['macro avg']['f1-score'])
    bal_acc = float(balanced_accuracy_score(y_true, y_preds) * 100)
    mcc = float(matthews_corrcoef(y_true, y_preds))

    cm = confusion_matrix(y_true, y_preds, labels=[0, 1, 2])
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12) * 100

    return {
        'accuracy': acc,
        'top2_accuracy': top2_acc,
        'log_loss': l_loss,
        'brier_score': brier,
        'ece_macro': ece_macro,
        'ece_per_class': ece_per_class,
        'rps': rps,
        'auc_macro': auc_macro,
        'auc_per_class': auc_per_class,
        'macro_f1': macro_f1,
        'balanced_accuracy': bal_acc,
        'mcc': mcc,
        'report': report,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_norm': cm_norm.tolist()
    }


def calcular_baseline_odds_mercado(y_true: np.ndarray, market_odds: np.ndarray) -> dict:
    """Calcula métricas da Baseline do Mercado usando probabilidades implícitas normalizadas."""
    return calcular_metricas_completas(y_true, market_odds)
