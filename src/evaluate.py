"""
Evaluation utilities: training curves, confusion matrix, per-class F1,
ROC curves, sensitivity/specificity, and a "full evaluation report" helper.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize


matplotlib.rcParams["figure.dpi"] = 100


def plot_training_curves(history: Dict, title: str = "",
                         save_path: Optional[str] = None):
    """Plot accuracy and loss curves side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_acc"]) + 1)
    best_ep = history.get("best_epoch", len(history["train_acc"]))

    ax1.plot(epochs, history["train_acc"], "b-o", markersize=3, label="Entrenamiento")
    ax1.plot(epochs, history["val_acc"], "r-o", markersize=3, label="Validación")
    ax1.axvline(x=best_ep, color="gray", linestyle="--",
                label=f"Mejor época ({best_ep})")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Curva de Accuracy — {title}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Entrenamiento")
    ax2.plot(epochs, history["val_loss"], "r-o", markersize=3, label="Validación")
    ax2.axvline(x=best_ep, color="gray", linestyle="--",
                label=f"Mejor época ({best_ep})")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Loss")
    ax2.set_title(f"Curva de Loss — {title}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def full_evaluation_report(model, loader, class_names: List[str],
                           device: str = "cpu") -> Dict:
    """
    Run inference on `loader`, return a dict of metrics and arrays.

    Keys:
        accuracy, weighted_f1, macro_f1, report_df, confusion_matrix,
        specificity, sensitivity, y_true, y_pred, y_prob
    """
    model = model.to(device)
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    labels_idx = list(range(len(class_names)))
    report = classification_report(y_true, y_pred,
                                   labels=labels_idx,
                                   target_names=class_names,
                                   output_dict=True,
                                   zero_division=0)
    report_df = pd.DataFrame(report).T
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

    # Per-class sensitivity (= recall) and specificity
    specificity, sensitivity = [], []
    total = cm.sum()
    for i in range(len(class_names)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity.append(spec)
        sensitivity.append(sens)

    return {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted",
                                      zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "report_df": report_df,
        "confusion_matrix": cm,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def plot_confusion_matrix(cm: np.ndarray,
                          class_names: List[str],
                          title: str = "Matriz de Confusión",
                          save_path: Optional[str] = None):
    """Plot row-normalised confusion matrix as a heatmap."""
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_norm = cm.astype(float) / row_sums

    n = len(class_names)
    side = max(8, n * 0.4)
    fig, ax = plt.subplots(figsize=(side, side * 0.9))
    sns.heatmap(cm_norm, annot=(n <= 20), fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                cbar=True, ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase real")
    ax.set_title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_per_class_f1(report_df: pd.DataFrame,
                      title: str = "F1 por clase (ordenado de peor a mejor)",
                      save_path: Optional[str] = None):
    """Horizontal bar chart of per-class F1 scores."""
    df = report_df.drop(index=["accuracy", "macro avg", "weighted avg"],
                       errors="ignore")
    df = df[["f1-score"]].sort_values("f1-score")
    fig, ax = plt.subplots(figsize=(8, max(6, len(df) * 0.25)))
    df["f1-score"].plot(kind="barh", ax=ax, color="steelblue",
                       edgecolor="white")
    ax.set_xlabel("F1-Score")
    ax.set_title(title)
    ax.axvline(x=df["f1-score"].mean(), color="red", linestyle="--",
              label=f"Media = {df['f1-score'].mean():.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_roc_curves(y_true: np.ndarray,
                    y_prob: np.ndarray,
                    class_names: List[str],
                    title: str = "Curvas ROC (one-vs-rest)",
                    max_classes: int = 10,
                    save_path: Optional[str] = None):
    """Plot ROC curves for up to `max_classes` classes (one-vs-rest)."""
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if n_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    classes_to_plot = list(range(min(n_classes, max_classes)))
    fig, ax = plt.subplots(figsize=(8, 7))
    for i in classes_to_plot:
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.5,
                label=f"{class_names[i]} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def find_top_confused_pairs(cm: np.ndarray,
                            class_names: List[str],
                            top_k: int = 5) -> pd.DataFrame:
    """Return the top-k most confused (true, predicted) pairs (off-diagonal)."""
    n = cm.shape[0]
    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if cm[i, j] > 0:
                pairs.append({
                    "true": class_names[i],
                    "predicted": class_names[j],
                    "count": int(cm[i, j]),
                })
    df = pd.DataFrame(pairs).sort_values("count", ascending=False).head(top_k)
    return df.reset_index(drop=True)
