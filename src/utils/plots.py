"""Plot training curves and append run summaries."""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _best_index(values: Sequence[float], mode: str) -> int:
    if mode == "min":
        return min(range(len(values)), key=lambda idx: values[idx])
    return max(range(len(values)), key=lambda idx: values[idx])


def plot_training_curves(history: List[Dict[str, Any]], output_dir: str) -> None:
    """Save loss and metric curves with best points annotated."""
    if not history:
        return

    os.environ.setdefault("MPLCONFIGDIR", str(Path(output_dir) / ".matplotlib_cache"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def plot_group(
        filename: str,
        title: str,
        series: Iterable[Tuple[str, str, str]],
        ylabel: str,
        mode: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        plotted = False

        epochs = [int(row["epoch"]) for row in history]
        for key, label, color in series:
            values = [_as_float(row.get(key)) for row in history]
            if any(value is None for value in values):
                continue
            numeric_values = [float(value) for value in values if value is not None]
            if not numeric_values:
                continue

            line, = ax.plot(epochs, numeric_values, marker="o", label=label, color=color)
            best_idx = _best_index(numeric_values, mode=mode)
            best_epoch = epochs[best_idx]
            best_value = numeric_values[best_idx]
            ax.scatter(
                [best_epoch],
                [best_value],
                s=70,
                color=line.get_color(),
                edgecolor="black",
                linewidth=0.8,
                zorder=5,
            )
            ax.annotate(
                f"{best_value:.4f}",
                xy=(best_epoch, best_value),
                xytext=(6, 8),
                textcoords="offset points",
                color=line.get_color(),
                fontsize=9,
                weight="bold",
            )
            plotted = True

        if not plotted:
            plt.close(fig)
            return

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=160)
        plt.close(fig)

    plot_group(
        filename="loss_curves.png",
        title="Training and Validation Loss",
        series=(
            ("train_loss", "Train loss", "tab:blue"),
            ("val_loss", "Validation loss", "tab:orange"),
        ),
        ylabel="Loss",
        mode="min",
    )
    plot_group(
        filename="accuracy_curves.png",
        title="Training and Validation Accuracy",
        series=(
            ("train_acc", "Train accuracy", "tab:green"),
            ("val_acc", "Validation accuracy", "tab:red"),
        ),
        ylabel="Accuracy (%)",
        mode="max",
    )
    plot_group(
        filename="f1_curves.png",
        title="Training and Validation Macro F1",
        series=(
            ("train_f1", "Train macro F1", "tab:purple"),
            ("val_f1", "Validation macro F1", "tab:brown"),
        ),
        ylabel="Macro F1",
        mode="max",
    )


def append_training_history(
    history_path: str,
    config: Dict[str, Any],
    epoch_history: List[Dict[str, Any]],
    exp_dir: str,
) -> None:
    """Append one run summary row to the global training history CSV."""
    if not epoch_history:
        return

    Path(os.path.dirname(history_path)).mkdir(parents=True, exist_ok=True)
    final_row = epoch_history[-1]

    best_val_acc_row = max(epoch_history, key=lambda row: _as_float(row.get("val_acc")) or float("-inf"))
    best_val_loss_row = min(epoch_history, key=lambda row: _as_float(row.get("val_loss")) or float("inf"))
    best_val_f1_row = max(epoch_history, key=lambda row: _as_float(row.get("val_f1")) or float("-inf"))

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": config.get("experiment_name"),
        "model_type": config.get("model_type"),
        "model_name": config.get("model_name"),
        "freeze_backbone": config.get("freeze_backbone"),
        "trainable_last_blocks": config.get("trainable_last_blocks"),
        "num_classes": config.get("num_classes"),
        "batch_size": config.get("batch_size"),
        "epochs_requested": config.get("epochs"),
        "epochs_completed": final_row.get("epoch"),
        "optimizer": config.get("optimizer"),
        "scheduler": config.get("scheduler"),
        "initial_lr": config.get("lr"),
        "backbone_lr": config.get("backbone_lr"),
        "head_lr": config.get("head_lr"),
        "weight_decay": config.get("weight_decay"),
        "label_smoothing": config.get("label_smoothing"),
        "head_dropout": config.get("head_dropout"),
        "train_color_jitter": config.get("train_color_jitter"),
        "init_checkpoint": config.get("init_checkpoint"),
        "best_val_acc": best_val_acc_row.get("val_acc"),
        "best_val_acc_epoch": best_val_acc_row.get("epoch"),
        "best_val_loss": best_val_loss_row.get("val_loss"),
        "best_val_loss_epoch": best_val_loss_row.get("epoch"),
        "best_val_f1": best_val_f1_row.get("val_f1"),
        "best_val_f1_epoch": best_val_f1_row.get("epoch"),
        "final_train_loss": final_row.get("train_loss"),
        "final_train_acc": final_row.get("train_acc"),
        "final_train_f1": final_row.get("train_f1"),
        "final_val_loss": final_row.get("val_loss"),
        "final_val_acc": final_row.get("val_acc"),
        "final_val_f1": final_row.get("val_f1"),
        "output_dir": exp_dir,
    }

    fieldnames = list(summary.keys())
    write_header = not os.path.exists(history_path)
    if not write_header:
        with open(history_path, newline="") as f:
            reader = csv.DictReader(f)
            old_rows = list(reader)
            old_fieldnames = reader.fieldnames or []

        if old_fieldnames != fieldnames:
            merged_fieldnames = list(old_fieldnames)
            merged_fieldnames.extend(
                field for field in fieldnames if field not in merged_fieldnames
            )
            with open(history_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=merged_fieldnames)
                writer.writeheader()
                writer.writerows(old_rows)
            fieldnames = merged_fieldnames

    with open(history_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(summary)
