import csv
import os
import shutil
import time
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.logging import setup_logger, MetricsLogger
from ..utils.checkpoints import save_checkpoint, load_checkpoint
from ..utils.metrics import compute_accuracy, compute_f1
from ..utils.plots import append_training_history, plot_training_curves
from .linear_probe import LinearProbeEvaluator


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.exp_dir = os.path.join(config["output_dir"], config["experiment_name"])
        self.ckpt_dir = os.path.join(self.exp_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.get("label_smoothing", 0.0),
        )

        self.optimizer = self._build_optimizer()

        self.scheduler = self._build_scheduler()

        self.use_amp = config.get("mixed_precision", False)
        amp_dtype_name = config.get("amp_dtype", "float16")
        self.amp_dtype = (
            torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp and self.amp_dtype == torch.float16,
        )

        self.logger = setup_logger("trainer", log_dir=self.exp_dir)
        self.metrics_logger = MetricsLogger(self.exp_dir)
        self.metrics_logger.save_config(config)

        self.start_epoch = 0
        self.best_val_acc = 0.0
        self.epoch_history = []
        self.probe_interval = config.get("probe_interval", 0)
        self.probe_train_loader = None
        self.probe_val_loader = None
        if self.probe_interval:
            from ..data.imagenet100 import get_imagenet100_probe_loaders

            self.probe_train_loader, self.probe_val_loader = (
                get_imagenet100_probe_loaders(
                    config,
                    train_fraction_per_class=config.get(
                        "probe_train_fraction_per_class",
                        0.2,
                    ),
                )
            )
            if hasattr(self.probe_train_loader.dataset, "indices"):
                self.config["probe_train_indices"] = list(
                    self.probe_train_loader.dataset.indices
                )
                self.metrics_logger.save_config(config)
        self.early_stopping_enabled = config.get("early_stopping", False)
        self.early_stopping_patience = config.get("early_stopping_patience", 6)
        self.early_stopping_min_delta = config.get("early_stopping_min_delta", 0.1)
        self.early_stopping_monitor = config.get(
            "early_stopping_monitor",
            "val_acc",
        )
        self.early_stopping_mode = config.get(
            "early_stopping_mode",
            "min" if "loss" in self.early_stopping_monitor else "max",
        )
        if self.early_stopping_mode not in {"min", "max"}:
            raise ValueError("early_stopping_mode must be 'min' or 'max'.")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1.")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta cannot be negative.")
        self.best_monitor_value = (
            float("inf") if self.early_stopping_mode == "min" else float("-inf")
        )
        self.early_stopping_reference_value = self.best_monitor_value
        self.epochs_without_meaningful_improvement = 0

        if config.get("resume_checkpoint"):
            self._resume(config["resume_checkpoint"])

    def _build_optimizer(self) -> torch.optim.Optimizer:
        opt_name = self.config.get("optimizer", "adam").lower()
        lr = self.config.get("lr", 1e-3)
        wd = self.config.get("weight_decay", 0.0)
        backbone_lr = self.config.get("backbone_lr")
        head_lr = self.config.get("head_lr")

        if backbone_lr is not None or head_lr is not None:
            if backbone_lr is None or head_lr is None:
                raise ValueError("backbone_lr and head_lr must be provided together.")

            backbone_params = [
                param
                for name, param in self.model.named_parameters()
                if param.requires_grad and not name.startswith("head.")
            ]
            head_params = [
                param
                for name, param in self.model.named_parameters()
                if param.requires_grad and name.startswith("head.")
            ]
            if not backbone_params:
                raise ValueError("No trainable backbone parameters found for backbone_lr.")
            if not head_params:
                raise ValueError("No trainable head parameters found for head_lr.")
            params = [
                {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
                {"params": head_params, "lr": head_lr, "name": "head"},
            ]
            lr = backbone_lr
        else:
            params = filter(lambda p: p.requires_grad, self.model.parameters())

        if opt_name == "sgd":
            return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
        elif opt_name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)
        elif opt_name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        sched_name = self.config.get("scheduler", "none").lower()
        epochs = self.config.get("epochs", 30)

        if sched_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        elif sched_name == "warmup_cosine":
            warmup_epochs = self.config.get("warmup_epochs", 5)
            if not 0 < warmup_epochs < epochs:
                raise ValueError("warmup_epochs must be between 1 and epochs - 1.")
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=self.config.get("warmup_start_factor", 0.1),
                total_iters=warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs - warmup_epochs,
                eta_min=self.config.get("min_lr", 0.0),
            )
            return torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        elif sched_name == "step":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif sched_name == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {sched_name}")

    def _resume(self, ckpt_path: str) -> None:
        self.logger.info(f"Resuming from {ckpt_path}")
        ckpt = load_checkpoint(
            ckpt_path,
            self.model,
            self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=str(self.device),
        )
        self.start_epoch = ckpt.get("epoch", 0) + 1
        metrics = ckpt.get("metrics", {})
        saved_monitor = metrics.get("early_stopping_monitor")
        self.best_val_acc = metrics.get("best_val_acc", metrics.get("val_acc", 0.0))
        self.best_monitor_value = metrics.get(
            "best_monitor_value",
            metrics.get(self.early_stopping_monitor, self.best_monitor_value),
        )
        self.early_stopping_reference_value = metrics.get(
            "early_stopping_reference_value",
            self.best_monitor_value,
        )
        self.epochs_without_meaningful_improvement = metrics.get(
            "epochs_without_meaningful_improvement",
            0,
        )
        metrics_path = os.path.join(self.exp_dir, "metrics.csv")
        if os.path.exists(metrics_path):
            with open(metrics_path, newline="") as file:
                self.epoch_history = list(csv.DictReader(file))
            historical_val_acc = [
                float(row["val_acc"])
                for row in self.epoch_history
                if row.get("val_acc") not in (None, "")
            ]
            if historical_val_acc:
                historical_best = max(historical_val_acc)
                self.best_val_acc = max(self.best_val_acc, historical_best)
            if (
                saved_monitor != self.early_stopping_monitor
                or "best_monitor_value" not in metrics
            ):
                self._restore_monitor_state_from_history()
                current_value = metrics.get(self.early_stopping_monitor)
                if (
                    current_value is not None
                    and abs(float(current_value) - self.best_monitor_value) < 1e-12
                ):
                    shutil.copy2(
                        ckpt_path,
                        os.path.join(self.ckpt_dir, "best.pt"),
                    )
                    self.logger.info(
                        f"Promoted resumed checkpoint to best.pt using "
                        f"{self.early_stopping_monitor}."
                    )
            current_val_acc = metrics.get("val_acc")
            if (
                current_val_acc is not None
                and abs(float(current_val_acc) - self.best_val_acc) < 1e-12
            ):
                shutil.copy2(
                    ckpt_path,
                    os.path.join(self.ckpt_dir, "best_val_acc.pt"),
                )
                self.logger.info("Promoted resumed checkpoint to best_val_acc.pt.")
        self.logger.info(
            f"Resumed at epoch {self.start_epoch}, best val_acc="
            f"{self.best_val_acc:.2f}, best {self.early_stopping_monitor}="
            f"{self.best_monitor_value:.4f}"
        )

    def _is_monitor_improvement(
        self,
        current: float,
        reference: float,
        min_delta: float = 0.0,
    ) -> bool:
        if self.early_stopping_mode == "min":
            return current < reference - min_delta
        return current > reference + min_delta

    def _restore_monitor_state_from_history(self) -> None:
        best = float("inf") if self.early_stopping_mode == "min" else float("-inf")
        reference = best
        without_improvement = 0
        for row in self.epoch_history:
            raw_value = row.get(self.early_stopping_monitor)
            if raw_value in (None, ""):
                continue
            current = float(raw_value)
            if self._is_monitor_improvement(current, best):
                best = current
            if self._is_monitor_improvement(
                current,
                reference,
                self.early_stopping_min_delta,
            ):
                reference = current
                without_improvement = 0
            else:
                without_improvement += 1
        self.best_monitor_value = best
        self.early_stopping_reference_value = reference
        self.epochs_without_meaningful_improvement = without_improvement

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_targets = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.amp.autocast(
                self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                logits = self.model(images)
                loss = self.criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch + 1}. "
                    "Stop the run and lower the learning rate."
                )

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            if self.config.get("gradient_clipping") is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["gradient_clipping"],
                    error_if_nonfinite=True,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += batch_size
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples
        metrics = {"train_loss": avg_loss, "train_acc": avg_acc}

        all_preds_np = np.concatenate(all_preds)
        all_targets_np = np.concatenate(all_targets)
        f1 = compute_f1(all_preds_np, all_targets_np)
        if f1 is not None:
            metrics["train_f1"] = f1
        metrics["train_source_images"] = total_samples
        metrics["vram_peak_mb"] = (
            torch.cuda.max_memory_allocated(self.device) / 1024**2
            if self.device.type == "cuda"
            else 0.0
        )
        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_targets = []

        pbar = tqdm(self.val_loader, desc="Validation", leave=False)
        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.amp.autocast(
                self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                logits = self.model(images)
                loss = self.criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite validation loss. The model weights are unstable; "
                    "do not resume from this checkpoint."
                )

            preds = logits.argmax(dim=1)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == targets).sum().item()
            total_samples += batch_size

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        avg_loss = total_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples

        all_preds_np = np.concatenate(all_preds)
        all_targets_np = np.concatenate(all_targets)
        f1 = compute_f1(all_preds_np, all_targets_np)

        metrics = {"val_loss": avg_loss, "val_acc": avg_acc}
        if f1 is not None:
            metrics["val_f1"] = f1
        return metrics

    def fit(self) -> None:
        epochs = self.config.get("epochs", 30)
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Model type: {self.config['model_type']}")
        self.logger.info(f"Freeze backbone: {self.config.get('freeze_backbone', False)}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Mixed precision: {self.use_amp}")
        self.logger.info(
            f"Regularization: label_smoothing="
            f"{self.config.get('label_smoothing', 0.0):g}, "
            f"head_dropout={self.config.get('head_dropout', 0.0):g}, "
            f"color_jitter={self.config.get('train_color_jitter', False)}"
        )
        if self.early_stopping_enabled:
            self.logger.info(
                "Early stopping: enabled "
                f"(monitor={self.early_stopping_monitor}, "
                f"mode={self.early_stopping_mode}, "
                f"patience={self.early_stopping_patience}, "
                f"min_delta={self.early_stopping_min_delta:.4f})"
            )
        else:
            self.logger.info("Early stopping: disabled")
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Parameters: {trainable:,} trainable / {total:,} total")

        stopped_early = False
        for epoch in range(self.start_epoch, epochs):
            t0 = time.time()

            train_metrics = self.train_one_epoch(epoch)

            val_metrics = self.validate()

            current_lr = self.optimizer.param_groups[0]["lr"]
            current_lrs = {
                group.get("name", f"group_{index}"): group["lr"]
                for index, group in enumerate(self.optimizer.param_groups)
            }
            if self.scheduler is not None:
                self.scheduler.step()

            is_best_val_acc = val_metrics["val_acc"] > self.best_val_acc
            if is_best_val_acc:
                self.best_val_acc = val_metrics["val_acc"]

            current_monitor_value = float(
                {**train_metrics, **val_metrics}[self.early_stopping_monitor]
            )
            is_best = self._is_monitor_improvement(
                current_monitor_value,
                self.best_monitor_value,
            )
            if is_best:
                self.best_monitor_value = current_monitor_value

            should_stop = False
            if self.early_stopping_enabled:
                if self._is_monitor_improvement(
                    current_monitor_value,
                    self.early_stopping_reference_value,
                    self.early_stopping_min_delta,
                ):
                    self.early_stopping_reference_value = current_monitor_value
                    self.epochs_without_meaningful_improvement = 0
                else:
                    self.epochs_without_meaningful_improvement += 1
                    should_stop = (
                        self.epochs_without_meaningful_improvement
                        >= self.early_stopping_patience
                    )

            probe_metrics = {
                "probe_loss": "",
                "probe_acc": "",
                "probe_f1": "",
                "feature_std_mean": "",
                "feature_std_min": "",
                "feature_norm_mean": "",
                "num_near_constant_dims": "",
                "effective_rank": "",
            }
            if self.probe_interval and (
                (epoch + 1) % self.probe_interval == 0 or epoch + 1 == epochs
            ):
                evaluator = LinearProbeEvaluator(
                    num_classes=self.config.get("num_classes", 30),
                    feature_dim=getattr(self.model.backbone, "feature_dim", 2048),
                    epochs=self.config.get("probe_epochs", 10),
                    lr=self.config.get("probe_lr", 1e-3),
                    weight_decay=self.config.get("probe_weight_decay", 1e-6),
                    batch_size=self.config.get("probe_batch_size", 256),
                    seed=self.config.get("probe_seed", 42),
                    near_constant_std_threshold=self.config.get(
                        "collapse_near_constant_std_threshold",
                        1e-4,
                    ),
                    compute_effective_rank=self.config.get(
                        "collapse_compute_effective_rank",
                        True,
                    ),
                    effective_rank_max_samples=self.config.get(
                        "collapse_effective_rank_max_samples",
                        2048,
                    ),
                )
                probe_metrics, _ = evaluator.run(
                    self.model.backbone,
                    self.probe_train_loader,
                    self.probe_val_loader,
                    self.device,
                )

            all_metrics = {**train_metrics, **val_metrics, **probe_metrics}
            checkpoint_metrics = {
                **all_metrics,
                "best_val_acc": self.best_val_acc,
                "best_monitor_value": self.best_monitor_value,
                "early_stopping_monitor": self.early_stopping_monitor,
                "early_stopping_reference_value": (
                    self.early_stopping_reference_value
                ),
                "epochs_without_meaningful_improvement": (
                    self.epochs_without_meaningful_improvement
                ),
            }
            save_checkpoint(
                self.model, self.optimizer, epoch,
                checkpoint_metrics, self.ckpt_dir, is_best=is_best,
                config=self.config,
                scheduler=self.scheduler,
                scaler=self.scaler,
            )
            if is_best_val_acc:
                shutil.copy2(
                    os.path.join(self.ckpt_dir, "last.pt"),
                    os.path.join(self.ckpt_dir, "best_val_acc.pt"),
                )

            elapsed = time.time() - t0
            log_row = {
                "epoch": epoch + 1,
                "lr": current_lr,
                **{
                    f"{name}_lr": value
                    for name, value in current_lrs.items()
                    if name in {"backbone", "head"}
                },
                **{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in all_metrics.items()},
                "best_val_acc": f"{self.best_val_acc:.2f}",
                "time_s": f"{elapsed:.1f}",
            }
            self.metrics_logger.log(log_row)
            self.epoch_history.append(log_row)

            f1_str = f"  val_f1={val_metrics['val_f1']:.4f}" if "val_f1" in val_metrics else ""
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"train_loss={train_metrics['train_loss']:.4f}  train_acc={train_metrics['train_acc']:.2f}% | "
                f"val_loss={val_metrics['val_loss']:.4f}  val_acc={val_metrics['val_acc']:.2f}%"
                f"{f1_str} | best={self.best_val_acc:.2f}% | {elapsed:.1f}s"
            )

            if should_stop:
                stopped_early = True
                self.logger.info(
                    f"Early stopping at epoch {epoch + 1}: "
                    f"{self.early_stopping_monitor} did not improve by more than "
                    f"{self.early_stopping_min_delta:.4f} "
                    f"for {self.early_stopping_patience} consecutive epochs."
                )
                break

        if stopped_early:
            best_path = os.path.join(self.ckpt_dir, "best.pt")
            load_checkpoint(best_path, self.model, device=str(self.device))
            self.logger.info(f"Restored best model weights from {best_path}")

        self.logger.info(f"Training complete. Best val accuracy: {self.best_val_acc:.2f}%")
        plot_training_curves(self.epoch_history, self.exp_dir)
        append_training_history(
            history_path=os.path.join(self.config["output_dir"], "training_history.csv"),
            config=self.config,
            epoch_history=self.epoch_history,
            exp_dir=self.exp_dir,
        )
        self.logger.info(f"Plots saved to {os.path.join(self.exp_dir, 'plots')}")
        self.logger.info(f"Run summary appended to {os.path.join(self.config['output_dir'], 'training_history.csv')}")
        last_path = os.path.join(self.ckpt_dir, "last.pt")
        if os.path.exists(last_path):
            shutil.copy2(last_path, os.path.join(self.ckpt_dir, "final.pt"))
