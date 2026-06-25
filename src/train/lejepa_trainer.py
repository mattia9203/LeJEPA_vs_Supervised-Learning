"""Self-supervised LeJEPA trainer for ResNet-50."""

import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..network.lejepa_loss import LeJEPALoss
from ..utils.logging import setup_logger
from ..utils.plots import plot_lejepa_curves
from .linear_probe import LinearProbeEvaluator


class LeJEPATrainer:
    """Train LeJEPA with 2 global and 6 local views.

    SIGReg is evaluated independently on every micro-batch, matching the
    official [views, batch, dim] API. Gradients are accumulated over 128 source
    images before each
    optimizer step. This avoids retaining all effective-batch computation
    graphs in 8 GB VRAM.
    """

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        probe_train_loader: DataLoader,
        probe_val_loader: DataLoader,
        config: dict,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.probe_train_loader = probe_train_loader
        self.probe_val_loader = probe_val_loader
        self.config = config
        self.device = torch.device(
            config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.exp_dir = Path(config["output_dir"]) / config["experiment_name"]
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger("lejepa_cnn", str(self.exp_dir))
        with open(self.exp_dir / "config.json", "w") as file:
            json.dump(config, file, indent=2, default=str)

        self.loss_fn = LeJEPALoss(
            sigreg_weight=config.get("sigreg_weight", 0.02),
            num_slices=config.get("sigreg_num_slices", 1024),
            num_points=config.get("sigreg_num_points", 17),
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("lr", 5e-4),
            weight_decay=config.get("weight_decay", 5e-4),
        )
        self.accumulation_steps = config.get("gradient_accumulation_steps")
        if self.accumulation_steps is None:
            micro = config.get("micro_batch_size", 32)
            effective = config.get("effective_batch_size", 128)
            if effective % micro:
                raise ValueError("effective_batch_size must be divisible by micro_batch_size.")
            self.accumulation_steps = effective // micro

        optimizer_steps_per_epoch = math.ceil(
            len(self.train_loader) / self.accumulation_steps
        )
        total_steps = max(1, config.get("epochs", 90) * optimizer_steps_per_epoch)
        warmup_steps = config.get("warmup_epochs", 5) * optimizer_steps_per_epoch
        warmup_steps = min(warmup_steps, max(0, total_steps - 1))
        if warmup_steps:
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,
                total_iters=warmup_steps,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, total_steps - warmup_steps),
                eta_min=config.get("min_lr", config.get("lr", 5e-4) / 1000),
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps],
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=config.get("min_lr", config.get("lr", 5e-4) / 1000),
            )

        self.use_amp = config.get("mixed_precision", True) and self.device.type == "cuda"
        dtype_name = config.get("amp_dtype", "float16")
        self.amp_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp and self.amp_dtype == torch.float16,
        )
        self.start_epoch = 0
        self.global_step = 0
        self.best_probe_acc = float("-inf")
        self.min_ssl_loss = float("inf")
        self.history = []
        if config.get("resume_checkpoint"):
            self._resume(config["resume_checkpoint"])

    def _checkpoint_state(
        self,
        epoch: int,
        metrics: Dict[str, float],
        probe_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> dict:
        return {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "backbone_state_dict": self.model.backbone.state_dict(),
            "projector_state_dict": self.model.projector.state_dict(),
            "probe_state_dict": probe_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "metrics": metrics,
            "best_probe_acc": self.best_probe_acc,
            "min_ssl_loss": self.min_ssl_loss,
            "config": self.config,
        }

    def _save(
        self,
        filename: str,
        epoch: int,
        metrics: Dict[str, float],
        probe_state=None,
    ) -> None:
        torch.save(
            self._checkpoint_state(epoch, metrics, probe_state),
            self.ckpt_dir / filename,
        )

    def _resume(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
        self.start_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint.get("global_step", 0)
        self.best_probe_acc = checkpoint.get("best_probe_acc", float("-inf"))
        self.min_ssl_loss = checkpoint.get("min_ssl_loss", float("inf"))

    def _run_probe(self):
        evaluator = LinearProbeEvaluator(
            num_classes=self.config.get("num_classes", 30),
            feature_dim=self.model.backbone.feature_dim,
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
        return evaluator.run(
            self.model.backbone,
            self.probe_train_loader,
            self.probe_val_loader,
            self.device,
        )

    def _log_row(self, row: dict) -> None:
        path = self.exp_dir / "metrics.csv"
        write_header = not path.exists()
        fieldnames = list(row.keys())
        if not write_header:
            with open(path, newline="") as file:
                reader = csv.DictReader(file)
                old_rows = list(reader)
                old_fieldnames = reader.fieldnames or []
            if old_fieldnames != fieldnames:
                merged_fieldnames = list(old_fieldnames)
                merged_fieldnames.extend(
                    field for field in fieldnames if field not in merged_fieldnames
                )
                with open(path, "w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=merged_fieldnames)
                    writer.writeheader()
                    writer.writerows(old_rows)
                fieldnames = merged_fieldnames
        with open(path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        totals = {"total_loss": 0.0, "invariance_loss": 0.0, "sigreg_loss": 0.0}
        source_images = 0
        num_batches = len(self.train_loader)
        self.optimizer.zero_grad(set_to_none=True)
        group_size = self.accumulation_steps

        progress = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [LeJEPA]", leave=False)
        for batch_index, batch in enumerate(progress):
            group_position = batch_index % self.accumulation_steps
            if group_position == 0:
                group_size = min(self.accumulation_steps, num_batches - batch_index)

            global_views = [
                view.to(self.device, non_blocking=True) for view in batch["global_views"]
            ]
            local_views = [
                view.to(self.device, non_blocking=True) for view in batch["local_views"]
            ]
            with torch.amp.autocast(
                self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model.forward_multicrop(global_views, local_views)
                losses = self.loss_fn(outputs["projections"])
                scaled_loss = losses["total_loss"] / group_size

            self.scaler.scale(scaled_loss).backward()
            is_group_end = (
                group_position + 1 == group_size or batch_index + 1 == num_batches
            )
            if is_group_end:
                if self.config.get("gradient_clipping") is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["gradient_clipping"],
                        error_if_nonfinite=True,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1

            batch_size = batch["labels"].shape[0]
            source_images += batch_size
            for key in totals:
                totals[key] += losses[key].detach().item() * batch_size
            progress.set_postfix(loss=f"{losses['total_loss'].item():.4f}")

        metrics = {key: value / source_images for key, value in totals.items()}
        metrics.update(
            {
                "source_images": source_images,
                "global_views": source_images * self.config.get("num_global_views", 2),
                "local_views": source_images * self.config.get("num_local_views", 6),
                "lr": self.optimizer.param_groups[0]["lr"],
                "vram_peak_mb": (
                    torch.cuda.max_memory_allocated(self.device) / 1024**2
                    if self.device.type == "cuda"
                    else 0.0
                ),
            }
        )
        return metrics

    def fit(self) -> None:
        epochs = self.config.get("epochs", 90)
        probe_interval = self.config.get("probe_interval", 5)
        self.logger.info(
            f"Training LeJEPA CNN for {epochs} epochs; "
            f"micro_batch={self.config.get('micro_batch_size', 32)}, "
            f"effective_batch={self.config.get('effective_batch_size', 128)}, "
            f"views={self.config.get('num_global_views', 2)}+"
            f"{self.config.get('num_local_views', 6)}"
        )
        for epoch in range(self.start_epoch, epochs):
            started = time.time()
            metrics = self.train_epoch(epoch)
            probe_state = None
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
            if (epoch + 1) % probe_interval == 0 or epoch + 1 == epochs:
                probe_metrics, probe_state = self._run_probe()
                if probe_metrics["probe_acc"] > self.best_probe_acc:
                    self.best_probe_acc = probe_metrics["probe_acc"]
                    self._save(
                        "best_probe_accuracy.pt",
                        epoch,
                        {**metrics, **probe_metrics},
                        probe_state,
                    )

            if metrics["total_loss"] < self.min_ssl_loss:
                self.min_ssl_loss = metrics["total_loss"]
                self._save("min_ssl_loss.pt", epoch, metrics, probe_state)

            row = {
                "epoch": epoch + 1,
                **metrics,
                **probe_metrics,
                "best_probe_acc": (
                    self.best_probe_acc if self.best_probe_acc != float("-inf") else ""
                ),
                "min_ssl_loss": self.min_ssl_loss,
                "time_s": time.time() - started,
            }
            self.history.append(row)
            self._log_row(row)
            self._save("last.pt", epoch, row, probe_state)
            checkpoint_interval = self.config.get("checkpoint_interval_epochs", 5)
            if checkpoint_interval and (epoch + 1) % checkpoint_interval == 0:
                self._save(
                    f"epoch_{epoch + 1:04d}.pt",
                    epoch,
                    row,
                    probe_state,
                )
            log_message = (
                f"Epoch {epoch + 1}/{epochs} | ssl={metrics['total_loss']:.4f} "
                f"inv={metrics['invariance_loss']:.4f} "
                f"sigreg={metrics['sigreg_loss']:.4f} "
                f"lr={metrics['lr']:.3e} "
                f"vram={metrics['vram_peak_mb']:.0f} MB"
            )
            if probe_metrics["probe_acc"] != "":
                log_message += (
                    f" | probe_acc={probe_metrics['probe_acc']:.2f}% "
                    f"probe_f1={probe_metrics['probe_f1']:.4f} "
                    f"std_mean={probe_metrics['feature_std_mean']:.6f} "
                    f"std_min={probe_metrics['feature_std_min']:.6f} "
                    f"norm_mean={probe_metrics['feature_norm_mean']:.4f} "
                    f"near_const={probe_metrics['num_near_constant_dims']}/"
                    f"{self.model.backbone.feature_dim} "
                    f"effective_rank={probe_metrics['effective_rank']:.2f}"
                )
            self.logger.info(log_message)

        self._save("final.pt", epochs - 1, self.history[-1])
        plot_lejepa_curves(self.history, str(self.exp_dir))
        paired = [
            (float(row["total_loss"]), float(row["probe_acc"]))
            for row in self.history
            if row.get("probe_acc") not in ("", None)
        ]
        correlation = None
        if len(paired) >= 2:
            ssl_values = np.asarray([item[0] for item in paired])
            probe_values = np.asarray([item[1] for item in paired])
            ssl_ranks = np.argsort(np.argsort(ssl_values))
            probe_ranks = np.argsort(np.argsort(probe_values))
            correlation = float(np.corrcoef(ssl_ranks, probe_ranks)[0, 1])
        with open(self.exp_dir / "ssl_probe_correlation.json", "w") as file:
            json.dump(
                {
                    "spearman_ssl_loss_vs_probe_accuracy": correlation,
                    "num_probe_points": len(paired),
                    "note": (
                        "A useful LeJEPA loss should correlate negatively with "
                        "downstream probe accuracy."
                    ),
                },
                file,
                indent=2,
            )
