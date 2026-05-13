"""
Training loop with optional early stopping, cosine annealing LR, mixed-precision
training, and state-dict checkpointing of the best validation epoch.
"""

import time
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    use_amp = scaler is not None
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def train_model(model,
                train_loader,
                val_loader,
                epochs: int = 50,
                optimizer=None,
                scheduler=None,
                criterion=None,
                device: str = "cpu",
                patience: int = 10,
                save_path: Optional[str] = None,
                experiment_name: str = "experiment",
                verbose: bool = True,
                use_amp: Optional[bool] = None) -> Dict:
    """
    Train a model with early stopping.

    use_amp: if None (default), enables mixed precision automatically on CUDA.

    Returns:
        history dict with keys train_loss, val_loss, train_acc, val_acc, best_epoch
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if scheduler is None:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    model = model.to(device)

    if use_amp is None:
        use_amp = (str(device).startswith("cuda") and torch.cuda.is_available())
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer,
                                      criterion, device, scaler=scaler)
        va_loss, va_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        elapsed = time.time() - t0
        if verbose:
            print(f"[{experiment_name}] Epoch {epoch:3d}/{epochs} | "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
                  f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} | "
                  f"{elapsed:.1f}s")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_epoch = epoch
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(best was epoch {best_epoch})")
                break

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    return history
