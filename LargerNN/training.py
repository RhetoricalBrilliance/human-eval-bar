
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from model import TabResNetModel
from dataloader import ParquetChessDataset  # unchanged API

def train(data_path, epochs=50, lr=2e-3, batch_size=256, val_frac=0.2, seed=42, weight_decay=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load full dataset once
    full_ds = ParquetChessDataset(data_path)
    input_dim = full_ds.X.shape[1]

    # split into train/val
    n_total = len(full_ds)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # model/opt/loss
    model = TabResNetModel(input_dim=input_dim, num_classes=3).to(device)
    # label smoothing can help on noisy labels / class overlap
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # cosine schedule is a robust default for AdamW
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def run_epoch(loader, train_mode: bool):
        model.train() if train_mode else model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.set_grad_enabled(train_mode):
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = criterion(logits, y)
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                total_loss += loss.item() * X.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return total_loss / max(total,1), correct / max(total,1)

    best_val = float("inf")
    best_path = "evalbar_model_large.pt"

    for epoch in range(1, epochs+1):
        tr_loss, tr_acc = run_epoch(train_loader, train_mode=True)
        va_loss, va_acc = run_epoch(val_loader,   train_mode=False)
        scheduler.step()
        print(f"Epoch {epoch:02d}/{epochs} | "
              f"Train: loss {tr_loss:.4f}, acc {tr_acc:.4f} | "
              f"Val: loss {va_loss:.4f}, acc {va_acc:.4f}")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)

    print(f"Best model (by val loss) saved to {best_path}")

if __name__ == "__main__":
    train("data/normalized/normalized.parquet", epochs=35)
