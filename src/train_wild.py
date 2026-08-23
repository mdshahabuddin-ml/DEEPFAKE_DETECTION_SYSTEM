"""
train_wild.py — Train on Wild Deepfake Dataset
Usage: python src/train_wild.py --split_csv results/splits/wild_deepfake_split.csv
"""
import os, sys, argparse, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import roc_auc_score, accuracy_score
from pathlib import Path
import csv

sys.path.insert(0, "src")

# ── Dataset ───────────────────────────────────────────────────────────────────
class WildDeepfakeDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, int(label)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            img = torch.zeros(3, 299, 299)
            return img, int(label)

def load_split_csv(csv_path):
    train, val, test = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = (row["path"], int(row["label"]))
            if row["split"] == "train": train.append(sample)
            elif row["split"] == "val": val.append(sample)
            else: test.append(sample)
    return train, val, test

def get_transforms(img_size=299, augment=True):
    if augment:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return train_tf, val_tf

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(model_name="efficientnet_b4", pretrained=True, dropout=0.5):
    model = timm.create_model(model_name, pretrained=pretrained, drop_rate=dropout)
    n_features = model.get_classifier().in_features
    model.reset_classifier(0)
    model.head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(n_features, 1)
    )
    return model

# ── Train/Eval ────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, all_labels, all_probs = 0, [], []
    for i, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        if hasattr(model, 'head'):
            logits = model.head(outputs).squeeze(1)
        else:
            logits = outputs.squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        if i % 10 == 0:
            print(f"  Batch [{i}/{len(loader)}] Loss: {loss.item():.4f}")
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    return total_loss / len(loader), auc

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_labels, all_probs = 0, [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.float().to(device)
            outputs = model(imgs)
            if hasattr(model, 'head'):
                logits = model.head(outputs).squeeze(1)
            else:
                logits = outputs.squeeze(1)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_probs])
    return total_loss / len(loader), auc, acc

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_csv", type=str, default="results/splits/wild_deepfake_split.csv")
    parser.add_argument("--model", type=str, default="efficientnet_b4")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=299)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print(f"Loading splits from {args.split_csv}...")
    train_data, val_data, test_data = load_split_csv(args.split_csv)
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    train_tf, val_tf = get_transforms(args.img_size)
    train_loader = DataLoader(WildDeepfakeDataset(train_data, train_tf), 
                              batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=device.type=="cuda")
    val_loader = DataLoader(WildDeepfakeDataset(val_data, val_tf), 
                            batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers)
    test_loader = DataLoader(WildDeepfakeDataset(test_data, val_tf), 
                             batch_size=args.batch_size, shuffle=False, 
                             num_workers=args.num_workers)

    # Build model
    print(f"Building {args.model} model...")
    model = timm.create_model(args.model, pretrained=True, num_classes=1, drop_rate=args.dropout)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Count class weights
    fake_count = sum(1 for _, l in train_data if l == 1)
    real_count = sum(1 for _, l in train_data if l == 0)
    pos_weight = torch.tensor([real_count / max(fake_count, 1)]).to(device)
    print(f"Real: {real_count} | Fake: {fake_count} | Pos weight: {pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    best_auc = 0.0
    best_path = os.path.join(args.save_dir, f"best_{args.model}_wild.pth")

    print("\n" + "="*50)
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"epoch": epoch, "model_name": args.model,
                        "model_state_dict": model.state_dict(),
                        "val_auc": val_auc, "args": vars(args)}, best_path)
            print(f"  Best model saved! Val AUC: {val_auc:.4f}")

    # Final test evaluation
    print("\n" + "="*50)
    print("FINAL TEST EVALUATION")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_auc, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test AUC:      {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss:     {test_loss:.4f}")

    # Save metrics
    os.makedirs("results", exist_ok=True)
    metrics_path = f"results/metrics_{args.model}_wild.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: Wild Deepfake\n")
        f.write(f"Best Val AUC: {best_auc:.4f}\n")
        f.write(f"Test AUC: {test_auc:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
    print(f"\nMetrics saved to {metrics_path}")
    print(f"Best model saved to {best_path}")

if __name__ == "__main__":
    main()     