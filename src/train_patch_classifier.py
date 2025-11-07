# src/train_patch_classifier.py
import argparse, json, time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def get_loaders(data_dir, img_size=96, batch_size=64):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(Path(data_dir) / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(Path(data_dir) / "val",   transform=val_tf)

    # --- Balanced sampling for class imbalance ---
    labels = [y for _, y in train_ds.samples]  # integer class IDs
    counts = Counter(labels)                   # e.g., {0:49, 1:11, 2:124}
    # inverse-frequency weights per sample
    class_w = {c: 1.0 / max(1, counts[c]) for c in counts}
    sample_w = torch.DoubleTensor([class_w[y] for _, y in train_ds.samples])
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, train_ds.classes, counts

# def build_model(num_classes):
#     # No internet needed: start without pretrained weights
#     m = models.mobilenet_v2(weights=None)
#     # DO NOT freeze — we’re training from scratch on your small dataset
#     in_f = m.classifier[1].in_features
#     m.classifier[1] = nn.Linear(in_f, num_classes)
#     return m

def build_model(num_classes):
    from torchvision import models
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    for p in m.features.parameters():
        p.requires_grad = False
    for p in m.features[-1].parameters():  # UNFREEZE LAST BLOCK
        p.requires_grad = True
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_f, num_classes)
    return m


def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    tot_loss, tot_correct, tot = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * x.size(0)
        tot_correct += (logits.argmax(1) == y).sum().item()
        tot += x.size(0)
    return tot_loss / tot, tot_correct / tot

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    tot_loss, tot_correct, tot = 0.0, 0, 0
    all_y, all_p = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        tot_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        tot_correct += (preds == y).sum().item()
        tot += x.size(0)
        all_y.append(y.cpu().numpy())
        all_p.append(preds.cpu().numpy())
    return tot_loss / tot, tot_correct / tot, np.concatenate(all_y), np.concatenate(all_p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/patches_split")
    ap.add_argument("--out",  default="models/patch_clf_mobilenet.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img_size", type=int, default=96)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print("Device:", device)

    train_loader, val_loader, classes, counts = get_loaders(args.data, args.img_size, args.batch_size)
    print("Class names:", classes)
    print("Train counts:", counts)

    model = build_model(num_classes=len(classes)).to(device)

    # --- Class-weighted loss (helps rare 'fauna' class) ---
    # weights indexed by class ID order in 'classes'
    total = sum(counts.values())
    weights = [total / max(1, counts.get(i, 1)) for i in range(len(classes))]
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_acc, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
        va_loss, va_acc, y_true, y_pred = evaluate(model, val_loader, device, criterion)
        print(f"Epoch {epoch:02d} | train {tr_loss:.4f}/{tr_acc:.3f} | val {va_loss:.4f}/{va_acc:.3f} | {time.time()-t0:.1f}s")
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = model.state_dict().copy()

    # save best
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "classes": classes}, args.out)
    print("Saved:", args.out)

    # final report
    model.load_state_dict(best_state)
    _, _, y_true, y_pred = evaluate(model, val_loader, device, criterion)
    print("\nConfusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nReport:\n", classification_report(y_true, y_pred, target_names=classes, digits=3))

    # label map for inference
    with open(Path(args.out).with_suffix(".labels.json"), "w") as f:
        json.dump({"classes": classes}, f, indent=2)

if __name__ == "__main__":
    main()
