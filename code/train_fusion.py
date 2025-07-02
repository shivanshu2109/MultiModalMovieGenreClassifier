
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm
import timm
from transformers import DistilBertModel
from multimodal_csv_pipeline import get_loaders

NUM_EPOCHS    = 10
LR            = 2e-4
WD            = 1e-5
FREEZE_TEXT   = True
FREEZE_IMAGE  = True
PATIENCE      = 3
ACCUM_STEPS   = 2

BATCH_SIZE      = 64
NUM_WORKERS     = 0
PIN_MEMORY      = True
DROP_LAST       = True

class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.text_enc = DistilBertModel.from_pretrained("distilbert-base-uncased")
        if FREEZE_TEXT:
            for p in self.text_enc.parameters(): p.requires_grad = False

        self.img_enc = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
        if FREEZE_IMAGE:
            for p in self.img_enc.parameters(): p.requires_grad = False

        dim_t = self.text_enc.config.hidden_size
        dim_i = self.img_enc.num_features
        fuse  = dim_t + dim_i

        self.head = nn.Sequential(
            nn.Linear(fuse, fuse // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fuse // 2, num_classes)
        )

    def forward(self, input_ids, attention_mask, image):
        txt = self.text_enc(input_ids=input_ids, attention_mask=attention_mask)
        tfeat = txt.last_hidden_state[:, 0]
        if hasattr(self.img_enc, "forward_features"):
            feats = self.img_enc.forward_features(image)
            if hasattr(self.img_enc, "forward_head") and self.img_enc.num_classes == 0:
                img_feat = self.img_enc.forward_head(feats)
            else:
                img_feat = feats
        else:
            img_feat = self.img_enc(image)
        return self.head(torch.cat([tfeat, img_feat], dim=1))

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_loaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=DROP_LAST
    )
    print(f"Train loaded: {len(train_loader.dataset):,} valid rows")
    print(f"Val loaded:   {len(val_loader.dataset):,} valid rows")

    num_classes = train_loader.dataset.num_classes
    model       = FusionModel(num_classes).to(device)
    optimizer   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion   = nn.BCEWithLogitsLoss()
    scaler      = GradScaler()

    best_f1 = 0.0
    no_imp  = 0

    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        start = time.time()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False), 1):
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            images         = batch["image"].to(device, non_blocking=True)
            labels         = batch["labels"].to(device, non_blocking=True)

            with autocast(device_type=device.type):
                logits = model(input_ids, attention_mask, images)
                loss   = criterion(logits, labels) / ACCUM_STEPS

            scaler.scale(loss).backward()
            running_loss += loss.item() * labels.size(0) * ACCUM_STEPS

            if step % ACCUM_STEPS == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        train_time = (time.time() - start) / 60
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_pred, all_lbl = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Valid]", leave=False):
                input_ids      = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                images         = batch["image"].to(device, non_blocking=True)
                labels         = batch["labels"].to(device, non_blocking=True)

                with autocast(device_type=device.type):
                    logits = model(input_ids, attention_mask, images)
                    loss   = criterion(logits, labels)

                val_loss += loss.item() * labels.size(0)
                all_pred.append(torch.sigmoid(logits).cpu())
                all_lbl.append(labels.cpu())

        val_loss /= len(val_loader.dataset)
        preds = torch.cat(all_pred); lbls = torch.cat(all_lbl)
        preds_bin = (preds > 0.5).float()
        tp = (preds_bin * lbls).sum().item()
        fp = (preds_bin * (1-lbls)).sum().item()
        fn = ((1-preds_bin) * lbls).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        print(f"\nEpoch {epoch} Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  F1: {f1:.4f} (P={precision:.4f}, R={recall:.4f})  Time: {train_time:.1f}m")

        if f1 > best_f1:
            best_f1 = f1
            no_imp = 0
            torch.save(model.state_dict(), "fusion_best.pth")
            print(f"🔖 New best model saved (F1={f1:.4f})")
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"⏹ Early stopping (no improvement in {PATIENCE} epochs)")
                break

    print("✅ Training complete!")

def build_model(num_classes):
    return FusionModel(num_classes)

if __name__ == "__main__":
    train()