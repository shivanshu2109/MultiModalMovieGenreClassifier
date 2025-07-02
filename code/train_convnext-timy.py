import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
import pandas as pd
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from torch.amp import GradScaler, autocast
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

class MoviePosterDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, labels, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.labels = labels
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        poster_path = os.path.join(self.image_dir, row['poster_path'])

        try:
            img = Image.open(poster_path).convert("RGB")
        except FileNotFoundError:
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, torch.tensor(label, dtype=torch.float32)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "movie_posters_new")
TRAIN_CSV = os.path.join(SCRIPT_DIR, "train_multimodal.csv")

GENRE_CLASSES_PATH = os.path.join(SCRIPT_DIR, "genre_classes.txt")
SAVE_PATH = os.path.join(SCRIPT_DIR, "genre_classifier_best.pth")

BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 5

with open(GENRE_CLASSES_PATH) as f:
    genre_classes = [line.strip() for line in f]
NUM_CLASSES = len(genre_classes)

df = pd.read_csv(TRAIN_CSV)
labels = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)

for idx, genres in enumerate(df["genres"]):
    for genre in genres.split(","):
        genre = genre.strip()
        if genre in genre_classes:
            labels[idx, genre_classes.index(genre)] = 1.0

from sklearn.model_selection import train_test_split
train_df, val_df, train_labels, val_labels = train_test_split(
    df, labels, test_size=0.2, random_state=42
)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_ds = MoviePosterDataset(train_df, train_labels, IMAGE_DIR, transform=train_transform)
val_ds   = MoviePosterDataset(val_df, val_labels, IMAGE_DIR, transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

genre_counts = np.sum(train_labels, axis=0)
class_weights = len(train_labels) / (NUM_CLASSES * genre_counts)
class_weights = torch.FloatTensor(class_weights).to(DEVICE)

model = timm.create_model("convnext_tiny", pretrained=True, num_classes=NUM_CLASSES)
model.head.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.head.fc.in_features, NUM_CLASSES)
)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
scaler = GradScaler(device="cuda")

best_val_loss = float("inf")
trigger_times = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        with autocast(device_type=DEVICE.type):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with autocast(device_type=DEVICE.type):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"✅ Best model saved to {SAVE_PATH}")
    else:
        trigger_times += 1
        if trigger_times >= PATIENCE:
            print("⏹️ Early stopping")
            break

print("✅ Training complete.")