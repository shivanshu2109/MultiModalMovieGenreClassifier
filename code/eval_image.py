import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import timm
from preprocess_data import MoviePosterDataset, preprocess_data
import torchvision.transforms.v2 as transforms
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

GENRE_CLASSES_PATH = os.path.join(DATA_DIR, "genre_classes_new.txt")
try:
    with open(GENRE_CLASSES_PATH, "r") as f:
        label_classes = [line.strip() for line in f]
except FileNotFoundError:
    logger.error(f"Error: Could not find 'genre_classes.txt' at {GENRE_CLASSES_PATH}")
    sys.exit(1)
logger.info(f"Loaded {len(label_classes)} genre classes")

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate_model():
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "poster_genres_filtered_all_titletypes_fixed.csv"))
    except FileNotFoundError:
        logger.error(f"Error: Could not find 'poster_genres_filtered_all_titletypes_fixed.csv' at {DATA_DIR}")
        sys.exit(1)

    labels = np.zeros((len(df), len(label_classes)), dtype=np.float32)
    for idx, genres in enumerate(df['genres']):
        for genre in genres.split(','):
            genre = genre.strip()
            if genre in label_classes:
                labels[idx, label_classes.index(genre)] = 1.0
            else:
                logger.warning(f"Genre '{genre}' not in label_classes")

    train_size = int(0.8 * len(df))
    val_size = len(df) - train_size
    train_subset, val_subset = random_split(df, [train_size, val_size])
    val_indices = val_subset.indices

    val_dataset = MoviePosterDataset(df.iloc[val_indices], labels[val_indices], transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    MODEL_PATH = os.path.join(DATA_DIR, "genre_classifier_best.pth")
    try:
        model = timm.create_model('convnext_tiny', pretrained=False, num_classes=len(label_classes))
        model.head.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.head.fc.in_features, len(label_classes))
        )
        state_dict = torch.load(MODEL_PATH)
        expected_classes = len(label_classes)
        model_classes = state_dict['head.fc.1.weight'].shape[0]
        if model_classes != expected_classes:
            logger.error(f"Model has {model_classes} output classes, but expected {expected_classes} based on genre_classes.txt")
            sys.exit(1)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        logger.error(f"Error: Could not find 'genre_classifier.pth' at {MODEL_PATH}")
        sys.exit(1)
    model = model.to(DEVICE)
    model.eval()
    logger.info(f"Loaded model and moved to device: {DEVICE}")

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating Image-Only", file=sys.stdout):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            sigmoid_outputs = torch.sigmoid(outputs)
            all_probs.append(sigmoid_outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    logger.info(f"Number of failed image loads: {val_dataset.get_failed_images_count()} out of {val_dataset.get_total_images_processed()}")

    print("\nSample sigmoid outputs (first 5 samples):", flush=True)
    for i in range(min(5, len(all_probs))):
        print(f"Sample {i}: {all_probs[i]}", flush=True)

    thresholds = np.arange(0.1, 0.6, 0.1)
    best_f1 = 0
    best_threshold = 0
    best_metrics = None

    for threshold in thresholds:
        preds = (all_probs > threshold).astype(np.float32)
        precision, recall, f1, support = precision_recall_fscore_support(all_labels, preds, average=None, zero_division=0)
        macro_f1 = np.mean(f1)
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        logger.info(f"Threshold {threshold:.1f}: Macro F1={macro_f1:.4f}, Precision={macro_precision:.4f}, Recall={macro_recall:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_threshold = threshold
            best_metrics = (precision, recall, f1, support, preds)

    precision, recall, f1, support, best_preds = best_metrics

    per_class_accuracy = []
    for i in range(len(label_classes)):
        correct = np.sum(all_labels[:, i] == best_preds[:, i])
        total = len(all_labels)
        acc = correct / total
        per_class_accuracy.append(acc)
    macro_accuracy = np.mean(per_class_accuracy)

    print("\nImage-Only Model Metrics (Best Threshold):", flush=True)
    print(f"Best Threshold: {best_threshold:.1f}", flush=True)
    print(f"Macro-averaged Accuracy: {macro_accuracy:.4f}", flush=True)
    print("\nPer-class metrics:", flush=True)
    for i, genre in enumerate(label_classes):
        print(f"{genre}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Accuracy={per_class_accuracy[i]:.4f}, Support={support[i]}", flush=True)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    print(f"\nMacro-averaged metrics: Precision={macro_precision:.4f}, Recall={macro_recall:.4f}, F1={macro_f1:.4f}", flush=True)

    return macro_f1

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    print(f"Number of genres: {len(label_classes)}", flush=True)
    print(f"Genres: {label_classes}", flush=True)
    image_f1 = evaluate_model()
    print(f"\nImage-Only Macro F1-Score: {image_f1:.4f}", flush=True)