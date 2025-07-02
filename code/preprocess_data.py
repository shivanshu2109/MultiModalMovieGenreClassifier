import torch
from torch.utils.data import Dataset, random_split
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms.v2 as transforms
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Determine the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR
IMAGE_DIR = os.path.join(DATA_DIR, "movie_posters_new")

# Validate image directory
if not os.path.exists(IMAGE_DIR):
    logger.error(f"Image directory does not exist at {IMAGE_DIR}")
    sys.exit(1)
if not os.listdir(IMAGE_DIR):
    logger.error(f"Image directory {IMAGE_DIR} is empty")
    sys.exit(1)
logger.info(f"Image directory validated: {IMAGE_DIR}")

# Load genre classes
GENRE_CLASSES_PATH = os.path.join(DATA_DIR, "genre_classes.txt")
logger.info(f"Looking for genre_classes.txt at: {GENRE_CLASSES_PATH}")
if not os.path.exists(GENRE_CLASSES_PATH):
    logger.error(f"File does not exist at {GENRE_CLASSES_PATH}. Directory contents: {os.listdir(DATA_DIR)}")
    sys.exit(1)
try:
    with open(GENRE_CLASSES_PATH, 'r') as f:
        genre_classes = [line.strip() for line in f]
except FileNotFoundError:
    logger.error(f"Error: Could not find 'genre_classes.txt' at {GENRE_CLASSES_PATH}")
    sys.exit(1)
NUM_CLASSES = len(genre_classes)
logger.info(f"Loaded {NUM_CLASSES} genre classes")

# Transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class MoviePosterDataset(Dataset):
    def __init__(self, df, labels, transform=None):
        self.df = df
        self.labels = labels
        self.transform = transform
        self.failed_images = 0
        self.total_images_processed = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        self.total_images_processed += 1
        poster_path = os.path.join(IMAGE_DIR, self.df.iloc[idx]['poster_path'])
        try:
            image = Image.open(poster_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {poster_path}: {e}")
            self.failed_images += 1
            image = Image.new('RGB', (224, 224))  # Fallback for missing/corrupted images
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

    def get_failed_images_count(self):
        return self.failed_images

    def get_total_images_processed(self):
        return self.total_images_processed

def preprocess_data():
    # Load dataset
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "poster_genres_filtered_all_titletypes_fixed.csv"))
    except FileNotFoundError:
        logger.error(f"Error: Could not find 'poster_genres_filtered_all_titletypes_fixed.csv' at {DATA_DIR}")
        sys.exit(1)
    
    # Memory usage warning
    dataset_size = len(df)
    memory_estimate = dataset_size * NUM_CLASSES * 4 / (1024**2)  # Estimate in MB (float32 = 4 bytes)
    if memory_estimate > 1024:  # Warn if memory usage exceeds 1GB
        logger.warning(f"Large dataset detected. Estimated memory for labels: {memory_estimate:.2f} MB")
    
    logger.info(f"Loaded dataset with {dataset_size} samples")
    labels = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)
    for idx, genres in enumerate(df['genres']):
        for genre in genres.split(','):
            genre = genre.strip()
            if genre in genre_classes:
                labels[idx, genre_classes.index(genre)] = 1.0

    # Split dataset
    train_size = int(0.8 * len(df))
    val_size = len(df) - train_size
    train_subset, val_subset = random_split(df, [train_size, val_size])

    train_indices = train_subset.indices
    val_indices = val_subset.indices

    train_dataset = MoviePosterDataset(df.iloc[train_indices], labels[train_indices], transform=train_transform)
    val_dataset = MoviePosterDataset(df.iloc[val_indices], labels[val_indices], transform=val_transform)

    logger.info(f"Created train dataset with {len(train_dataset)} samples and val dataset with {len(val_dataset)} samples")
    return train_dataset, val_dataset

if __name__ == "__main__":
    # Ensure multiprocessing works on Windows
    torch.multiprocessing.set_start_method('spawn', force=True)
    train_dataset, val_dataset = preprocess_data()
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")