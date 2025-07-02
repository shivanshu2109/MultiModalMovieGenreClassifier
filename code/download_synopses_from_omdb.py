
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm
from PIL import Image
from transformers import DistilBertTokenizerFast
import torchvision.transforms.v2 as transforms

SCRIPT_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR           = SCRIPT_DIR
IMDB_IDS_FILE      = os.path.join(DATA_DIR, "imdb_ids.txt")
POSTER_CSV         = os.path.join(DATA_DIR, "poster_genres_filtered_all_titletypes_fixed.csv")
GENRE_CLASSES_FILE = os.path.join(DATA_DIR, "genre_classes.txt")
TRAIN_CSV          = os.path.join(DATA_DIR, "train_multimodal.csv")
VAL_CSV            = os.path.join(DATA_DIR, "val_multimodal.csv")
IMAGE_DIR          = os.path.join(DATA_DIR, "movie_posters_new")

OMDB_API_KEY = "insert_your_omdb_api_key_here"  # Replace with your OMDb API key
OMDB_URL     = "http://www.omdbapi.com/"

BATCH_SIZE         = 256
NUM_WORKERS        = 8
PIN_MEMORY         = True
PREFETCH_FACTOR    = 2
PERSISTENT_WORKERS = True
DROP_LAST          = True

tokenizer    = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
MAX_SEQ_LEN  = 256
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def fetch_plot(imdb_id: str) -> str:
    "Fetch full plot from OMDb."
    try:
        r = requests.get(OMDB_URL, params={
            'apikey': OMDB_API_KEY,
            'i': imdb_id,
            'plot': 'full'
        }, timeout=5)
        data = r.json()
        return data.get('Plot', '') if data.get('Response') == 'True' else ''
    except:
        return ''

class MultiModalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_transform):
        self.df     = df.reset_index(drop=True)
        self.xform  = image_transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(IMAGE_DIR, row['poster_path'])
        img = Image.open(img_path).convert('RGB')
        img = self.xform(img)
        enc = self.tokenizer(
            row['synopsis'], padding='max_length', truncation=True,
            max_length=MAX_SEQ_LEN, return_tensors='pt'
        )
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        genres = row['genres'].split(',')
        label = torch.zeros(len(self.genre_classes), dtype=torch.float32)
        for g in genres:
            g = g.strip()
            if g in self.genre_to_idx:
                label[self.genre_to_idx[g]] = 1.0
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image': img,
            'label': label
        }

if __name__ == '__main__':
    poster_df = pd.read_csv(POSTER_CSV)
    with open(IMDB_IDS_FILE) as f:
        imdb_ids = [l.strip() for l in f if l.strip()]
    with open(GENRE_CLASSES_FILE) as f:
        genre_classes = [g.strip() for g in f]
    genre_to_idx = {g: i for i, g in enumerate(genre_classes)}
    NUM_CLASSES = len(genre_classes)

    syn_map = {}
    with ThreadPoolExecutor(max_workers=NUM_WORKERS*5) as ex:
        futures = {ex.submit(fetch_plot, mid): mid for mid in imdb_ids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc='Downloading OMDb plots'):
            syn_map[futures[fut]] = fut.result() or ''

    df = poster_df[poster_df['tconst'].isin(imdb_ids)].copy()
    df['synopsis'] = df['tconst'].map(syn_map)
    print(f"Merged dataset: {len(df)} samples")

    train_size = int(0.8 * len(df))
    val_size   = len(df) - train_size
    train_sub, val_sub = random_split(df, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_df = df.iloc[train_sub.indices].reset_index(drop=True)
    val_df   = df.iloc[val_sub.indices].reset_index(drop=True)

    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    print(f"Saved train CSV: {TRAIN_CSV} ({len(train_df)} samples)")
    print(f"Saved val CSV:   {VAL_CSV} ({len(val_df)} samples)")

    torch.backends.cudnn.benchmark = True
    train_ds = MultiModalDataset(train_df, train_transform)
    val_ds   = MultiModalDataset(val_df,   val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              prefetch_factor=PREFETCH_FACTOR,
                              persistent_workers=PERSISTENT_WORKERS,
                              drop_last=DROP_LAST)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                            prefetch_factor=PREFETCH_FACTOR,
                            persistent_workers=PERSISTENT_WORKERS)

    batch = next(iter(train_loader))
    print({k: v.shape for k, v in batch.items()})