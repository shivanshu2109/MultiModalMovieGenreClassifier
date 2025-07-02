import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from transformers import DistilBertTokenizerFast
import torchvision.transforms as transforms
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.manual_seed(42)

BASEDIR    = os.path.dirname(__file__)
TRAIN_CSV  = os.path.join(BASEDIR, "train_multimodal.csv")
VAL_CSV    = os.path.join(BASEDIR, "val_multimodal.csv")
IMAGE_DIR  = os.path.join(BASEDIR, "movie_posters_new")
GENRE_FILE = os.path.join(BASEDIR, "genre_classes_new.txt")

DEFAULT_BATCH_SIZE      = 64
DEFAULT_NUM_WORKERS     = 0
DEFAULT_PIN_MEMORY      = True
DEFAULT_DROP_LAST       = True
MAX_SEQ_LEN             = 256

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

train_img_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

val_img_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, engine="python")
    df = df[
        df['synopsis'].notna() &
        df['synopsis'].str.strip().astype(bool) &
        ~df['synopsis'].str.strip().str.upper().eq("N/A")
    ].reset_index(drop=True)
    return df

def filter_unknown_genres(df: pd.DataFrame, genre_to_idx: dict, split_name: str) -> pd.DataFrame:
    cleaned = []
    for raw in tqdm(df['genres'], desc=f"Filtering {split_name}", unit="sample"):
        kept = [g.strip() for g in raw.split(',') if g.strip() in genre_to_idx]
        cleaned.append(','.join(kept) if kept else None)
    df = df.copy()
    df['genres'] = cleaned
    df = df[df['genres'].notna()].reset_index(drop=True)
    return df

class MultiModalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_tf, genre_to_idx: dict):
        self.df           = df
        self.img_tf       = img_tf
        self.tokenizer    = tokenizer
        self.genre_to_idx = genre_to_idx
        self.num_classes  = len(genre_to_idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        try:
            img = Image.open(os.path.join(IMAGE_DIR, row['poster_path'])).convert("RGB")
        except FileNotFoundError:
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        img = self.img_tf(img)

        enc = self.tokenizer(
            row['synopsis'],
            padding='max_length',
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors='pt'
        )
        input_ids      = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)

        label = torch.zeros(self.num_classes, dtype=torch.float32)
        for g in row['genres'].split(','):
            idx_g = self.genre_to_idx.get(g.strip())
            if idx_g is not None:
                label[idx_g] = 1.0

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "image":          img,
            "labels":         label
        }

def get_loaders(
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    pin_memory: bool = DEFAULT_PIN_MEMORY,
    drop_last: bool = DEFAULT_DROP_LAST
):
    with open(GENRE_FILE) as f:
        genres = [g.strip() for g in f]
    genre_to_idx = {g: i for i, g in enumerate(genres)}

    train_df = load_and_clean(TRAIN_CSV)
    val_df   = load_and_clean(VAL_CSV)

    train_df = filter_unknown_genres(train_df, genre_to_idx, "train")
    val_df   = filter_unknown_genres(val_df, genre_to_idx, "val")

    print(f"Train loaded: {len(train_df):,} valid rows")
    print(f"Val loaded:   {len(val_df):,} valid rows")

    train_ds = MultiModalDataset(train_df, train_img_tf, genre_to_idx)
    val_ds   = MultiModalDataset(val_df,   val_img_tf,   genre_to_idx)

    train_kwargs = {
        "dataset": train_ds,
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        train_kwargs["persistent_workers"] = True

    val_kwargs = {
        "dataset": val_ds,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if num_workers > 0:
        val_kwargs["persistent_workers"] = True

    return DataLoader(**train_kwargs), DataLoader(**val_kwargs)

if __name__ == "__main__":
    tr, vl = get_loaders()
    batch = next(iter(tr))
    print({k: v.shape for k, v in batch.items()})