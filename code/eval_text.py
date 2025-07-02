import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

class MovieSynopsisDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.encodings = tokenizer(
            texts,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

def evaluate_model():
    PT_DATA_PATH   = "movie_text_data_nonempty.pt"
    MODEL_DIR_PATH = "text_model.pt"

    data = torch.load(PT_DATA_PATH)
    val_texts  = data['val_texts']
    val_labels = data['val_labels']

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    val_dataset = MovieSynopsisDataset(val_texts, val_labels, tokenizer)
    val_loader  = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=val_labels.shape[1],
        problem_type="multi_label_classification"
    )
    model.load_state_dict(torch.load(MODEL_DIR_PATH, map_location="cpu"))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.sigmoid(outputs.logits)
            preds   = (probs > 0.5).float()

            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()

    f1       = f1_score(y_true, y_pred, average="micro")
    precision= precision_score(y_true, y_pred, average="micro")
    recall   = recall_score(y_true, y_pred, average="micro")

    print(f"\n Evaluation Results:")
    print(f"F1 Micro:        {f1:.4f}")
    print(f"Precision Micro: {precision:.4f}")
    print(f"Recall Micro:    {recall:.4f}")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    evaluate_model()