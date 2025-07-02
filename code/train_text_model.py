import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score, precision_score, recall_score

class MovieSynopsisDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.float32)
        }

def compute_metrics(pred):
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs > 0.5).int().numpy()
    trues = labels.astype(int)
    return {
        "f1_micro": f1_score(trues, preds, average="micro"),
        "precision_micro": precision_score(trues, preds, average="micro"),
        "recall_micro": recall_score(trues, preds, average="micro"),
    }

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_CSV  = os.path.join(SCRIPT_DIR, "train_multimodal.csv")
    VAL_CSV    = os.path.join(SCRIPT_DIR, "val_multimodal.csv")
    GENRE_PATH = os.path.join(SCRIPT_DIR, "genre_classes_new.txt")
    MODEL_PATH = os.path.join(SCRIPT_DIR, "text_model.pt")

    with open(GENRE_PATH) as f:
        genre_list = [line.strip() for line in f]
    genre_to_idx = {g: i for i, g in enumerate(genre_list)}
    num_classes = len(genre_list)

    def load_df(path):
        df = pd.read_csv(path)
        texts = df['synopsis'].tolist()
        labels = []
        for gstr in df['genres']:
            label = [0.0] * num_classes
            for g in gstr.split(','):
                g = g.strip()
                if g in genre_to_idx:
                    label[genre_to_idx[g]] = 1.0
            labels.append(label)
        return texts, labels

    train_texts, train_labels = load_df(TRAIN_CSV)
    val_texts, val_labels     = load_df(VAL_CSV)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_classes,
        problem_type="multi_label_classification"
    )
    model.gradient_checkpointing_enable()

    data_collator = DataCollatorWithPadding(tokenizer)
    train_dataset = MovieSynopsisDataset(train_texts, train_labels, tokenizer)
    val_dataset   = MovieSynopsisDataset(val_texts,   val_labels,   tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_steps=300,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=200,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        fp16=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        dataloader_drop_last=False,
        push_to_hub=False,
        report_to="none",
    )

    class MultiLabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))

    trainer.train()
    trainer.save_model(MODEL_PATH)
    print(f"✅ Trained model saved to {MODEL_PATH}")

    print("Evaluating the final model on the validation set...")
    eval_results = trainer.evaluate()
    print("Final Validation Metrics:")
    print(f"Eval Loss: {eval_results['eval_loss']:.6f}")
    print(f"F1 Micro: {eval_results['eval_f1_micro']:.6f}")
    print(f"Precision Micro: {eval_results['eval_precision_micro']:.6f}")
    print(f"Recall Micro: {eval_results['eval_recall_micro']:.6f}")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()