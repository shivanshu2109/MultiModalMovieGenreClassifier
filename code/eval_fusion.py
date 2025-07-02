import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from fusion_training import get_loaders, build_model
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fusion_best.pth")
GENRE_CLASSES_PATH = os.path.join(os.path.dirname(__file__), "genre_classes_new.txt")
THRESHOLD = 0.5

with open(GENRE_CLASSES_PATH) as f:
    genre_classes = [line.strip() for line in f]

model = build_model(len(genre_classes)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

_, val_loader = get_loaders()

all_preds, all_labels = [], []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        images = batch['image'].to(DEVICE)
        labels = batch['labels'].cpu().numpy()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, image=images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > THRESHOLD).astype(int)

        all_preds.append(preds)
        all_labels.append(labels)

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)

f1 = f1_score(all_labels, all_preds, average='micro')
precision = precision_score(all_labels, all_preds, average='micro')
recall = recall_score(all_labels, all_preds, average='micro')

print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

metrics = [precision, recall, f1]
names = ['Precision', 'Recall', 'F1 Score']
plt.bar(names, metrics)
plt.title("Fusion Model Evaluation Metrics")
plt.ylim(0, 1)
plt.savefig("fusion_model_metrics.png")
plt.show()