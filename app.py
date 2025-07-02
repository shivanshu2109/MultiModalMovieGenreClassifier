from flask import Flask, request, render_template
import torch
from torch import nn
from torch.amp import autocast
from transformers import DistilBertTokenizerFast, DistilBertModel
import timm
from PIL import Image
import os
from torchvision import transforms

app = Flask(__name__)

# ─── Configuration ─────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GENRE_PATH = "genre_classes_new.txt"
MODEL_PATH = "fusion_best.pth"

# ─── Load Genre Classes ────────────────────────────────────
with open(GENRE_PATH) as f:
    GENRE_CLASSES = [line.strip() for line in f]
NUM_CLASSES = len(GENRE_CLASSES)

# ─── Image Transform ───────────────────────────────────────
IMAGE_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─── Text Tokenizer ────────────────────────────────────────
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# ─── Fusion Model ───────────────────────────────────────────
class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.text_enc = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.img_enc = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
        self.head = nn.Sequential(
            nn.Linear(self.text_enc.config.hidden_size + self.img_enc.num_features, 768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, num_classes)
        )

    def forward(self, input_ids=None, attention_mask=None, image=None):
        features = []

        if input_ids is not None and attention_mask is not None:
            text_feat = self.text_enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            features.append(text_feat)

        if image is not None:
            img_feat = self.img_enc.forward_features(image)
            if hasattr(self.img_enc, "forward_head") and self.img_enc.num_classes == 0:
                img_feat = self.img_enc.forward_head(img_feat)
            features.append(img_feat)

        if not features:
            raise ValueError("At least one input (text or image) is required.")

        combined = torch.cat(features, dim=1) if len(features) > 1 else features[0]
        return self.head(combined)

# ─── Load Model ─────────────────────────────────────────────
model = FusionModel(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ─── Routes ────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("poster")
    synopsis = request.form.get("synopsis", "").strip()

    if not file and not synopsis:
        return render_template("index.html", top3=[], error="Please provide at least a poster or a synopsis.")

    # Default dummy tensors
    image_tensor = torch.zeros(1, 3, 224, 224)
    input_ids = torch.zeros(1, 256).long()
    attention_mask = torch.zeros(1, 256).long()

    # ─ Image processing ─
    if file and file.filename != "":
        try:
            image = Image.open(file.stream).convert("RGB")
            image_tensor = IMAGE_TF(image).unsqueeze(0)  # [1, 3, 224, 224]
        except Exception as e:
            return render_template("index.html", top3=[], error="Could not process image.")

    # ─ Text processing ─
    if synopsis:
        tokens = tokenizer(
            synopsis,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

    # ─ Inference ─
    with torch.no_grad(), autocast(device_type=DEVICE.type):
        logits = model(input_ids.to(DEVICE), attention_mask.to(DEVICE), image_tensor.to(DEVICE))
        probs = torch.sigmoid(logits).squeeze(0).cpu()

    topk = torch.topk(probs, 3)
    top3 = [(GENRE_CLASSES[i], probs[i].item()) for i in topk.indices]

    return render_template("index.html", top3=top3)

from flask import send_from_directory

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

# ─── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    from flask import send_from_directory
    app.run(host="0.0.0.0", port=5000, debug=False)