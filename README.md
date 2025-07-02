# 🎬 Multi-Modal Movie Genre Classification

A deep learning system that classifies movie genres by fusing both **visual features** from movie posters and **semantic embeddings** from plot summaries using state-of-the-art models.

---

## 🧠 Tech Stack

- 🎨 **ConvNeXt-Tiny** — for feature extraction from movie posters  
- ✍️ **DistilBERT** — for understanding plot summaries  
- 🔗 **Fusion Layer** — combines visual + text embeddings  
- ⚙️ **PyTorch**, **Transformers**, **Flask**, **Scikit-learn**

---

## 📁 Key Files & Folders

- [`app.py`](app.py) — Flask web interface for live predictions  
- [`code/multimodal_csv_pipeline.py`](code/multimodal_csv_pipeline.py) — preprocesses and formats dataset  
- [`code/train_fusion.py`](code/train_fusion.py) — trains the multimodal fusion model  
- [`models/`](models/) — saved `.pth` model checkpoints  
- [`templates/`](templates/) — HTML templates for the UI

---

## ⚙️ How to Run

> Quick steps to get started locally:

1. **Clone & install dependencies**
2. **(Optional)** Run `code/multimodal_csv_pipeline.py` to prepare your dataset  
3. **(Optional)** Train model via `code/train_fusion.py`  
4. Run `app.py` and open `http://127.0.0.1:5000/` in your browser  
5. Upload a poster or input a plot summary for genre prediction
## Gor to this file for better understanding - how_to_run.txt
---

## 📦 Dataset & Model Access

Due to licensing, datasets are not included.  
You may use:
- [Kaggle](https://www.kaggle.com/)
- [IMDb](https://www.imdb.com/)
- [TMDb](https://www.themoviedb.org/)
- [Hugging Face](https://huggingface.co/datasets)

The trained model is available in the [`models/`](models) folder. Check `model.txt` for download links if applicable.

---

## 🤝 Contributions

Have an idea or improvement?  
Feel free to fork, submit a PR, or open an issue!

---

## 📬 Contact

For queries or collaboration: **[your-email@example.com]**

---

> Built with ❤️ for film lovers, data geeks, and AI explorers.
