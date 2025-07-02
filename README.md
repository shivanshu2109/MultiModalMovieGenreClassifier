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
- [`how_to_run.txt`](how_to_run.txt) — detailed setup and run instructions  

---

## ⚙️ How to Run

> Quick steps to get started locally:

1. **Clone & install dependencies**
2. **(Optional)** Run `code/multimodal_csv_pipeline.py` to prepare your dataset  
3. **(Optional)** Train the model using `code/train_fusion.py`  
4. Run `app.py` and open `http://127.0.0.1:5000/` in your browser  
5. Upload a poster or input a plot summary to get genre predictions  

👉 For full setup instructions, refer to [`how_to_run.txt`](how_to_run.txt)

---

## 📦 Dataset & Model Access

Due to licensing restrictions, datasets are not included.  
You can collect your own from:
- [Kaggle](https://www.kaggle.com/)
- [IMDb](https://www.imdb.com/)
- [TMDb](https://www.themoviedb.org/)
- [Hugging Face Datasets](https://huggingface.co/datasets)

Trained models are available in the [`models/`](models) directory.  

---

## 🤝 Contributions

Have an idea or improvement?  
Feel free to fork the repository, submit a pull request, or open an issue!

---

## 📬 Contact

For questions or collaboration: **[shivanshu985@gmail.com]**

---

> Built with ❤️ for film lovers, data geeks, and AI explorers.
