# 🎬 Multi-Modal Movie Genre Classification

A deep learning project that classifies movie genres using **movie posters** and **plot summaries** via a multimodal fusion approach.

---

## 📂 Dataset

You can download datasets containing movie posters and plot summaries from popular sources such as:  
- [Hugging Face](https://huggingface.co/)  
- [Kaggle](https://www.kaggle.com/)  
- [IMDb](https://www.imdb.com/)  
- [OMDb API](https://www.omdbapi.com/)  
- [TMDb](https://www.themoviedb.org/)  

These sources provide both poster images and textual plot summaries that can be used for training and testing.

---

## 🛠 Training

- **Posters** → [`ConvNeXt-Tiny`](https://arxiv.org/abs/2201.03545)  
- **Summaries** → [`DistilBERT`](https://arxiv.org/abs/1910.01108)  
- **Fusion** → Poster + Text embeddings fused for genre classification

### 🧪 Training Workflow
1. Prepare dataset pipeline: [`code/multimodal_csv_pipeline.py`](code/multimodal_csv_pipeline.py)  
2. Train the multimodal model: [`code/train_fusion.py`](code/train_fusion.py)

You can also separately train poster or text-based models if required.

---

## 🚀 Run the Web App

1. Open this project in your IDE or terminal.
2. Run the Flask app using: [`app.py`](app.py)  
3. Click on the localhost link (e.g., `http://127.0.0.1:5000/`) in the terminal.
4. Upload a poster or enter a plot summary to see genre predictions live!

---

## 📁 Project Structure

