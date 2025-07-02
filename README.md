# Multi-Modal Movie Genre Classification

A deep learning project that classifies movie genres by leveraging both **movie posters** and **plot summaries** using multimodal fusion.

---

## 📂 Dataset

You can download datasets containing movie posters and plot summaries from popular sources such as:  
- [Hugging Face](https://huggingface.co/)  
- [Kaggle](https://www.kaggle.com/)  
- [IMDb](https://www.imdb.com/)  
- [OMDb API](https://www.omdbapi.com/)  
- [TMDb](https://www.themoviedb.org/)  

These sources provide both poster images and textual plot summaries which you can use for training and testing.

---

## 🛠 Training

- For **poster images**, we use the **ConvNeXt-Tiny** model.  
- For **plot summaries**, we use the **DistilBERT** model.  
- Finally, the features from both modalities are fused for multi-modal classification.

**Training workflow:**  
1. Run `multimodal_csv_pipeline.py` to prepare your dataset pipeline.  
2. Run `train_fusion.py` to train the fusion model.

You can also train the individual poster or text models separately if needed.

---

## 🚀 Running the Application

1. Open this project folder in your preferred IDE.  
2. Run the `app.py` script.  
3. Open the localhost URL shown in the terminal by **Ctrl + Click** (usually `http://127.0.0.1:5000/`).  
4. Try uploading movie posters and plot summaries through the web interface and see the genre predictions live!

---

## 📁 Project Structure

- `app.py` — Flask app for serving the model and UI  
- `train_fusion.py` — Script to train the multimodal fusion model  
- `multimodal_csv_pipeline.py` — Dataset preprocessing pipeline  
- `models/` — Folder containing pre-trained or saved models  
- `templates/` — HTML files for the web interface  

---

## 🤝 Contributions

Feel free to open issues or submit pull requests. Contributions to improve the model or UI are welcome!

---

## ⚠️ Note on Dataset and Model Sharing

Due to copyright and licensing restrictions, **movie poster images and plot summary datasets are not included** in this repository. Please download datasets from the sources mentioned above.  

The trained model files (if any) can be found [here](link_to_model_files) or you can train your own by following the training instructions.

---

## 📬 Contact

For questions or collaboration, reach out via [your-email@example.com].

---

Thank you for checking out this project! Enjoy exploring multimodal movie genre classification! 🎬🍿
