# Flower Recognition Project

This project trains a ResNet-18 model to classify 5 types of flowers (daisy, dandelion, rose, sunflower, tulip) using a Kaggle dataset. It is implemented in Google Colab with PyTorch and includes a Gradio interface for testing predictions. The model uses data augmentation, dropout, and early stopping to prevent overfitting and underfitting.

## Dataset
- **Source**: Kaggle [Flowers Recognition Dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)
- **Structure**: Subfolders for each class (`daisy`, `dandelion`, `rose`, `sunflower`, `tulip`), each containing images.
- **Setup**: Download the dataset and upload it to your Google Drive (e.g., `/MyDrive/Flowers/flowers`).

## Files
- `Lab4 Flower Recognition.ipynb`: Colab notebook with code for training and testing the model.
- `best_flower_model.pth`: Trained ResNet-18 model weights (download from [Google Drive](https://drive.google.com/your-model-link)).
- `requirements.txt`: List of required Python libraries.

## Setup Instructions
1. **Open in Colab**:
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/flower-recognition/blob/main/Lab4%20Flower%20Recognition.ipynb)
2. **Upload Dataset**:
   - Download the Kaggle dataset and place it in your Google Drive (e.g., `/MyDrive/Flowers/flowers`).
   - Update `dataset_path` in Cell 2 of the notebook to match your dataset path.
3. **Upload Model Weights**:
   - Download `best_flower_model.pth` from the provided Google Drive link.
   - Place it in `/MyDrive/` in your Google Drive.
4. **Enable GPU**:
   - In Colab, go to **Runtime** > **Change runtime type** and select **GPU**.
5. **Run the Notebook**:
   - Execute all cells sequentially.
   - The notebook installs dependencies, trains the model, and launches a Gradio interface for testing.
6. **Test Predictions**:
   - Use the Gradio interface (Cell 8) to upload a flower image and view predictions (e.g., "sunflower" with confidence score).

## Requirements
See `requirements.txt` for dependencies. Install them in Colab using:
```bash
!pip install -r requirements.txt
```

## Expected Output
For a sunflower image, the Gradio interface outputs:
```
Predicted Flower: sunflower
Confidence: 95.67%
Probabilities:
daisy: 1.23%
dandelion: 0.98%
rose: 1.45%
sunflower: 95.67%
tulip: 0.67%
```

## Notes
- **Training**: Takes ~10–20 minutes for ~2,575 images on Colab’s GPU.
- **Model Size**: `best_flower_model.pth` (~45 MB) is hosted externally due to GitHub’s file size limits.
- **Dataset**: Not included in the repository due to size; download from Kaggle.

## License
MIT License
