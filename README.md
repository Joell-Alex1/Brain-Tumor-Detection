# Brain Tumor Detection (CNN) — Colab Notebook

This repository contains a Colab notebook (Brain_Tumor_Detection.ipynb) that trains a convolutional neural network (CNN) to classify brain MRI scans as containing a tumor (yes) or not (no).

Quick highlights
- Built with TensorFlow / Keras, OpenCV, and scikit-learn.
- Uses the "navoneel/brain-mri-images-for-brain-tumor-detection" dataset (downloaded via KaggleHub in the notebook).
- Example demo cells include training, evaluation (accuracy, loss, confusion matrix), and single-image prediction.

NOTE: This is a research/demo project only. It is not a medical device and must not be used for clinical diagnosis.

---

## Files
- Brain_Tumor_Detection.ipynb — main Colab notebook (training, evaluation, demo).
- README.md — this file.

---

## Requirements

Recommended (Colab has most preinstalled):
- Python 3.8+
- TensorFlow (tested with TF 2.x)
- opencv-python
- numpy
- scikit-learn
- matplotlib
- seaborn
- kagglehub (used in the notebook to download the dataset)

If running locally, create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows

pip install --upgrade pip
pip install tensorflow opencv-python numpy scikit-learn matplotlib seaborn kagglehub
```

(You can also add these to a requirements.txt file if desired.)

---

## Dataset

The notebook expects the dataset structure:

brain_tumor_dataset/
- yes/
- no/

The Colab notebook uses `kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")` to fetch and cache the dataset. If you prefer to download manually:

1. Download from Kaggle: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
2. Unzip and ensure the folder `brain_tumor_dataset` with `yes` and `no` subfolders exists.
3. Update `data_dir` in the notebook to point to that folder.

---

## How to run (Colab)

1. Open the notebook in Google Colab:
   - Upload `Brain_Tumor_Detection.ipynb` to Colab or open from GitHub.
2. Run the cells top-to-bottom.
   - The notebook installs `kagglehub` and downloads the dataset automatically.
   - Training runs on Colab CPU by default; enable GPU: Runtime → Change runtime type → GPU.
3. Use the “Predict on a User-Uploaded Image” cell to upload a single MRI and see the model prediction.

---

## How to run (Locally)

1. Prepare dataset locally (see Dataset section).
2. Open the notebook with Jupyter Notebook / JupyterLab or convert to a .py script.
3. Ensure `data_dir` in the notebook points to the local dataset directory.
4. Run the notebook. For faster training use a machine with a CUDA-enabled GPU and the compatible TensorFlow build.

Optional: save the trained model in the notebook after training:

```python
model.save("brain_tumor_cnn.h5")
# Later load:
from tensorflow import keras
model = keras.models.load_model("brain_tumor_cnn.h5")
```

---

## Notebook details & configurable hyperparameters

Key variables you may want to tune in the notebook:
- IMG_SIZE (default 150) — input image size
- epochs (default 20) — training epochs
- batch_size (default 32)
- model architecture — add/remove Conv2D or Dense layers
- Dropout rate (default 0.5)

The notebook normalizes images to [0,1], uses grayscale images (shape: IMG_SIZE x IMG_SIZE x 1), and uses `binary_crossentropy` with a sigmoid output for binary classification.

---

## Evaluation & Outputs

The notebook:
- Prints model.summary()
- Plots training/validation accuracy and loss
- Produces a classification report (precision, recall, f1-score)
- Displays a confusion matrix heatmap
- Demonstrates predictions on individual images

---

## Troubleshooting

- "Directory not found" errors: verify `data_dir` path and that `yes` / `no` folders exist.
- Corrupted or non-image files: notebook catches and logs read errors for individual files.
- GPU issues locally: ensure CUDA and cuDNN are installed and TensorFlow version matches CUDA.

---

## Ethical & Legal Notice / Disclaimer

This project is for educational and research purposes only. It is not validated for clinical use. Do not use this model to make medical decisions. Always consult qualified medical professionals.

---

## Credits & Acknowledgements

- Dataset: navoneel / brain-mri-images-for-brain-tumor-detection (Kaggle)
- Built with: TensorFlow, OpenCV, scikit-learn, Matplotlib, Seaborn

---

NOTES:
This is for educational purposes only

```
