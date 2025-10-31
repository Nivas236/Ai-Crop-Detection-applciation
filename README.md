## AgriGuard AI — Smart Crop Disease Recognition

AI-powered crop disease detection from leaf images using TensorFlow and a custom CNN, packaged as an easy-to-use Streamlit web app.

### What this project does
- **Classifies 38 classes** across multiple plants and diseases (including healthy leaves)
- **Streamlit UI** to upload a leaf photo and get an instant diagnosis
- **Unknown Leaf Detection**: Automatically detects when uploaded images don't match any plant in the database
- **Confidence Scoring**: Shows prediction confidence levels to help users assess reliability
- **Disease Information**: Provides detailed descriptions, symptoms, and treatment recommendations
- **Trained model** saved as `trained_model.keras`

---

### Quick start
1) Create and activate a virtual environment
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) Run the app
```bash
streamlit run web.py
```

4) Open the app in your browser (auto-opens). In the sidebar, go to "Crop Disease Recognition", upload a leaf image (jpg/png/jpeg), and click "Start Analysis".

---

### ✨ Key Features

#### 1. **Non-Leaf Image Detection**
The app automatically detects when non-leaf images are uploaded:
- Color and texture analysis using computer vision
- Immediate warning with clear error message
- Prevents false predictions on unrelated images

#### 2. **Unknown Leaf Detection**
The app uses confidence thresholding to detect when an uploaded leaf doesn't match any supported plant in the database:
- If confidence < 50%, a warning is displayed
- Users are informed the plant may not be supported yet
- Recommendation to consult agricultural experts

#### 3. **Confidence Scoring**
Every prediction displays a confidence percentage:
- **High (≥80%)**: Green indicator — reliable prediction
- **Moderate (50-80%)**: Yellow indicator — decent confidence
- **Low (<50%)**: Red indicator — uncertain, likely unknown plant

#### 4. **Disease Information Database**
Comprehensive information for detected diseases:
- **Description**: What causes the disease
- **Symptoms**: Common visual indicators
- **Treatment**: Actionable recommendations for farmers

#### 5. **Real-time Analysis**
- Fast processing (under 5 seconds)
- Mobile-friendly interface
- Professional UI with logos and theming

---

### Repository layout
- `web.py` — Streamlit app for inference and UI
- `trained_model.keras` — Saved Keras model used for predictions
- `crop-detection.ipynb` — Notebook used to train and evaluate the CNN
- `requirements.txt` — Python dependencies
- `homeIMG.jpg`, `logo3.png` — UI images for the app

---

### How it works

#### Inference flow (`web.py`)
1. User uploads an image in the app.
2. The app checks if the image contains a leaf using color and texture analysis.
3. If not a leaf, displays warning and stops processing.
4. If it's a leaf, the app loads `trained_model.keras` using `tf.keras.models.load_model`.
5. The image is resized to 128×128 and converted to a batch of shape `(1, 128, 128, 3)`.
6. The model outputs class probabilities; the argmax index selects the predicted class.
7. The app maps the index to a human-readable label and checks confidence score.
8. If confidence < 50%, displays "Unknown Leaf" warning.
9. For known diseases, displays detailed information and treatment recommendations.

Key function:
```python
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    confidence = prediction[0][result_index]
    return result_index, confidence
```

Notes:
- The app expects `trained_model.keras` to be in the project root next to `web.py`.
- UI uses simple CSS for theming; images `logo3.png` and `homeIMG.jpg` are referenced directly.

#### Training flow (`crop-detection.ipynb`)
- Loads data using `tf.keras.utils.image_dataset_from_directory` from `Datasets/train` and `Datasets/valid` with image size 128×128 and categorical labels across 38 classes.
- CNN architecture (Sequential): stacked Conv2D/MaxPool blocks → Dropout → Flatten → Dense(1500) → Dropout → Dense(38, softmax).
- Trained with Adam optimizer, categorical cross-entropy, tracked accuracy; history saved to `training_hist.json`; final model saved to `trained_model.keras`.

Minimalized sketch of the architecture:
```python
model = Sequential([
    Conv2D(32, 3, padding="same", activation="relu", input_shape=(128,128,3)),
    Conv2D(32, 3, activation="relu"),
    MaxPool2D(2),
    Conv2D(64, 3, padding="same", activation="relu"),
    Conv2D(64, 3, activation="relu"),
    MaxPool2D(2),
    Conv2D(128, 3, padding="same", activation="relu"),
    Conv2D(128, 3, activation="relu"),
    MaxPool2D(2),
    Conv2D(256, 3, padding="same", activation="relu"),
    Conv2D(256, 3, activation="relu"),
    MaxPool2D(2),
    Conv2D(512, 3, padding="same", activation="relu"),
    Conv2D(512, 3, activation="relu"),
    MaxPool2D(2),
    Dropout(0.25),
    Flatten(),
    Dense(1500, activation="relu"),
    Dropout(0.4),
    Dense(38, activation="softmax"),
])
```

---

### Supported classes (38)
- Apple — Apple Scab
- Apple — Black Rot
- Apple — Cedar Apple Rust
- Apple — Healthy
- Blueberry — Healthy
- Cherry — Powdery Mildew
- Cherry — Healthy
- Corn — Cercospora Leaf Spot
- Corn — Common Rust
- Corn — Northern Leaf Blight
- Corn — Healthy
- Grape — Black Rot
- Grape — Esca (Black Measles)
- Grape — Leaf Blight
- Grape — Healthy
- Orange — Huanglongbing (Citrus Greening)
- Peach — Bacterial Spot
- Peach — Healthy
- Bell Pepper — Bacterial Spot
- Bell Pepper — Healthy
- Potato — Early Blight
- Potato — Late Blight
- Potato — Healthy
- Raspberry — Healthy
- Soybean — Healthy
- Squash — Powdery Mildew
- Strawberry — Leaf Scorch
- Strawberry — Healthy
- Tomato — Bacterial Spot
- Tomato — Early Blight
- Tomato — Late Blight
- Tomato — Leaf Mold
- Tomato — Septoria Leaf Spot
- Tomato — Spider Mites
- Tomato — Target Spot
- Tomato — Yellow Leaf Curl Virus
- Tomato — Mosaic Virus
- Tomato — Healthy

---

### Requirements
See `requirements.txt`. Core stack:
- streamlit, tensorflow/keras, numpy, pandas, matplotlib, scikit-learn, opencv-python, scipy

Python 3.10+ is recommended.

---

### Re-training the model (optional)
1) Prepare dataset directories:
```
Datasets/
  train/  # 38 subfolders, one per class
  valid/  # 38 subfolders, one per class
```
2) Open `crop-detection.ipynb` and run cells top-to-bottom. Adjust epochs, batch size, and augmentations as needed.
3) The notebook saves `trained_model.keras` in the project root.

Tips:
- Ensure class subfolder names and ordering match between train/valid; `image_dataset_from_directory` infers labels by folder order.
- Keep the input size (128×128) consistent with the app.

---

### Troubleshooting
- App won’t start: verify environment activation and `pip install -r requirements.txt`.
- Model not found: make sure `trained_model.keras` exists in the project root next to `web.py`.
- Import errors on Windows: upgrade pip and install build tools if TensorFlow fails.
  ```bash
  python -m pip install --upgrade pip
  pip install --upgrade tensorflow
  ```
- Black screen or missing images in UI: check presence of `homeIMG.jpg` and `logo3.png` in the root.

---

### Credits and license
- Dataset: Kaggle — "New Plant Diseases Dataset"
- Model and app: AgriGuard AI (Rohit)
- License: MIT (add your preferred license if different)

If you use this repo, consider citing or linking back. Contributions welcome!

