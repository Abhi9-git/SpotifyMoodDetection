# Spotify Mood Detection

A machine learning project designed to automatically classify the mood of Spotify songs based on their acoustic features. This model utilizes **Multinomial Logistic Regression** to sort tracks into four distinct emotional categories: Calm, Happy, Energetic, and Sad.

## High-Level Implementation Overview

### 1. Data Acquisition
The project begins with a comprehensive dataset downloaded from Kaggle (`278k_song_labelled.csv`). This dataset contains over 278,000 Spotify tracks, each annotated with a numerical mood label and accompanied by key audio features:
- **Features:** Duration, Danceability, Energy, Loudness, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo, and Spec Rate.
- **Labels:** The dataset maps to four primary moods which we decoded based on Spotify's audio analysis quadrants (Valence vs. Energy):
  - `0`: **Calm / Relaxed** (Low energy, medium/low valence, high acousticness)
  - `1`: **Happy / Upbeat** (High energy, high valence)
  - `2`: **Energetic / Intense** (Very high energy, loud)
  - `3`: **Sad / Melancholic** (Very low energy, very low valence, very acoustic)

### 2. Data Preprocessing & Exploratory Data Analysis (EDA)
Handled primarily in `data_processing.py`:
- **Cleaning:** The dataset is parsed using `pandas`. Redundant index columns (`Unnamed: 0`) and missing values are dropped to ensure data integrity.
- **Scaling:** Because Logistic Regression relies on gradient descent and distance measurements, we apply `StandardScaler` from `scikit-learn` to normalize all audio features to have a mean of 0 and a variance of 1.
- **Splitting:** The data is split into an 80% training set and a 20% testing set using a stratified split, ensuring that the proportional representation of each mood is perfectly preserved across both sets.
- **EDA:** The script includes methods to generate correlation matrices and label distributions (saved to the `eda_plots/` directory) to better understand the underlying patterns.

### 3. Model Building & Training
Handled in `train_model.py`:
- We initialize a **Multinomial Logistic Regression** model. Given the massive size of the dataset (~222,000 training rows), we configure the model to use the robust `saga` solver which is optimized for large-scale, multi-class classification problems.
- The model is trained on the scaled training features. 
- After training, the model achieves an impressive **~84% Accuracy**. 
- Evaluation metrics (Precision, Recall, F1-Score) and a visual Confusion Matrix are generated and saved to the `evaluation/` directory.
- Finally, the trained model state (`logistic_regression_model.pkl`) and the feature scaler (`scaler.pkl`) are serialized and saved to the `models/` directory using `joblib`.

### 4. Inference and Prediction Pipeline
Handled in `predict.py`:
- We provide a standalone script meant for production or testing usage.
- `predict_mood()` accepts a dictionary of raw audio features for any new song.
- The script automatically loads the serialized model and scaler, scales the incoming features identically to the training phase, and predicts the mood.
- It returns not only the definitive predicted mood label but also the distinct probabilistic confidence scores across all four possible moods.

## Running the Project

Follow these steps to set up the environment, train the model, and make predictions on your local machine.

### 1. Setup Virtual Environment
It's highly recommended to use a virtual environment so the project's dependencies do not conflict with your global Python packages.
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
With the virtual environment active, install all required packages:
```bash
pip install pandas scikit-learn matplotlib seaborn numpy joblib
```

### 3. Generate EDA Plots (Optional)
If you want to view the data distributions and feature correlations, you can run the data processing script directly. This will create an `eda_plots/` directory with helpful visuals.
```bash
python data_processing.py
```

### 4. Train the Model
This step will load the dataset, normalize it, and train the logistic regression model. It will take a minute or two to complete since it processes ~278k songs.
```bash
python train_model.py
```
* **Output:** Once finished, it will display the accuracy and classification report in your terminal. It will also create a `models/` directory containing `logistic_regression_model.pkl` and `scaler.pkl`, and an `evaluation/` directory containing a Confusion Matrix image.*

### 5. Make Predictions
You can test the model on new songs by running the prediction script. 
```bash
python predict.py
```
* **Customizing Inputs:** If you open `predict.py` in your text editor, you can scroll down to the bottom where `sample_song` is defined. Feel free to modify the audio feature values (e.g. changing `danceability` to `0.9` or `valence` to `0.1`) to see how the model's mood prediction changes!*