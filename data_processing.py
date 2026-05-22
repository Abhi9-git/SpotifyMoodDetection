import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

DATA_PATH = r"c:\Users\abhin\OneDrive\Desktop\SpotifyMood\SpotifyMoodDetection\278k_song_labelled.csv\278k_song_labelled.csv"

def load_and_preprocess_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Drop the unnamed index column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    print(f"Data loaded. Shape: {df.shape}")
    
    # Check for missing values
    if df.isnull().sum().any():
        print("Handling missing values...")
        df = df.dropna()
        print(f"Data shape after dropping NA: {df.shape}")

    # Features and labels
    X = df.drop(columns=['labels'])
    y = df['labels']

    print("\nFeature Summary:")
    print(X.describe())
    
    print("\nClass distribution:")
    print(y.value_counts())

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for easier handling if needed
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    print("\nData preprocessing complete.")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, df

def perform_eda(df):
    print("Generating EDA plots...")
    
    # Create an 'eda_plots' directory
    os.makedirs('eda_plots', exist_ok=True)
    
    # Correlation Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Features")
    plt.tight_layout()
    plt.savefig('eda_plots/correlation_matrix.png')
    plt.close()

    # Distribution of labels
    plt.figure(figsize=(6, 4))
    sns.countplot(x='labels', data=df)
    plt.title("Distribution of Mood Labels")
    plt.tight_layout()
    plt.savefig('eda_plots/label_distribution.png')
    plt.close()

    print("EDA plots saved in 'eda_plots' directory.")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, df = load_and_preprocess_data()
    perform_eda(df)
