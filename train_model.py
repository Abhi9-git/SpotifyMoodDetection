import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

from data_processing import load_and_preprocess_data

def train_and_evaluate():
    # 1. Get the preprocessed data
    X_train, X_test, y_train, y_test, scaler, _ = load_and_preprocess_data()
    
    # 2. Initialize the model
    print("Initializing Multinomial Logistic Regression model...")
    # saga solver is good for large datasets and supports multinomial loss
    model = LogisticRegression(solver='saga', max_iter=1000, n_jobs=-1, random_state=42)
    
    # 3. Train the model
    print("Training the model (this might take a moment)...")
    model.fit(X_train, y_train)
    print("Model training complete.")
    
    # 4. Evaluate the model
    print("Evaluating the model on the test set...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 5. Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    os.makedirs('evaluation', exist_ok=True)
    plt.tight_layout()
    plt.savefig('evaluation/confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved in 'evaluation' directory.")
    
    # 6. Save the model and the scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/logistic_regression_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Model and scaler saved to the 'models' directory.")

if __name__ == "__main__":
    train_and_evaluate()
