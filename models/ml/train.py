import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import joblib
import os

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('models/ml', exist_ok=True)

# Download heart disease dataset if it doesn't exist
if not os.path.exists('data/heart_disease.csv'):
    print("Downloading heart disease dataset...")
    url = "https://raw.githubusercontent.com/datasets/heart-disease/master/data/heart.csv"
    df = pd.read_csv(url)
    
    # Add some imperfections for data preprocessing
    # Replace 5% of values with NaN
    mask = np.random.rand(*df.shape) < 0.05
    df_copy = df.copy()
    df_copy[mask] = np.nan
    
    # Add some outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        outlier_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
        multiplier = np.random.choice([2, 3, 4], size=len(outlier_indices))
        df_copy.loc[outlier_indices, col] = df_copy.loc[outlier_indices, col] * multiplier
    
    df_copy.to_csv('data/heart_disease.csv', index=False)
    print("Dataset saved to data/heart_disease.csv")
else:
    print("Heart disease dataset already exists")
    df_copy = pd.read_csv('data/heart_disease.csv')

# Exploratory Data Analysis
print("\nDataset Information:")
print(df_copy.info())
print("\nDataset Statistics:")
print(df_copy.describe())
print("\nMissing Values:")
print(df_copy.isnull().sum())

# Data preparation
print("\nPreparing data...")

# Define features and target
X = df_copy.drop('target', axis=1)
y = df_copy['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define preprocessing for numerical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Model Training
print("\nTraining models...")

# List of models to try
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True)
}

# Dictionary to store results
results = {}

# Train and evaluate models
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Create pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'model': pipeline,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    print(f"{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Classification Report:")
    print(classification_report(y_test, y_pred))

# Find the best model based on F1 score
best_model_name = max(results, key=lambda k: results[k]['f1'])
best_model = results[best_model_name]['model']
print(f"\nBest Model: {best_model_name} with F1 Score: {results[best_model_name]['f1']:.4f}")

# Calculate feature importance for Random Forest or Gradient Boosting
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print("\nFeature Importance:")
    importances = best_model.named_steps['model'].feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    print(importance_df)

# Save the best model
print("\nSaving the best model...")
joblib.dump(best_model, 'models/ml/model.pkl')
print("Model saved to models/ml/model.pkl")

# Save model results for streamlit app
model_info = {
    'model_name': best_model_name,
    'accuracy': results[best_model_name]['accuracy'],
    'precision': results[best_model_name]['precision'],
    'recall': results[best_model_name]['recall'],
    'f1': results[best_model_name]['f1'],
    'feature_names': list(X.columns)
}

joblib.dump(model_info, 'models/ml/model_info.pkl')
print("Model info saved to models/ml/model_info.pkl")

print("\nMachine Learning training completed successfully!")