import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

st.set_page_config(page_title="ML Model Explanation", page_icon="📊", layout="wide")

st.title("Machine Learning Model Explanation")

# Load dataset
if os.path.exists('data/heart_disease.csv'):
    df = pd.read_csv('data/heart_disease.csv')
    
    # Load model info if available
    model_info = None
    if os.path.exists('models/machine_learning/model_info.pkl'):
        model_info = joblib.load('models/machine_learning/model_info.pkl')

    st.header("Heart Disease Dataset")
    
    st.subheader("1. Dataset Source")
    st.write("""
    The Heart Disease dataset was obtained from the UCI Machine Learning Repository. The original dataset contains various medical attributes that can be used to predict the presence of heart disease in a patient.
    
    For this project, we introduced some imperfections to the dataset to demonstrate data preparation techniques:
    - Added missing values (~5% of the data)
    - Introduced outliers to numeric fields (~2% of the data)
    """)
    
    st.subheader("2. Dataset Features")
    
    feature_descriptions = {
        'age': 'Age in years',
        'sex': 'Gender (1 = male, 0 = female)',
        'cp': 'Chest pain type (0-3)',
        'trestbps': 'Resting blood pressure (in mm Hg)',
        'chol': 'Serum cholesterol in mg/dl',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)',
        'restecg': 'Resting electrocardiographic results (0-2)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1 = yes, 0 = no)',
        'oldpeak': 'ST depression induced by exercise relative to rest',
        'slope': 'Slope of the peak exercise ST segment (0-2)',
        'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
        'thal': 'Thalium stress test result (1 = normal, 2 = fixed defect, 3 = reversible defect)',
        'target': 'Heart disease diagnosis (1 = presence, 0 = absence)'
    }
    
    # Display feature table
    feature_df = pd.DataFrame({
        'Feature': feature_descriptions.keys(),
        'Description': feature_descriptions.values()
    })
    
    st.table(feature_df)
    
    # Display dataset sample
    st.subheader("3. Dataset Sample")
    st.dataframe(df.head())
    
    # Display dataset statistics
    st.subheader("4. Dataset Statistics")
    st.write(df.describe())
    
    # Display missing values
    st.subheader("5. Missing Values Analysis")
    missing_vals = df.isnull().sum()
    missing_df = pd.DataFrame({
        'Feature': missing_vals.index,
        'Missing Values': missing_vals.values,
        'Percentage': (missing_vals.values / len(df) * 100).round(2)
    })
    st.table(missing_df)
    
    # Visualizations 
    st.header("Data Visualization and Analysis")
    
    # Distribution of target variable
    st.subheader("1. Distribution of Heart Disease Cases")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='target', data=df, ax=ax)
    ax.set_xlabel('Heart Disease')
    ax.set_ylabel('Count')
    ax.set_xticklabels(['No Disease', 'Disease'])
    ax.set_title('Distribution of Heart Disease Cases')
    
    # Add count labels on bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'bottom',
                    xytext = (0, 5), textcoords = 'offset points')
    
    st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("2. Correlation Between Features")
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)
    
    # Age distribution by target
    st.subheader("3. Age Distribution by Heart Disease")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='age', hue='target', multiple='dodge', bins=10, ax=ax)
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    ax.set_title('Age Distribution by Heart Disease')
    ax.legend(['No Disease', 'Disease'])
    st.pyplot(fig)
    
    # Chest pain type by target
    st.subheader("4. Chest Pain Type vs Heart Disease")
    fig, ax = plt.subplots(figsize=(10, 6))
    cp_counts = df.groupby(['cp', 'target']).size().unstack()
    cp_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel('Chest Pain Type')
    ax.set_ylabel('Count')
    ax.set_title('Chest Pain Type vs Heart Disease')
    ax.legend(['No Disease', 'Disease'])
    st.pyplot(fig)
    
    st.header("Data Preparation Steps")
    
    st.write("""
    The following data preparation steps were implemented to address the imperfections in the dataset:
    
    1. **Handling Missing Values**: We used a median imputation strategy for numerical features. The median is robust to outliers and preserves the distribution of the data better than mean imputation.
    
    2. **Feature Scaling**: Standardization was applied to all numerical features to ensure they have a mean of 0 and a standard deviation of 1. This is important for algorithms that are sensitive to the scale of features, like logistic regression and SVM.
    
    3. **Pipeline Creation**: We created a preprocessing pipeline using scikit-learn's `Pipeline` and `ColumnTransformer` to ensure consistent application of the preprocessing steps during both training and inference.
    """)
    
    st.header("Machine Learning Approach")
    
    st.subheader("1. Algorithms Used")
    
    # Define algorithm descriptions
    algorithms = {
        'Logistic Regression': """
        Logistic Regression is a statistical method for binary classification that estimates the probability of a binary outcome based on one or more predictor variables. It uses the logistic function to model the probability of the default class.
        
        **Key Characteristics**:
        - Simple and interpretable model
        - Works well for linearly separable data
        - Less prone to overfitting with proper regularization
        - Outputs probability scores which can be useful for decision making
        """,
        
        'Random Forest': """
        Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes for classification problems. It helps to overcome overfitting problems that can occur with single decision trees.
        
        **Key Characteristics**:
        - Handles non-linear relationships well
        - Robust to outliers and missing values
        - Provides feature importance metrics
        - Typically requires less tuning than other algorithms
        """,
        
        'Gradient Boosting': """
        Gradient Boosting is another ensemble technique that builds trees sequentially, with each new tree correcting errors made by the previously trained trees. It combines multiple weak learners to form a strong predictor.
        
        **Key Characteristics**:
        - Often achieves higher accuracy than Random Forest
        - More sensitive to hyperparameter tuning
        - Can capture complex patterns in the data
        - May be prone to overfitting without proper regularization
        """,
        
        'Support Vector Machine (SVM)': """
        SVM is a discriminative classifier that finds an optimal hyperplane that maximizes the margin between two classes. It can use different kernel functions to handle non-linear decision boundaries.
        
        **Key Characteristics**:
        - Effective in high-dimensional spaces
        - Memory efficient as it uses a subset of training points (support vectors)
        - Versatile through different kernel functions
        - Sensitive to feature scaling
        """
    }
    
    for algo, description in algorithms.items():
        with st.expander(f"{algo}"):
            st.write(description)
    
    st.subheader("2. Model Selection Process")
    
    st.write("""
    We implemented a systematic model selection process:
    
    1. **Train-Test Split**: We divided the dataset into 75% training and 25% testing sets to evaluate model performance on unseen data.
    
    2. **Multiple Algorithm Evaluation**: We trained and evaluated four different algorithms to find the best performer:
        - Logistic Regression
        - Random Forest Classifier
        - Gradient Boosting Classifier
        - Support Vector Machine
    
    3. **Performance Metrics**: The models were compared using multiple metrics:
        - Accuracy: Overall correctness of the model
        - Precision: Ratio of true positives to all positive predictions
        - Recall: Ratio of true positives to all actual positives
        - F1 Score: Harmonic mean of precision and recall
    """)
    
    # Display model performance if available
    if model_info:
        st.subheader("3. Model Performance Comparison")
        
        st.write(f"The best performing model was **{model_info['model_name']}** with the following metrics:")
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [
                f"{model_info['accuracy']:.4f}",
                f"{model_info['precision']:.4f}",
                f"{model_info['recall']:.4f}",
                f"{model_info['f1']:.4f}"
            ]
        })
        
        st.table(metrics_df)
        
        st.write("""
        The model was selected based on F1 score, which is a balanced metric that considers both precision and recall. This is particularly important for medical diagnosis problems where both false positives and false negatives can have significant consequences.
        """)
    
    else:
        st.info("Model performance data not available. Please run the training script first.")
else:
    st.error("Heart disease dataset not found. Please run the training script first.")