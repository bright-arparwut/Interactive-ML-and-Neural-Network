import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Heart Disease Prediction Demo", page_icon="❤️", layout="wide")

st.title("Heart Disease Prediction Demo")

# Check if model exists
model_exists = os.path.exists('models/ml/model.pkl')
model_info_exists = os.path.exists('models/ml/model_info.pkl')

if model_exists and model_info_exists:
    # Load model and model info
    model = joblib.load('models/ml/model.pkl')
    model_info = joblib.load('models/ml/model_info.pkl')
    
    st.header("Predict Heart Disease Risk")
    
    st.write(f"""
    This demo allows you to input patient information and predict their risk of heart disease using the 
    **{model_info['model_name']}** model with an accuracy of **{model_info['accuracy']:.2%}**.
    
    Enter the patient's information below and click the "Predict" button to get a prediction.
    """)
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=50)
        sex = st.selectbox("Sex", options=["Male", "Female"], index=0)
        sex_value = 1 if sex == "Male" else 0
        
        cp_options = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
        cp = st.selectbox("Chest Pain Type", options=cp_options, index=0)
        cp_value = cp_options.index(cp)
        
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"], index=0)
        fbs_value = 1 if fbs == "Yes" else 0
        
        restecg_options = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
        restecg = st.selectbox("Resting ECG Results", options=restecg_options, index=0)
        restecg_value = restecg_options.index(restecg)
    
    with col2:
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina", options=["No", "Yes"], index=0)
        exang_value = 1 if exang == "Yes" else 0
        
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        
        slope_options = ["Upsloping", "Flat", "Downsloping"]
        slope = st.selectbox("Slope of Peak Exercise ST Segment", options=slope_options, index=0)
        slope_value = slope_options.index(slope)
        
        ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
        
        thal_options = ["Normal", "Fixed Defect", "Reversible Defect"]
        thal = st.selectbox("Thalium Stress Test Result", options=thal_options, index=0)
        # Map thal values to 1, 2, 3 (original dataset uses 1-based indexing for thal)
        thal_value = thal_options.index(thal) + 1
    
    # Create a dataframe with the input values
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex_value],
        'cp': [cp_value],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs_value],
        'restecg': [restecg_value],
        'thalach': [thalach],
        'exang': [exang_value],
        'oldpeak': [oldpeak],
        'slope': [slope_value],
        'ca': [ca],
        'thal': [thal_value]
    })
    
    # Display the input data
    st.subheader("Patient Information Summary")
    
    # Convert numerical codes back to descriptive labels for display
    display_data = input_data.copy()
    display_data['sex'] = "Male" if sex_value == 1 else "Female"
    display_data['cp'] = cp_options[cp_value]
    display_data['fbs'] = "Yes" if fbs_value == 1 else "No"
    display_data['restecg'] = restecg_options[restecg_value]
    display_data['exang'] = "Yes" if exang_value == 1 else "No"
    display_data['slope'] = slope_options[slope_value]
    display_data['thal'] = thal_options[thal_value - 1]
    
    # Rename columns for better readability
    display_data = display_data.rename(columns={
        'age': 'Age',
        'sex': 'Gender',
        'cp': 'Chest Pain Type',
        'trestbps': 'Resting BP (mm Hg)',
        'chol': 'Cholesterol (mg/dl)',
        'fbs': 'High Fasting Blood Sugar',
        'restecg': 'Resting ECG',
        'thalach': 'Max Heart Rate',
        'exang': 'Exercise Angina',
        'oldpeak': 'ST Depression',
        'slope': 'ST Segment Slope',
        'ca': 'Number of Vessels',
        'thal': 'Thalium Test'
    })
    
    # Display as two columns of key-value pairs
    col1, col2 = st.columns(2)
    
    # Split the columns
    display_cols = display_data.columns.tolist()
    half = len(display_cols) // 2 + 1
    
    with col1:
        for col in display_cols[:half]:
            st.write(f"**{col}:** {display_data.iloc[0][col]}")
    
    with col2:
        for col in display_cols[half:]:
            st.write(f"**{col}:** {display_data.iloc[0][col]}")
    
    # Add a prediction button
    if st.button("Predict Heart Disease Risk", type="primary"):
        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        # Display results
        st.header("Prediction Result")
        
        # Get the prediction probability of heart disease (class 1)
        disease_probability = prediction_proba[0][1]
        
        # Create columns for result display
        result_col1, result_col2 = st.columns([1, 2])
        
        with result_col1:
            if prediction[0] == 1:
                st.error("❗ **Heart Disease Detected**")
            else:
                st.success("✅ **No Heart Disease Detected**")
                
            st.write(f"Probability: {disease_probability:.2%}")
        
        with result_col2:
            # Create a gauge-like visualization for the probability
            fig, ax = plt.subplots(figsize=(8, 3))
            
            # Define risk levels
            risk_levels = [(0.0, 0.25, "Low Risk", "green"),
                          (0.25, 0.5, "Moderate Risk", "yellow"),
                          (0.5, 0.75, "High Risk", "orange"),
                          (0.75, 1.0, "Very High Risk", "red")]
            
            # Draw the gauge background
            for start, end, label, color in risk_levels:
                ax.barh(0, end-start, left=start, height=0.5, color=color, alpha=0.6)
                ax.text((start+end)/2, 0.7, label, ha='center', va='center')
            
            # Add the needle/marker
            ax.scatter(disease_probability, 0, color='black', s=200, zorder=10)
            ax.text(disease_probability, -0.3, f"{disease_probability:.2%}", ha='center', va='center', fontweight='bold')
            
            # Adjust plot appearance
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, 1)
            ax.set_title('Heart Disease Risk Level')
            ax.axis('off')
            
            st.pyplot(fig)
        
        # Display interpretation
        st.subheader("Interpretation")
        
        if prediction[0] == 1:
            if disease_probability > 0.75:
                st.write("""
                The model predicts a very high likelihood of heart disease. It is strongly recommended to consult with a cardiologist immediately.
                Key risk factors from the input data may include:
                """)
            elif disease_probability > 0.5:
                st.write("""
                The model predicts a high likelihood of heart disease. A consultation with a healthcare provider is recommended.
                Key risk factors from the input data may include:
                """)
            else:
                st.write("""
                The model predicts a moderate likelihood of heart disease. Consider discussing these results with a healthcare provider.
                Key risk factors from the input data may include:
                """)
        else:
            if disease_probability < 0.25:
                st.write("""
                The model predicts a very low likelihood of heart disease. Continue to maintain a healthy lifestyle.
                Positive factors from the input data may include:
                """)
            else:
                st.write("""
                The model predicts a low likelihood of heart disease, but there are some risk factors to be aware of.
                Consider discussing these results with a healthcare provider. Risk factors from the input data may include:
                """)
        
        # Add disclaimer
        st.warning("""
        **Disclaimer**: This prediction is based on a machine learning model and should not be used as a substitute for professional medical advice. 
        Always consult with a healthcare provider for proper diagnosis and treatment.
        """)
        
        # Show feature importance if available for certain model types
        if model_info['model_name'] in ['Random Forest', 'Gradient Boosting']:
            st.subheader("Feature Importance")
            st.write("The chart below shows which factors were most important in the model's prediction:")
            
            # Get feature importance from the model
            try:
                importances = model.named_steps['model'].feature_importances_
                feature_names = model_info['feature_names']
                
                # Create a dataframe for feature importance
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                })
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                # Plot feature importance
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(importance_df['Feature'], importance_df['Importance'])
                ax.set_xlabel('Importance')
                ax.set_title('Feature Importance')
                ax.invert_yaxis()
                
                st.pyplot(fig)
            except:
                st.write("Feature importance could not be calculated for this model.")
    
    # Add information about the model
    with st.expander("About the Heart Disease Prediction Model"):
        st.write(f"""
        This prediction is performed using a **{model_info['model_name']}** model trained on the Heart Disease dataset.
        
        **Model Performance Metrics**:
        - Accuracy: {model_info['accuracy']:.4f}
        - Precision: {model_info['precision']:.4f}
        - Recall: {model_info['recall']:.4f}
        - F1 Score: {model_info['f1']:.4f}
        
        The model was trained on a dataset with similar features to those input above. It uses a machine learning algorithm to identify patterns associated with heart disease based on patient characteristics and medical test results.
        """)
    
else:
    st.error("""
    Model files not found. Please run the training script first to generate the model.
    
    The following files are required:
    - models/ml/model.pkl
    - models/ml/model_info.pkl
    """)