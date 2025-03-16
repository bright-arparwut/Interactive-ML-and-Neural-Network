import streamlit as st

st.set_page_config(
    page_title="ML & Neural Network Project",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Machine Learning & Neural Network Project")

st.markdown("""
## Welcome to our ML & NN Project Demo

This web application demonstrates the implementation of machine learning and neural network models on two different datasets:

1. **Heart Disease Dataset**: Used for traditional machine learning models
2. **Fashion MNIST Dataset**: Used for neural network model

### Project Overview

This project follows these key steps:
1. Data acquisition and understanding
2. Data preparation and preprocessing
3. Model development (both ML and NN)
4. Model evaluation and optimization
5. Model deployment and demonstration

### Navigation

Use the sidebar to navigate through the different pages:
- **ML Model Explanation**: Details about the heart disease dataset and ML approach
- **Neural Network Explanation**: Details about the Fashion MNIST dataset and NN approach
- **ML Model Demo**: Interactive demo of the heart disease prediction model
- **Neural Network Demo**: Interactive demo of the Fashion MNIST classification model

Developed as part of a project requirement for Machine Learning & Neural Networks course.
""")

st.sidebar.title("Navigation")
st.sidebar.info(
    """
    Select a page from the dropdown in the sidebar to explore different aspects of the project.
    """
)