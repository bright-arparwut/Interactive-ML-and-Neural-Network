import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd

st.set_page_config(page_title="Neural Network Explanation", page_icon="🧠", layout="wide")

st.title("Neural Network Model Explanation")

# Check if model and data files exist
fashion_mnist_exists = os.path.exists('data/fashion_mnist')
model_exists = os.path.exists('models/neural_network/model.h5')
model_info_exists = os.path.exists('models/neural_network/model_info.pkl')
history_exists = os.path.exists('models/neural_network/training_history.pkl')
class_names_exists = os.path.exists('data/fashion_mnist/class_names.pkl')
samples_exist = os.path.exists('data/fashion_mnist/processed_samples.pkl')

if fashion_mnist_exists and class_names_exists:
    # Load class names
    with open('data/fashion_mnist/class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)
        
    # Load model info if available
    model_info = None
    if model_info_exists:
        with open('models/neural_network/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
    
    # Load training history if available
    history = None
    if history_exists:
        with open('models/neural_network/training_history.pkl', 'rb') as f:
            history = pickle.load(f)
            
    st.header("Fashion MNIST Dataset")
    
    st.subheader("1. Dataset Source")
    st.write("""
    The Fashion MNIST dataset is a collection of 70,000 grayscale images of 10 different clothing categories. 
    It was designed as a drop-in replacement for the original MNIST dataset, offering more variety and a greater challenge for machine learning algorithms.
    
    For this project, we introduced some imperfections to demonstrate data preparation techniques:
    - Added random noise to 10% of the images
    - Rotated 5% of the images by random degrees
    - Created missing data by setting 10% of pixels to 0 in some images
    """)
    
    st.subheader("2. Dataset Features")
    
    # Display class information
    class_df = pd.DataFrame({
        'Class ID': range(10),
        'Class Name': class_names,
        'Description': [
            'Upper body clothing with short sleeves',
            'Lower body clothing covering legs',
            'Upper body clothing with long sleeves',
            'Full body clothing',
            'Upper body outerwear',
            'Footwear with open design',
            'Upper body clothing with buttons',
            'Athletic footwear',
            'Accessory for carrying items',
            'Footwear that covers the ankle'
        ]
    })
    
    st.table(class_df)
    
    st.write("""
    **Image Properties**:
    - Format: Grayscale images
    - Size: 28x28 pixels
    - Value Range: 0-255 (8-bit grayscale)
    - Training set: 60,000 images
    - Test set: 10,000 images
    """)
    
    # Show example images
    st.subheader("3. Dataset Samples")
    
    if samples_exist:
        with open('data/fashion_mnist/processed_samples.pkl', 'rb') as f:
            samples = pickle.load(f)
        
        # Display original samples
        st.write("**Sample Images from Different Classes**")
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for i in range(10):
            sample_idx = np.where(samples['y_samples'] == i)[0][0]
            axes[i].imshow(samples['x_samples'][sample_idx].reshape(28, 28), cmap='gray')
            axes[i].set_title(class_names[i])
            axes[i].axis('off')
            
        st.pyplot(fig)
    
    # Show imperfections examples if files exist
    if os.path.exists('data/fashion_mnist/noisy_examples.png'):
        st.write("**Examples of Images with Noise**")
        st.image('data/fashion_mnist/noisy_examples.png')
        
    if os.path.exists('data/fashion_mnist/rotated_examples.png'):
        st.write("**Examples of Rotated Images**")
        st.image('data/fashion_mnist/rotated_examples.png')
        
    if os.path.exists('data/fashion_mnist/missing_data_examples.png'):
        st.write("**Examples of Images with Missing Data**")
        st.image('data/fashion_mnist/missing_data_examples.png')
    
    st.header("Data Preparation Steps")
    
    st.write("""
    We implemented the following data preparation steps to address the imperfections and prepare the data for neural network training:
    
    1. **Image Normalization**: 
       - Scaled pixel values from 0-255 to 0-1 by dividing by 255
       - This helps the neural network converge faster during training
    
    2. **Reshaping**: 
       - Reshaped images from (28, 28) to (28, 28, 1) to add a channel dimension required by convolutional layers
    
    3. **Label One-Hot Encoding**: 
       - Converted categorical labels (0-9) to one-hot encoded vectors
       - Each vector has 10 elements, with a 1 in the position of the correct class and 0s elsewhere
    
    4. **Data Augmentation**: 
       - Instead of cleaning the imperfections (noise, rotation, missing data), we embraced them as a form of data augmentation
       - This makes the model more robust to variations and helps prevent overfitting
    """)
    
    st.header("Neural Network Architecture")
    
    st.write("""
    We designed a Convolutional Neural Network (CNN) architecture specifically tailored for image classification tasks. CNNs are well-suited for processing grid-like data such as images, as they can automatically learn spatial hierarchies of features.
    """)
    
    st.subheader("1. Model Architecture")
    
    # Display model architecture diagram
    st.write("**CNN Architecture for Fashion MNIST Classification**")
    
    model_architecture = {
        "Layer Type": [
            "Input", 
            "Conv2D", 
            "BatchNormalization", 
            "Conv2D", 
            "MaxPooling2D", 
            "Dropout",
            "Conv2D",
            "BatchNormalization",
            "Conv2D",
            "MaxPooling2D",
            "Dropout",
            "Flatten",
            "Dense",
            "BatchNormalization",
            "Dropout",
            "Dense (Output)"
        ],
        "Output Shape": [
            "(28, 28, 1)",
            "(28, 28, 32)",
            "(28, 28, 32)",
            "(28, 28, 32)",
            "(14, 14, 32)",
            "(14, 14, 32)",
            "(14, 14, 64)",
            "(14, 14, 64)",
            "(14, 14, 64)",
            "(7, 7, 64)",
            "(7, 7, 64)",
            "(3136)",
            "(512)",
            "(512)",
            "(512)",
            "(10)"
        ],
        "Parameters": [
            "0",
            "320",
            "128",
            "9,248",
            "0",
            "0",
            "18,496",
            "256",
            "36,928",
            "0",
            "0",
            "0",
            "1,606,144",
            "2,048",
            "0",
            "5,130"
        ],
        "Activation": [
            "None",
            "ReLU",
            "None",
            "ReLU",
            "None",
            "None",
            "ReLU",
            "None",
            "ReLU",
            "None",
            "None",
            "None",
            "ReLU",
            "None",
            "None",
            "Softmax"
        ]
    }
    
    arch_df = pd.DataFrame(model_architecture)
    st.table(arch_df)
    
    st.subheader("2. Key Components Explained")
    
    components = {
        "Convolutional Layers": """
        The CNN uses two pairs of convolutional layers with increasing filter counts (32 to 64).
        These layers learn to detect features like edges, textures, and patterns in the images.
        Each pair increases in complexity, allowing the network to learn hierarchical representations.
        """,
        
        "Batch Normalization": """
        Applied after convolutional layers to normalize the activations, which:
        - Accelerates training by allowing higher learning rates
        - Reduces the dependency on careful initialization
        - Acts as a regularizer, reducing the need for dropout in some cases
        """,
        
        "MaxPooling Layers": """
        Reduces spatial dimensions by taking the maximum value in each 2x2 window.
        This helps to:
        - Reduce the number of parameters and computation in the network
        - Provide translation invariance (the ability to recognize objects regardless of their position)
        - Extract dominant features
        """,
        
        "Dropout Layers": """
        Randomly sets a fraction of input units to 0 during training, which:
        - Prevents overfitting by forcing the network to learn redundant representations
        - Approximates ensemble learning by creating multiple "thinned" networks during training
        - Applied after pooling and dense layers with rates of 0.25 and 0.5, respectively
        """,
        
        "Dense Layers": """
        The flattened features are passed through a dense layer with 512 units.
        This layer combines the spatial features learned by the convolutional layers.
        The final dense layer with 10 units and softmax activation produces class probabilities.
        """
    }
    
    for component, description in components.items():
        with st.expander(component):
            st.write(description)
    
    st.subheader("3. Training Process")
    
    st.write("""
    The model was trained with the following settings:
    
    - **Optimizer**: Adam optimizer with default learning rate
    - **Loss Function**: Categorical Crossentropy (standard for multi-class classification)
    - **Batch Size**: 128 images per batch
    - **Epochs**: Up to 20 epochs with early stopping
    - **Validation Split**: 10,000 images used for validation (test set)
    
    We implemented early stopping to prevent overfitting, saving the best model based on validation accuracy.
    """)
    
    # Show training history if available
    if history:
        st.write("**Training and Validation Metrics**")
        
        # Plot accuracy and loss
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(history['accuracy'], label='Training Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(history['loss'], label='Training Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        st.pyplot(fig)
    
    # Display model performance if available
    if model_info:
        st.subheader("4. Model Performance")
        
        st.write(f"The neural network achieved the following performance on the test dataset:")
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Loss'],
            'Value': [
                f"{model_info['accuracy']:.4f}",
                f"{model_info['loss']:.4f}"
            ]
        })
        
        st.table(metrics_df)
        
        st.write("""
        The accuracy metric represents the proportion of correctly classified images. 
        For a 10-class classification problem, achieving this level of accuracy indicates that the model has learned meaningful patterns from the data despite the introduced imperfections.
        """)
        
        # Show sample predictions
        if 'sample_predictions' in model_info:
            st.write("**Sample Predictions**")
            
            sample_preds = model_info['sample_predictions']
            pred_df = pd.DataFrame({
                'True Class': [p['true_class_name'] for p in sample_preds],
                'Predicted Class': [p['predicted_class_name'] for p in sample_preds],
                'Confidence': [f"{p['confidence']:.4f}" for p in sample_preds],
                'Correct': [p['true_class'] == p['predicted_class'] for p in sample_preds]
            })
            
            st.dataframe(pred_df)
    
    else:
        st.info("Model performance data not available. Please run the training script first.")
        
else:
    st.error("Fashion MNIST dataset or model not found. Please run the training script first.")