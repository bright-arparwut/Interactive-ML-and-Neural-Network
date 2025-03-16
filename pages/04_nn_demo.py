import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps, ImageDraw
import io
import pandas as pd
from PIL import Image, ImageDraw

st.set_page_config(page_title="Fashion Classification Demo", page_icon="👚", layout="wide")

st.title("Fashion Item Classification Demo")

# Check if model exists
model_exists = os.path.exists('models/nn/model.h5')
class_names_exist = os.path.exists('data/fashion_mnist/class_names.pkl')

if model_exists and class_names_exist:
    # Load model and class names
    model = load_model('models/nn/model.h5')
    
    with open('data/fashion_mnist/class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)
    
    # Load model info if available
    model_info = None
    if os.path.exists('models/nn/model_info.pkl'):
        with open('models/nn/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
    
    st.header("Classify Fashion Items")
    
    st.write(f"""
    This demo allows you to classify fashion items using a neural network trained on the Fashion MNIST dataset.
    The model can identify 10 different categories of clothing items with an accuracy of 
    **{model_info['accuracy']:.2%}** on the test dataset.
    
    You can try the model in three different ways:
    1. Draw a clothing item
    2. Upload an image (will be converted to grayscale and resized)
    3. Test with random examples from the dataset
    """)
    
    # Create tabs for different demo options
    tab1, tab2, tab3 = st.tabs(["Draw", "Upload", "Random Examples"])
    
    with tab1:
        st.subheader("Draw a Fashion Item")
        
        st.write("""
        Use the canvas below to draw a clothing item that matches one of the 10 categories:
        T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, or Ankle boot.
        
        **Tips**:
        - Draw in white on the black background
        - Try to center your drawing
        - Keep it simple and clear
        """)
        
        # Create a canvas for drawing
        # Since we can't directly use streamlit_drawable_canvas in this environment,
        # we'll create a simplified version using PIL and st.image
        
        # Create a blank canvas with a size of 280x280 (10x the model input size)
        canvas_size = 280
        
        # Create a canvas state
        if 'canvas' not in st.session_state:
            # Create a blank black canvas
            canvas = Image.new('RGB', (canvas_size, canvas_size), (0, 0, 0))
            draw = ImageDraw.Draw(canvas)
            st.session_state.canvas = canvas
            st.session_state.drawing = False
            st.session_state.last_x = None
            st.session_state.last_y = None
        
        # Display instructions for using our simplified drawing tool
        st.write("""
        **Since we're using a simplified drawing tool:**
        1. Click 'Draw Mode' to activate drawing
        2. Click on the black canvas to draw dots
        3. When finished, click 'Classify Drawing'
        """)
        
        # Display the canvas
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.image(st.session_state.canvas, caption="Drawing Canvas", use_column_width=True)
        
        with col2:
            # Add buttons for drawing controls
            if st.button("Clear Canvas"):
                canvas = Image.new('RGB', (canvas_size, canvas_size), (0, 0, 0))
                st.session_state.canvas = canvas
                st.experimental_rerun()
            
            drawing_mode = st.checkbox("Draw Mode", value=st.session_state.drawing)
            st.session_state.drawing = drawing_mode
            
            if drawing_mode:
                st.info("Click on the canvas to draw")
                
                # Get mouse position with a placeholder
                mouse_x = st.number_input("X position", min_value=0, max_value=canvas_size, value=canvas_size//2, label_visibility="collapsed")
                mouse_y = st.number_input("Y position", min_value=0, max_value=canvas_size, value=canvas_size//2, label_visibility="collapsed")
                
                if st.button("Draw at current position"):
                    # Draw a white circle at the mouse position
                    draw = ImageDraw.Draw(st.session_state.canvas)
                    draw.ellipse((mouse_x-10, mouse_y-10, mouse_x+10, mouse_y+10), fill=(255, 255, 255))
                    st.experimental_rerun()
        
        # Add a button to classify the drawing
        if st.button("Classify Drawing", type="primary"):
            # Preprocess the drawing for the model
            img = st.session_state.canvas.convert('L')  # Convert to grayscale
            img = img.resize((28, 28), Image.Resampling.LANCZOS)  # Resize to 28x28
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Normalize
            img_array = img_array.astype('float32') / 255.0
            
            # Reshape for the model
            img_array = img_array.reshape(1, 28, 28, 1)
            
            # Make prediction
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            # Display results
            st.subheader("Classification Result")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write(f"**Predicted Class:** {class_names[predicted_class]}")
                st.write(f"**Confidence:** {confidence:.2%}")
            
            with col2:
                # Display prediction probabilities as a bar chart
                fig, ax = plt.subplots(figsize=(10, 5))
                y_pos = np.arange(len(class_names))
                ax.barh(y_pos, prediction[0], align='center')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(class_names)
                ax.invert_yaxis()  # Labels read top-to-bottom
                ax.set_xlabel('Probability')
                ax.set_title('Class Probabilities')
                
                st.pyplot(fig)
    
    with tab2:
        st.subheader("Upload an Image")
        
        st.write("""
        Upload an image of a clothing item. The image will be converted to grayscale, 
        resized to 28x28 pixels, and then classified by the model.
        
        For best results, upload an image with:
        - A single clothing item
        - A plain background
        - The item centered in the frame
        """)
        
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read and preprocess the uploaded image
            image = Image.open(uploaded_file)
            
            # Display the original image
            st.image(image, caption="Uploaded Image", width=300)
            
            # Preprocess the image
            # Convert to grayscale
            image = image.convert('L')
            
            # Resize to 28x28
            image = image.resize((28, 28), Image.Resampling.LANCZOS)
            
            # Display the preprocessed image
            st.image(image, caption="Preprocessed Image (28x28 pixels)", width=200)
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Normalize
            img_array = img_array.astype('float32') / 255.0
            
            # Invert if needed (assuming fashion items are light on dark background)
            if img_array.mean() > 0.5:  # If the image is mostly white
                img_array = 1 - img_array  # Invert colors
            
            # Reshape for the model
            img_array = img_array.reshape(1, 28, 28, 1)
            
            # Make prediction
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            # Display results
            st.subheader("Classification Result")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write(f"**Predicted Class:** {class_names[predicted_class]}")
                st.write(f"**Confidence:** {confidence:.2%}")
            
            with col2:
                # Display prediction probabilities as a bar chart
                fig, ax = plt.subplots(figsize=(10, 5))
                y_pos = np.arange(len(class_names))
                ax.barh(y_pos, prediction[0], align='center')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(class_names)
                ax.invert_yaxis()  # Labels read top-to-bottom
                ax.set_xlabel('Probability')
                ax.set_title('Class Probabilities')
                
                st.pyplot(fig)
    
    with tab3:
        st.subheader("Test with Random Examples")
        
        st.write("""
        Click the button below to test the model with a random example from the Fashion MNIST test dataset.
        This helps demonstrate how the model performs on the actual data it was trained on.
        """)
        
        # Load some test examples if available
        samples_exist = os.path.exists('data/fashion_mnist/processed_samples.pkl')
        
        if samples_exist:
            with open('data/fashion_mnist/processed_samples.pkl', 'rb') as f:
                samples = pickle.load(f)
            
            if st.button("Get Random Example", type="primary"):
                # Select a random example
                idx = np.random.randint(0, len(samples['x_samples']))
                
                # Get the image and true label
                image = samples['x_samples'][idx]
                true_label = samples['y_samples'][idx]
                
                # Display the image
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(image.reshape(28, 28), cmap='gray')
                ax.set_title(f"True Label: {class_names[true_label]}")
                ax.axis('off')
                
                st.pyplot(fig)
                
                # Make prediction
                prediction = model.predict(image.reshape(1, 28, 28, 1))
                predicted_class = np.argmax(prediction[0])
                confidence = prediction[0][predicted_class]
                
                # Display results
                if predicted_class == true_label:
                    st.success(f"✅ Correctly classified as: {class_names[predicted_class]}")
                else:
                    st.error(f"❌ Incorrectly classified as: {class_names[predicted_class]} (should be {class_names[true_label]})")
                
                st.write(f"**Confidence:** {confidence:.2%}")
                
                # Display prediction probabilities as a bar chart
                fig, ax = plt.subplots(figsize=(10, 5))
                y_pos = np.arange(len(class_names))
                ax.barh(y_pos, prediction[0], align='center')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(class_names)
                ax.invert_yaxis()  # Labels read top-to-bottom
                ax.set_xlabel('Probability')
                ax.set_title('Class Probabilities')
                
                st.pyplot(fig)
                
                # Show examples of the predicted class
                st.subheader(f"Examples of {class_names[predicted_class]}")
                
                # Find examples of the predicted class
                class_indices = np.where(samples['y_samples'] == predicted_class)[0][:5]
                
                if len(class_indices) > 0:
                    fig, axes = plt.subplots(1, min(5, len(class_indices)), figsize=(12, 3))
                    if len(class_indices) == 1:
                        axes = [axes]  # Make axes iterable if there's only one image
                    
                    for i, idx in enumerate(class_indices):
                        if i < len(axes):
                            axes[i].imshow(samples['x_samples'][idx].reshape(28, 28), cmap='gray')
                            axes[i].set_title(f"Example {i+1}")
                            axes[i].axis('off')
                    
                    st.pyplot(fig)
        else:
            st.warning("Sample data not found. Please run the training script first to generate sample data.")
    
    # Add information about the model
    with st.expander("About the Fashion Classification Model"):
        st.write(f"""
        This classification is performed using a Convolutional Neural Network (CNN) trained on the Fashion MNIST dataset.
        
        **Model Performance Metrics**:
        - Accuracy: {model_info['accuracy']:.4f}
        - Loss: {model_info['loss']:.4f}
        
        The model was trained to recognize 10 different categories of clothing items:
        """)
        
        # Display class examples in a grid
        if samples_exist:
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            axes = axes.flatten()
            
            for i, class_name in enumerate(class_names):
                # Find an example of this class
                class_indices = np.where(samples['y_samples'] == i)[0]
                if len(class_indices) > 0:
                    idx = class_indices[0]
                    axes[i].imshow(samples['x_samples'][idx].reshape(28, 28), cmap='gray')
                    axes[i].set_title(class_name)
                    axes[i].axis('off')
            
            st.pyplot(fig)
    
else:
    st.error("""
    Model files not found. Please run the training script first to generate the model.
    
    The following files are required:
    - models/nn/model.h5
    - data/fashion_mnist/class_names.pkl
    """)