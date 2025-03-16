import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import random

# Create directories if they don't exist
os.makedirs('models/nn', exist_ok=True)
os.makedirs('data/fashion_mnist', exist_ok=True)

print("Loading Fashion MNIST dataset...")
# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.copy()
x_test = x_test.copy()

# Add some imperfections to the dataset
print("Adding imperfections to the dataset...")

# 1. Add noise to some images
noise_indices = np.random.choice(len(x_train), size=int(0.1 * len(x_train)), replace=False)
for idx in noise_indices:
    noise = np.random.normal(0, 25, x_train[idx].shape)
    x_train[idx] = np.clip(x_train[idx] + noise, 0, 255)

# 2. Rotate some images
rotate_indices = np.random.choice(len(x_train), size=int(0.05 * len(x_train)), replace=False)
for idx in rotate_indices:
    k = random.choice([1, 2, 3])  # Rotate 90, 180, or 270 degrees
    x_train[idx] = np.rot90(x_train[idx], k)

# 3. Create a subset with missing data (let's say 10% of pixels are set to 0)
missing_indices = np.random.choice(len(x_train), size=int(0.1 * len(x_train)), replace=False)
for idx in missing_indices:
    mask = np.random.rand(*x_train[idx].shape) < 0.1
    x_train[idx][mask] = 0

# Save some examples of imperfect images
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[noise_indices[i]], cmap='gray')
    plt.title("Noisy")
    plt.axis('off')
plt.savefig('data/fashion_mnist/noisy_examples.png')

plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[rotate_indices[i]], cmap='gray')
    plt.title("Rotated")
    plt.axis('off')
plt.savefig('data/fashion_mnist/rotated_examples.png')

plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[missing_indices[i]], cmap='gray')
    plt.title("Missing Data")
    plt.axis('off')
plt.savefig('data/fashion_mnist/missing_data_examples.png')

# Data preparation
print("Preparing data...")

# Reshape and normalize
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Save some processed data for visualization
with open('data/fashion_mnist/processed_samples.pkl', 'wb') as f:
    pickle.dump({
        'x_samples': x_test[:20],
        'y_samples': np.argmax(y_test[:20], axis=1)
    }, f)

# Define class names for reference
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

with open('data/fashion_mnist/class_names.pkl', 'wb') as f:
    pickle.dump(class_names, f)

# Build the neural network model
print("Building neural network model...")

model = Sequential([
    # First convolutional layer
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Second convolutional layer
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Flatten and dense layers
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Print model summary
model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
checkpoint = ModelCheckpoint(
    'models/nn/model.h5',
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    verbose=1,
    mode='max',
    restore_best_weights=True
)

# Train the model
print("Training neural network model...")
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=20,
    batch_size=128,
    callbacks=[checkpoint, early_stopping]
)

# Save the model
model.save('models/nn/model.h5')

# Save training history for visualization
with open('models/nn/training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Evaluate the model
print("Evaluating model...")
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")

# Generate predictions for sample images
predictions = model.predict(x_test[:10])
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test[:10], axis=1)

# Save model info
model_info = {
    'accuracy': float(accuracy),
    'loss': float(loss),
    'class_names': class_names,
    'sample_predictions': [{
        'true_class': int(true_classes[i]),
        'true_class_name': class_names[true_classes[i]],
        'predicted_class': int(predicted_classes[i]),
        'predicted_class_name': class_names[predicted_classes[i]],
        'confidence': float(predictions[i][predicted_classes[i]])
    } for i in range(10)]
}

with open('models/nn/model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)

print("Neural Network training completed successfully!")