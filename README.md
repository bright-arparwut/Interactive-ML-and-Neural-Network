🌟 Features

Interactive ML Model Exploration: Understand traditional machine learning through a heart disease prediction model
Neural Network Visualization: Explore convolutional neural networks with the Fashion MNIST dataset
Real-time Predictions: Test models with your own inputs or randomly generated examples
Educational Insights: Learn about data preparation, model architecture, and evaluation metrics
Drawing Interface: Draw your own fashion items and see classification results instantly
Comprehensive Visualizations: Explore data relationships and model performance through intuitive charts

💻 Technologies

Frontend: Streamlit
Data Processing: Pandas, NumPy
Machine Learning: Scikit-learn
Neural Networks: TensorFlow
Visualization: Matplotlib, Seaborn, Plotly
Image Processing: Pillow

📋 Project Structure
intellilearn/
├── app.py                   # Main application entry point
├── pages/                   # Application pages
│   ├── 01_ml_ex.py          # ML model explanation
│   ├── 02_nn_ex.py          # Neural network explanation
│   ├── 03_ml_demo.py        # Heart disease prediction demo
│   └── 04_nn_demo.py        # Fashion MNIST classification demo
├── data/                    # Dataset storage
│   ├── heart_disease.csv
│   └── fashion_mnist/
├── models/                  # Trained models
│   ├── ml/                  # Machine learning models
│   └── nn/                  # Neural network models
└── requirements.txt         # Project dependencies
🚀 Getting Started
Prerequisites

Python 3.8 or higher
pip (Python package manager)

Installation

Clone the repository:
bashgit clone https://github.com/yourusername/intellilearn.git
cd intellilearn

Create a virtual environment (optional but recommended):
bashpython -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

Install dependencies:
bashpip install -r requirements.txt

Run the application:
bashstreamlit run app.py

Open your browser and navigate to http://localhost:8501
