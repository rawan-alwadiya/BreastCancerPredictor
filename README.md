# BreastCancerPredictor: ANN-Based Diagnostic Tool for Early Detection

BreastCancerPredictor is a deep learning project that uses an artificial neural network (ANN) to predict breast cancer diagnoses (malignant or benign) from 30 diagnostic features in the Breast Cancer Wisconsin dataset.  
It demonstrates how ANN-based models can be applied in healthcare to create accurate, efficient, and accessible diagnostic tools.

---

## Demo

[View LinkedIn Demo Post](https://www.linkedin.com/posts/rawan-alwadeya-17948a305_deeplearning-artificialneuralnetworks-binaryclassification-activity-7362198947546755072-FdN6?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE3YzG0BAZw48kimDDr_guvq8zXgSjDgk_I)

---

## Project Overview

BreastCancerPredictor uses a complete deep learning pipeline — from EDA and preprocessing to model training, evaluation, and deployment via Streamlit.

The ANN was trained on 30 numerical features extracted from digitized images of fine needle aspirates (FNA) of breast masses.

### Performance Metrics
- **Accuracy (Train)**: 98.02%  
- **Accuracy (Test)**: 97.37%  
- **Precision**: 98%  
- **Recall**: 97%  
- **F1 Score**: 97%

---

## Project Workflow

- **Exploration & Visualization**: Data distributions, skewness check, outlier detection  
- **Preprocessing**: Standard feature scaling, label encoding of target variable, train-test split  
- **Modeling**: ANN with Dense & Dropout layers, trained using EarlyStopping to avoid overfitting  
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score  
- **Deployment**: Interactive Streamlit app for real-time predictions

---

## Dataset

- **Source**: [Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)  
- **Samples**: 569  
- **Features**: 30 numerical measurements  
- **Target Classes**: Malignant / Benign

---

## Real-World Impact

This project highlights the potential of deep learning in medical diagnostics, particularly for early detection — helping to improve treatment outcomes and potentially save lives.

---

## Project Links

- **Kaggle Notebook**: [View on Kaggle](https://www.kaggle.com/code/rawanalwadeya/breastcancerpredictor-ann-based-diagnostic-tool)  
- **Live Streamlit App**: [Try it Now](https://breastcancerpredictor-dpjxeyvzje8xdjfbtvbjkh.streamlit.app)

---

## Tech Stack

**Languages & Libraries**:  
- Python, Pandas, NumPy  
- scikit-learn, TensorFlow/Keras  
- Matplotlib, Seaborn  
- Streamlit (Deployment)  

**Techniques**:  
- ANN (Artificial Neural Network)  
- Feature Scaling  
- Model Evaluation  
- EDA  
- Streamlit Deployment  
