# **BreastCancerPredictor: ANN-Based Diagnostic Tool for Early Detection**

BreastCancerPredictor is a deep learning project that applies an **Artificial Neural Network (ANN)** to predict breast cancer diagnoses (**Malignant** or **Benign**) from 30 diagnostic features in the Breast Cancer Wisconsin dataset.  
The project demonstrates an **end-to-end machine learning workflow** including **EDA, preprocessing, ANN modeling, evaluation, and deployment with Streamlit**.

---

## **Demo**

- 🎥 [View LinkedIn Demo Post](https://www.linkedin.com/posts/rawan-alwadeya-17948a305_deeplearning-artificialneuralnetworks-binaryclassification-activity-7362198947546755072-FdN6?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE3YzG0BAZw48kimDDr_guvq8zXgSjDgk_I)  
- 🌐 [Try the App Live on Streamlit](https://breastcancerpredictor-dpjxeyvzje8xdjfbtvbjkh.streamlit.app)  

![Malignant Prediction Example](https://github.com/rawan-alwadiya/BreastCancerPredictor/blob/main/Malignant%20Prediction.png)

---

## **Project Overview**

The workflow includes:  
- **Exploration & Visualization**: data distributions, skewness check, outlier detection  
- **Preprocessing**: feature scaling, label encoding, train-test split  
- **Modeling (ANN)**: Dense & Dropout layers with EarlyStopping  
- **Evaluation**: accuracy, precision, recall, F1-score  
- **Deployment**: interactive **Streamlit web app** for real-time predictions  

---

## **Objective**

Develop and deploy a robust **ANN-based classifier** to assist in early detection of breast cancer, supporting timely treatment decisions.

---

## **Dataset**

- **Source**: [Breast Cancer Wisconsin Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)  
- **Samples**: 569  
- **Features**: 30 numerical measurements  
- **Target Classes**: Malignant / Benign  

---

## **Project Workflow**

- **EDA & Visualization**: feature distribution analysis, skewness check, outlier detection  
- **Preprocessing**:  
  - Standard feature scaling  
  - Label encoding of target variable  
  - Train-test split  
- **Modeling (ANN)**:  
  - Multiple Dense layers with ReLU activation  
  - Dropout layers to prevent overfitting  
  - Output layer with Sigmoid for binary classification  
- **Training Setup**:  
  - Optimizer: Adam  
  - Loss: Binary Crossentropy  
  - Callback: EarlyStopping (to avoid overfitting)  

---

## **Performance Results**

**Artificial Neural Network Classifier:**  
- **Accuracy (Train)**: 98.02%  
- **Accuracy (Test)**: 97.37%  
- **Precision**: 0.98  
- **Recall**: 0.97  
- **F1-score**: 0.97  

The model achieved **high accuracy and balanced precision-recall**, making it reliable for breast cancer diagnosis.

---

## **Project Links**

- **Kaggle Notebook**: [View on Kaggle](https://www.kaggle.com/code/rawanalwadeya/breastcancerpredictor-ann-based-diagnostic-tool)  
- **Streamlit App**: [Try it Now](https://breastcancerpredictor-dpjxeyvzje8xdjfbtvbjkh.streamlit.app)  

---

## **Tech Stack**

**Languages & Libraries**:  
- Python, Pandas, NumPy  
- scikit-learn, TensorFlow / Keras  
- Matplotlib, Seaborn  
- Streamlit (Deployment)  

**Techniques**:  
- ANN (Artificial Neural Network with Dense & Dropout layers)  
- Feature Scaling & Label Encoding  
- EarlyStopping for regularization  
- Streamlit Deployment for real-time predictions  
