# 👁️ Eye Disease Classification Using CNN and Transfer Learning (MobileNet)

A deep learning–based computer vision project for automatic eye disease classification from retinal images.
The project compares a custom CNN model with a Transfer Learning approach using MobileNet, achieving a significant performance improvement.

## 🔍 Problem Statement
Eye diseases such as Glaucoma, Cataract, and Diabetic Retinopathy can lead to vision loss if not detected early. Manual diagnosis is time‑consuming and depends heavily on expert availability.

🎯 Goal: Build an accurate and scalable deep learning model that automatically classifies eye diseases from retinal images.

## 📊 Dataset
- Source: Kaggle – Eye Diseases Classification Dataset       
🔗 https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification/code

- Classes & Encoding:
```py
image_class = {
"glaucoma": 0,
"normal": 1,
"diabetic_retinopathy": 2,
"cataract": 3
}
```

## 📈 Class Distribution
Class | Label | Samples|
----|-----|------|
Diabetic Retinopathy |2|1098|
Normal|1|1074|
Cataract|3|1038|
Glaucoma|0|1007|

## 🖼️ Sample Images per Class
- ![Glaucoma](https://github.com/user-attachments/assets/4d0c8a7f-126e-4894-911a-86b654ac636d)
- ![Normal](https://github.com/user-attachments/assets/4616e8c4-7792-47e7-9f01-5fb0a70aba43)
- ![diabetic_retinopathy](https://github.com/user-attachments/assets/283c67ed-1026-4f4c-abf3-0fd222a552ca)
- ![cataract](https://github.com/user-attachments/assets/f05799b0-7d18-4a72-8811-38fea07c3d88)

## 🧠 Model Architectures

### 1️⃣ Custom CNN (Baseline Model)

- Input size: 224 × 224 × 3

- Multiple Conv2D + MaxPooling layers

- Fully connected classifier

📉 Accuracy: **77.9%**

📊 Classification Report (CNN)
```
precision   recall   f1-score   support


0      0.77     0.54      0.64      252
1      0.78     0.69      0.73      269
2      1.00     0.99      0.99      275
3      0.62     0.88      0.73      259


accuracy                0.78     1055
macro avg      0.79   0.78   0.77   1055
weighted avg   0.80   0.78   0.78   1055
```

### 2️⃣ Transfer Learning – MobileNet

To improve performance, Transfer Learning was applied using MobileNet pretrained on ImageNet.

Key Steps:

- Freeze base MobileNet layers

- Add custom classification head

- Fine‑tune top layers

🚀 Accuracy After Transfer Learning: 90.62%

📊 Classification Report (MobileNet)
```
precision   recall   f1-score   support


0      0.84     0.84    0.84    252
1      0.84     0.86    0.85    269
2      0.99     0.98    0.98    275
3      0.95     0.95    0.95    259


accuracy              0.91   1055
macro avg       0.91  0.91  0.91  1055
weighted avg    0.91  0.91  0.91  1055
```

## 🏗️ Model Architecture Visualization

Generated using:
```py
from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True)
```
![Model Architecture](https://github.com/user-attachments/assets/f8a1d888-6356-46db-8e21-7ee74b4893c5)


## 🌐 Streamlit Web Application

An interactive **Streamlit app** was built to deploy the trained model.

Features:

- Image upload

- Real‑time prediction

- Confidence scores for each class

- Clean and user‑friendly UI

🔗 Live Demo: 

📸 Application Screenshot: ![Application Screenshot](https://github.com/user-attachments/assets/5a713fd3-1688-4942-8689-389174f87f97)

## 🛠️ Technologies Used

- Python

- TensorFlow / Keras

- OpenCV

- NumPy, Pandas

- Matplotlib & Seaborn

- Scikit‑learn

- Streamlit
## 📌 Key Results
Model	| Accuracy|
----|----|
Custom CNN	|77.9%
MobileNet (Transfer Learning)	|90.62%

✔ Transfer Learning significantly improved performance and generalization.


## ⚠️ Disclaimer

This project is for educational and research purposes only and should not be used as a replacement for professional medical diagnosis.


## 👨‍💻 Author

***Mohamed Abdelghany***     
Machine Learning & Deep Learning Engineer    

--- 

⭐ If you find this project useful, feel free to star the repository!