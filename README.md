<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="UIT Logo">
  </a>
</p>

<h1 align="center">CS231.P21.KHTN - Introduction to Computer Vision</h1>
<h2 align="center">Final Project: Animal Classification</h2>

## Course information:
- **University**: University of Information Technology - VNUHCM UIT.
- **Faculty**: Computer Science
- **Semester**: 2
- **Year**: 2024 - 2025
- **Teacher**: Dr. Mai Tien Dung
- **Course length**: 15 weeks
- **Final Project**: Animal Classification
- 
## 👨‍💻 Team Members
| Student ID | Name                     | Email                         |
|------------|--------------------------|-------------------------------|
| 23521418   | Nguyen Van Hong Thai     | 23521418@gm.uit.edu.vn       |
| 23521437   | Cao Le Cong Thanh        | 23521437@gm.uit.edu.vn       |

## 📌 Overview
This project aims to build a machine learning system to classify animal images into different categories. The task is challenging due to the diversity in image backgrounds, lighting conditions, and animal poses.

## 📚 Motivation
Animal classification is a popular application in computer vision with uses in wildlife monitoring, biology research, and education. The project allows us to apply various ML and DL techniques, compare their performance, and understand how feature engineering and model selection affect results.

## 🧠 Methods
We implemented and evaluated the following models:

- **Traditional ML pipeline**:
  - Feature Extraction: Histogram of Oriented Gradients (HOG)
  - Dimensionality Reduction: Principal Component Analysis (PCA)
  - Classifiers: SVM, KNN, Random Forest

- **Deep Learning**:
  - Transfer Learning using **VGG16** pretrained on ImageNet

## 🗃️ Dataset
- Source: [Kaggle - Animals Dataset](https://www.kaggle.com/datasets/antobenedetti/animals)
- Classes: Dog, Cat, Elephant, Horse, Lion
- Preprocessing: Resize, Grayscale (for ML), Normalization

## 📈 Evaluation Metrics
- Accuracy
- Classification Report (Precision, Recall, F1-score)

## 📊 Results
- Best accuracy with VGG16: **~92%**
- Traditional ML (HOG + PCA + SVM): **~80%**

## 🔧 Requirements
- Python 3
- NumPy, OpenCV, scikit-learn, matplotlib
- TensorFlow / Keras

## 📂 Folder Structure

