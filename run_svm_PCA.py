import os
import re
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

train_dir = 'Users/hongthai/Downloads/animals/train'

def load_images(folder_path, ori=9):
    features = []
    labels = []

    for img in os.listdir(folder_path):
        if not re.search(r'\.(jpg|jpeg|png|bmp|tiff)$', img, re.IGNORECASE):
            continue
        img_path = os.path.join(folder_path, img)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.resize(image, (64, 64))
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog_features = hog(grey_image,
                           orientations=ori,
                           block_norm='L2',
                           pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2))
        features.append(hog_features)
        labels.append(re.split(r'\d+', img)[0].strip('_'))

    return features, labels

# ✅ Lọc thư mục hợp lệ
list_folders_train = [folder for folder in os.listdir(train_dir)
                      if os.path.isdir(os.path.join(train_dir, folder))]

with open('result_svm_PCA.txt', 'w') as f:
    for ori in [9, 18, 36]:
        print(f"[INFO] Processing HOG orientation: {ori}")
        X = []
        y = []

        for folder_train in list_folders_train:
            folder_path_train = os.path.join(train_dir, folder_train)
            features, labels = load_images(folder_path_train, ori)
            X.extend(features)
            y.extend(labels)

        X_ = np.array(X)
        y_ = np.array(y)

        X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=0.3, random_state=42)

        encoder = LabelEncoder()
        y_train_encoded = encoder.fit_transform(y_train)
        y_val_encoded = encoder.transform(y_val)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        for ncomponents in [50, 100, 200, 500]:
            print(f"  [INFO] Applying PCA with n_components={ncomponents}")
            f.write(f'orientations: {ori}, PCA components: {ncomponents}\n')

            pca = PCA(n_components=ncomponents)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_val_pca = pca.transform(X_val_scaled)

            best_svm = {'kernel': '', 'C': 0, 'val_acc': 0.0}

            for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
                for c in [0.01, 0.1, 1, 10]:
                    clf = SVC(kernel=kernel, C=c)
                    clf.fit(X_train_pca, y_train_encoded)
                    acc_train = accuracy_score(y_train_encoded, clf.predict(X_train_pca))
                    acc_val = accuracy_score(y_val_encoded, clf.predict(X_val_pca))

                    f.write(f'kernel: {kernel}, C: {c}, accuracy train: {acc_train:.4f}, accuracy val: {acc_val:.4f}\n')
                    f.flush()

                    if acc_val > best_svm['val_acc']:
                        best_svm = {'kernel': kernel, 'C': c, 'val_acc': acc_val}

            f.write(f'[SVM BEST] kernel={best_svm["kernel"]}, C={best_svm["C"]}, val_accuracy={best_svm["val_acc"]:.4f}\n')
            f.write('-' * 120 + '\n')

        f.write('=' * 120 + '\n')
        f.flush()
