# ==============================================
# 🧠 Generative AI Hazard Detection Project
# ==============================================
# Author: Your Name
# Run this in Google Colab
# ==============================================

# STEP 1: Install dependencies
!pip install tensorflow streamlit scikit-learn pandas matplotlib joblib

# STEP 2: Import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# STEP 3: Load your dataset (update path if needed)
# Example assumes CSV dataset: columns = ['feature1', 'feature2', ..., 'label']
from zipfile import ZipFile

# Unzip your uploaded file
zip_path = "/content/archive (2).zip"
with ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("/content/hazard_data")

# Find CSV file inside extracted folder
import glob
csv_files = glob.glob("/content/hazard_data/**/*.csv", recursive=True)
print("Found CSV files:", csv_files)

# Load first CSV
df = pd.read_csv(csv_files[0])
print(df.head())

# STEP 4: Data Preprocessing
X = df.drop('label', axis=1).values  # features
y = df['label'].values               # target

# Encode target if needed
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 5: Build a Simple Neural Network
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# STEP 6: Train Model
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2, verbose=1)

# STEP 7: Evaluate Model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("Accuracy:", accuracy_score(y_true, y_pred_classes))
print(classification_report(y_true, y_pred_classes))

# Save model
model.save("hazard_model.h5")
print("✅ Model saved successfully!")

# STEP 8: Plot training
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
