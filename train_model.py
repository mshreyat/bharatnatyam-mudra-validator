#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

print("Training started...")

# 1. Load dataset ###########################################################
data = pd.read_csv("dataset.csv", header=None)
print("Dataset shape:", data.shape)

# Split features (42 landmarks) and labels (mudra names)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 2. Encode labels ##########################################################
le = LabelEncoder()
y = le.fit_transform(y)

# Save the encoder for real-time decoding in main script
joblib.dump(le, "label_encoder.pkl")
print("Label encoder saved as label_encoder.pkl")

# 3. Train-Test Split #######################################################
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Define and Spot-Check Models ############################################
models = {
    "MLP": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=1000,
        random_state=42
    ),
    "SVM": SVC(probability=True, random_state=42),  # Added probability tracking if you shift away from MLP
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Evaluate all architectures
for name, model in models.items():
    print(f"\n===== {name} =====")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 5. Train & Export the Best Performing Model (MLP) ########################
print("\nTraining best model (MLP with architecture 128x64)...")

# Creating a fresh reference with the explicit top-tier layout
best_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=1000,
    random_state=42
)
best_model.fit(X_train, y_train)

# Save the trained weights
joblib.dump(best_model, "best_model.pkl")
print("Model saved as best_model.pkl")

# 6. Generate Diagnostics specifically for the Best Model ###################
mlp_test_preds = best_model.predict(X_test)

# Plotting accurate confusion matrices for your presentation/report
ConfusionMatrixDisplay.from_predictions(
    y_test, 
    mlp_test_preds, 
    display_labels=le.classes_, 
    xticks_rotation='vertical'
)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Confusion matrix plot saved as confusion_matrix.png")
plt.show()

# 7. Print Class Balance Distribution #######################################
print("\n===== Class Sample Distribution =====")
df = pd.read_csv("dataset.csv", header=None)
print(df.iloc[:, -1].value_counts())