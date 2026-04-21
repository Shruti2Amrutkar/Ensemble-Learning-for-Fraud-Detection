import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_recall_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE
import pandas as pd

df = pd.read_csv(r'C:\Users\acer\Downloads\Minor\creditcard.csv')

print("Original Shape:", df.shape)

# ===============================
# 🔹 4. DATA CLEANING (IMPORTANT)
# ===============================

# Convert all values to numeric (handles hidden errors)
df = df.apply(pd.to_numeric, errors='coerce')

# Check missing values
print("\nMissing Values Before Cleaning:\n", df.isnull().sum())

# Drop rows with missing values
df = df.dropna()

# Reset index
df = df.reset_index(drop=True)

print("\nShape After Cleaning:", df.shape)

# ===============================
# 🔹 5. FEATURE & TARGET SPLIT
# ===============================

# Ensure target column exists
if 'Class' not in df.columns:
    raise ValueError("Column 'Class' not found in dataset")

X = df.drop('Class', axis=1)
y = df['Class']

# Final check (VERY IMPORTANT)
print("\nNaN in target:", y.isnull().sum())

# ===============================
# 🔹 6. SCALING
# ===============================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 🔹 7. TRAIN-TEST SPLIT
# ===============================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTrain-Test Split Done Successfully")
print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)
# ===============================
# 🔹 5. Handle Imbalance (SMOTE)
# ===============================
# Use subset for faster training
X_train_small = X_train[:50000]
y_train_small = y_train[:50000]

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train_small, y_train_small)

print("After SMOTE:", np.bincount(y_res))

# ===============================
# 🔹 6. Models
# ===============================
lr = LogisticRegression(max_iter=500)
rf = RandomForestClassifier(n_estimators=50)
dt = DecisionTreeClassifier()

# Ensemble
model = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('dt', dt)],voting='soft')

# Train
model.fit(X_res, y_res)

print("Model Trained Successfully")

# ===============================
# 🔹 7. Evaluation
# ===============================
y_prob = model.predict_proba(X_test)[:,1]

# Threshold tuning
y_pred = (y_prob > 0.3).astype(int)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# ===============================
# 🔹 8. 🔥 PREDICTION FUNCTION
# ===============================
def predict_transaction(input_data):
    # Convert to DataFrame with column names
    input_df = pd.DataFrame([input_data], columns=X.columns)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prob = model.predict_proba(input_scaled)[0][1]

    result = "FRAUD" if prob > 0.3 else "NORMAL"

    print("\nPrediction:", result)
    print("Fraud Probability:", prob)

# ===============================
# 🔹 9. TEST PREDICTION
# ===============================
# Take any row from dataset
sample = X.iloc[0].values

predict_transaction(sample)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

# ===============================
# 🔹 Create Figure
# ===============================
plt.figure(figsize=(15,5))

# ===============================
# 🔹 1. Confusion Matrix
# ===============================
plt.subplot(1,3,1)
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, cm[i][j], ha='center', va='center')

# ===============================
# 🔹 2. ROC Curve
# ===============================
plt.subplot(1,3,2)
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")

# ===============================
# 🔹 3. Precision-Recall Curve
# ===============================
plt.subplot(1,3,3)
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.plot(recall, precision)
plt.title("Precision-Recall")
plt.xlabel("Recall")
plt.ylabel("Precision")

# ===============================
# 🔹 Show All
# ===============================
plt.tight_layout()
plt.show()