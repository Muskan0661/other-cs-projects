# =============================================================
#  Assignment 3 - Machine Learning
#  Model 5: LightGBM Classifier
#  Competition: Predicting Irrigation Need (S6E4)
#
#  LightGBM (Light Gradient Boosting Machine) by Microsoft.
#  Faster than XGBoost on large datasets due to leaf-wise growth.
#  Often achieves the HIGHEST accuracy on Kaggle tabular tasks.
#  Install with: pip install lightgbm
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb


# =============================================================
# STEP 1: LOAD DATA
# =============================================================

print("Loading data...")
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
print(f"Train shape: {train.shape} | Test shape: {test.shape}")


# =============================================================
# STEP 2: PREPROCESSING
# =============================================================

X      = train.drop(columns=['id', 'Irrigation_Need'])
y      = train['Irrigation_Need']
X_test = test.drop(columns=['id'])

categorical_cols = X.select_dtypes(include='object').columns.tolist()

for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]], axis=0)
    le.fit(combined)
    X[col]      = le.transform(X[col])
    X_test[col] = le.transform(X_test[col])

target_encoder = LabelEncoder()
y_encoded  = target_encoder.fit_transform(y)
class_names = target_encoder.classes_
num_classes = len(class_names)

print(f"Classes: {class_names}")
print("Preprocessing done!")


# =============================================================
# STEP 3: K-FOLD CROSS VALIDATION
# =============================================================

print("\n--- K-Fold Cross Validation (K=5) ---")

SAMPLE_SIZE = 50000
X_sample = X.sample(n=SAMPLE_SIZE, random_state=42)
y_sample = y_encoded[X_sample.index]

# Key hyperparameters:
# num_leaves      : main complexity control (larger = more expressive, can overfit)
# learning_rate   : small = better generalization, needs more n_estimators
# feature_fraction: fraction of features per tree
# bagging_fraction: fraction of rows per tree (enables bagging)
# min_child_samples: min data in leaf (prevents tiny leaves)
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=20,
    objective='multiclass',
    num_class=num_classes,
    metric='multi_logloss',
    n_jobs=-1,
    random_state=42,
    verbose=-1
)

kf     = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_sample, y_sample, cv=kf, scoring='accuracy', n_jobs=-1)

print(f"Fold Accuracies : {[round(s, 4) for s in scores]}")
print(f"Mean Accuracy   : {scores.mean():.4f}")
print(f"Std Deviation   : {scores.std():.4f}")


# =============================================================
# STEP 4: CONFUSION MATRIX (80/20 split)
# =============================================================

X_tr, X_val, y_tr, y_val = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
)
y_pred = model.predict(X_val)

print(f"\nValidation Accuracy : {accuracy_score(y_val, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=class_names))

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'LightGBM - Confusion Matrix\nAccuracy: {accuracy_score(y_val, y_pred):.4f}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('cm_lightgbm.png', dpi=150)
plt.show()
print("Confusion matrix saved as cm_lightgbm.png")


# =============================================================
# STEP 5: FEATURE IMPORTANCE PLOT
# =============================================================

importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(20)

plt.figure(figsize=(8, 6))
top_features.plot(kind='barh', color='mediumpurple')
plt.title('LightGBM - Top 20 Feature Importances')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_lgbm.png', dpi=150)
plt.show()
print("Feature importance plot saved as feature_importance_lgbm.png")


# =============================================================
# STEP 6: RETRAIN ON FULL DATA + GENERATE SUBMISSION
# =============================================================

print("\nRetraining on full training data...")
final_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=20,
    objective='multiclass',
    num_class=num_classes,
    metric='multi_logloss',
    n_jobs=-1,
    random_state=42,
    verbose=-1
)
final_model.fit(X, y_encoded)

predictions = target_encoder.inverse_transform(final_model.predict(X_test))

submission = pd.DataFrame({'id': test['id'], 'Irrigation_Need': predictions})
submission.to_csv('submission_lightgbm.csv', index=False)

print(f"submission_lightgbm.csv saved!")
print(f"Prediction counts:\n{submission['Irrigation_Need'].value_counts()}")
