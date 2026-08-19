# =============================================================
#  Assignment 3 - Machine Learning
#  Model 4: XGBoost Classifier
#  Competition: Predicting Irrigation Need (S6E4)
#
#  XGBoost (Extreme Gradient Boosting) builds trees sequentially,
#  each correcting the errors of the previous one. It is consistently
#  one of the top performers on structured/tabular Kaggle datasets.
#  Install with: pip install xgboost
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
from xgboost import XGBClassifier


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

# XGBoost works well without scaling (tree-based)
# But we encode the target as integers 0, 1, 2 ...
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
# n_estimators   : number of boosting rounds
# learning_rate  : shrinks contribution of each tree (lower = safer but slower)
# max_depth      : depth of each tree (3-6 is typical)
# subsample      : fraction of rows per tree (adds randomness, reduces overfit)
# colsample_bytree: fraction of features per tree
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    num_class=num_classes,
    eval_metric='mlogloss',
    use_label_encoder=False,
    n_jobs=-1,
    random_state=42
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

model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
y_pred = model.predict(X_val)

print(f"\nValidation Accuracy : {accuracy_score(y_val, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=class_names))

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'XGBoost - Confusion Matrix\nAccuracy: {accuracy_score(y_val, y_pred):.4f}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('cm_xgboost.png', dpi=150)
plt.show()
print("Confusion matrix saved as cm_xgboost.png")


# =============================================================
# STEP 5: FEATURE IMPORTANCE PLOT
# =============================================================

importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(20)

plt.figure(figsize=(8, 6))
top_features.plot(kind='barh', color='steelblue')
plt.title('XGBoost - Top 20 Feature Importances')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_xgb.png', dpi=150)
plt.show()
print("Feature importance plot saved as feature_importance_xgb.png")


# =============================================================
# STEP 6: RETRAIN ON FULL DATA + GENERATE SUBMISSION
# =============================================================

print("\nRetraining on full training data...")
final_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    num_class=num_classes,
    eval_metric='mlogloss',
    use_label_encoder=False,
    n_jobs=-1,
    random_state=42
)
final_model.fit(X, y_encoded)

predictions = target_encoder.inverse_transform(final_model.predict(X_test))

submission = pd.DataFrame({'id': test['id'], 'Irrigation_Need': predictions})
submission.to_csv('submission_xgboost.csv', index=False)

print(f"submission_xgboost.csv saved!")
print(f"Prediction counts:\n{submission['Irrigation_Need'].value_counts()}")
