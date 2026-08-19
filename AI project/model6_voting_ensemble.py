# =============================================================
#  Assignment 3 - Machine Learning
#  Model 6: Voting Ensemble (RF + XGBoost + LightGBM)
#  Competition: Predicting Irrigation Need (S6E4)
#
#  Ensembling combines predictions from multiple diverse models.
#  This often achieves HIGHER accuracy than any single model.
#  "Soft voting" averages probability outputs → more nuanced.
#  Install: pip install xgboost lightgbm
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
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
# STEP 3: DEFINE BASE MODELS
# =============================================================

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)

xgb_model = XGBClassifier(
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

lgbm_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=20,
    objective='multiclass',
    num_class=num_classes,
    n_jobs=-1,
    random_state=42,
    verbose=-1
)

# Soft voting: each model outputs class probabilities; final prediction
# is the class with the highest AVERAGE probability across all models.
ensemble = VotingClassifier(
    estimators=[
        ('rf',   rf_model),
        ('xgb',  xgb_model),
        ('lgbm', lgbm_model)
    ],
    voting='soft',
    n_jobs=-1
)


# =============================================================
# STEP 4: K-FOLD CROSS VALIDATION
# =============================================================

print("\n--- K-Fold Cross Validation (K=5) ---")
print("(Using 30,000-row sample for speed — ensemble is slower to train)")

SAMPLE_SIZE = 30000
X_sample = X.sample(n=SAMPLE_SIZE, random_state=42)
y_sample = y_encoded[X_sample.index]

kf     = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(ensemble, X_sample, y_sample, cv=kf, scoring='accuracy', n_jobs=-1)

print(f"Fold Accuracies : {[round(s, 4) for s in scores]}")
print(f"Mean Accuracy   : {scores.mean():.4f}")
print(f"Std Deviation   : {scores.std():.4f}")


# =============================================================
# STEP 5: CONFUSION MATRIX (80/20 split)
# =============================================================

X_tr, X_val, y_tr, y_val = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

ensemble.fit(X_tr, y_tr)
y_pred = ensemble.predict(X_val)

print(f"\nValidation Accuracy : {accuracy_score(y_val, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=class_names))

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Voting Ensemble (RF+XGB+LGBM) - Confusion Matrix\nAccuracy: {accuracy_score(y_val, y_pred):.4f}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('cm_voting_ensemble.png', dpi=150)
plt.show()
print("Confusion matrix saved as cm_voting_ensemble.png")


# =============================================================
# STEP 6: RETRAIN ON FULL DATA + GENERATE SUBMISSION
# =============================================================

print("\nRetraining ensemble on full training data (this takes a few minutes)...")
final_ensemble = VotingClassifier(
    estimators=[
        ('rf',   RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42)),
        ('xgb',  XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                                subsample=0.8, colsample_bytree=0.8,
                                objective='multi:softmax', num_class=num_classes,
                                eval_metric='mlogloss', use_label_encoder=False,
                                n_jobs=-1, random_state=42)),
        ('lgbm', lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                                     feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                                     min_child_samples=20, objective='multiclass',
                                     num_class=num_classes, n_jobs=-1, random_state=42, verbose=-1))
    ],
    voting='soft',
    n_jobs=-1
)
final_ensemble.fit(X, y_encoded)

predictions = target_encoder.inverse_transform(final_ensemble.predict(X_test))

submission = pd.DataFrame({'id': test['id'], 'Irrigation_Need': predictions})
submission.to_csv('submission_voting_ensemble.csv', index=False)

print(f"submission_voting_ensemble.csv saved!")
print(f"Prediction counts:\n{submission['Irrigation_Need'].value_counts()}")
