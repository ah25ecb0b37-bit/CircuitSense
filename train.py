import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("paste.txt.csv")
X = df.drop(columns=["label"]).values
y = df["label"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

AUGMENT_PER_SAMPLE = 200
np.random.seed(42)

X_aug_list = [X]
y_aug_list = [y_enc]

for _ in range(AUGMENT_PER_SAMPLE):
    noise_std = np.random.uniform(0.01, 0.05)
    noise = np.random.normal(0, noise_std, X.shape)
    scale = np.random.uniform(0.95, 1.05, (X.shape[0], 1))
    shift = np.random.randint(-5, 6)
    X_shifted = np.roll(X, shift, axis=1)
    X_aug_list.append((X_shifted + noise) * scale)
    y_aug_list.append(y_enc)

X_aug = np.vstack(X_aug_list)
y_aug = np.concatenate(y_aug_list)

def add_features(X):
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    maxv = X.max(axis=1, keepdims=True)
    minv = X.min(axis=1, keepdims=True)
    rng = maxv - minv
    energy = (X ** 2).mean(axis=1, keepdims=True)
    skew = (((X - mean) ** 3).mean(axis=1, keepdims=True)) / (std ** 3 + 1e-8)
    kurt = (((X - mean) ** 4).mean(axis=1, keepdims=True)) / (std ** 4 + 1e-8)
    return np.hstack([X, mean, std, maxv, minv, rng, energy, skew, kurt])

X_aug = add_features(X_aug)

print(f"Augmented dataset: {X_aug.shape[0]} samples, {X_aug.shape[1]} features")
print(f"Classes: {le.classes_}")

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_aug, y_aug, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"\n5-Fold CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

X_train, X_test, y_train, y_test = train_test_split(
    X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

all_labels = np.arange(len(le.classes_))

print(f"\nTest Accuracy: {accuracy_score(y_test, pred):.4f}")
print("\nClassification Report:")
print(classification_report(
    y_test,
    pred,
    labels=all_labels,
    target_names=le.classes_,
    zero_division=0
))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred, labels=all_labels))

model.fit(X_aug, y_aug)
joblib.dump(model, "model.joblib")
joblib.dump(le, "label_encoder.joblib")

results = {
    "model": "RandomForest + shift/noise/scale augmentation + engineered stats",
    "cv_accuracy": float(scores.mean()),
    "test_accuracy": float(accuracy_score(y_test, pred)),
    "classes": le.classes_.tolist(),
    "original_samples": int(len(df)),
    "augmented_samples": int(len(X_aug))
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nDone. Saved model.joblib, label_encoder.joblib, results.json")