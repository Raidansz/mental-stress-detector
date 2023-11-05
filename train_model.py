import numpy as np
from data_generation import generate_ecg_data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Generate data
not_stressed = [generate_ecg_data(100) for _ in range(100)]
stressed = [generate_ecg_data(100, stress=True) for _ in range(100)]

X = not_stressed + stressed
y = [0] * 100 + [1] * 100  # 0 for not_stressed and 1 for stressed

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = SVC()
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")
