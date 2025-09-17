import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Load preprocessed data
X_train = joblib.load("X_train.pkl")
X_val   = joblib.load("X_val.pkl")
X_test  = joblib.load("X_test.pkl")
y_train = joblib.load("y_train.pkl")
y_val   = joblib.load("y_val.pkl")
y_test  = joblib.load("y_test.pkl")

# Hyperparameter tuning for K
param_grid = {"n_neighbors": list(range(1, 31))}
knn = KNeighborsClassifier()

grid_search = GridSearchCV(knn, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

print("Best K:", grid_search.best_params_["n_neighbors"])
print("Best CV accuracy:", grid_search.best_score_)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=grid_search.best_params_["n_neighbors"])
knn.fit(X_train, y_train)

# Evaluate
train_acc = accuracy_score(y_train, knn.predict(X_train))
val_acc   = accuracy_score(y_val, knn.predict(X_val))
test_acc  = accuracy_score(y_test, knn.predict(X_test))

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

joblib.dump(knn, "knn_model.pkl")
print("KNN model saved as knn_model.pkl")