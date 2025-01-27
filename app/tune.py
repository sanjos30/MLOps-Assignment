import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X = data.data
y = data.target


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define parameter grid for tuning
param_grid = {
    "n_estimators": [10, 50, 100],
    "max_depth": [3, 5, None],
    "min_samples_split": [2, 5, 10],
}

# Initialize model
model = RandomForestClassifier(random_state=42)


# Perform grid search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy"
)
grid_search.fit(X_train, y_train)

# Get best parameters and score
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log results to MLflow
mlflow.set_experiment("Iris Classifier Tuning")
with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(best_model, artifact_path="model")

# Output results
print(f"Best Parameters: {best_params}")
print(f"Test Accuracy: {accuracy}")
