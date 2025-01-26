import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply scaling to avoid potential data leakage
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define experiment in MLflow
mlflow.set_experiment("Iris Classifier Experiment")

# Hyperparameters for experimentation
param_grid = [
    {"n_estimators": 10, "max_depth": 3},
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": None},
]

# Run experiments for different parameter combinations
for params in param_grid:
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)

        # Train model
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log model with input example
        input_example = pd.DataFrame(X_test, columns=data.feature_names)
        # Log and register the model
        model_name = "IrisClassifier"
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            registered_model_name=model_name
        )

        print(f"Model registered as: {model_name}")

        # Debugging information
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Parameters: {params}")
        print(f"Accuracy: {accuracy}")
