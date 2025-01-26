import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Define experiment
mlflow.set_experiment("Iris Classifier Experiment")

# Start MLflow run
with mlflow.start_run():
    # Define parameters
    params = {"n_estimators": 22, "max_depth": 9}
    model = RandomForestClassifier(**params)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log parameters, metrics, and the model
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    print(f"Run ID: {mlflow.active_run().info.run_id}")
