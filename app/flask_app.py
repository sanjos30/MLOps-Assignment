import mlflow
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model from the mounted path
model_path = (
    "/app/mlruns/158105573897580942/"
    "fa77e9f876e5497d908ecdb1f381749f/"
    "artifacts/model"
)
model = mlflow.sklearn.load_model(model_path)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data["features"]
    prediction = model.predict([features])
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
