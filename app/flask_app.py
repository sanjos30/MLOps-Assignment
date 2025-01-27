import pickle
from flask import Flask, request, jsonify
import mlflow.sklearn

app = Flask(__name__)

# Load the model from MLflow
model = mlflow.sklearn.load_model("runs:/fa77e9f876e5497d908ecdb1f381749f/model") 

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data["features"]
    prediction = model.predict([features])
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
