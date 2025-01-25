from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the CI/CD pipeline demo!"})


@app.route("/status")
def status():
    return jsonify({"status": "Application is running"})


if __name__ == "__main__":
    app.run(debug=True)
