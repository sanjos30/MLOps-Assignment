FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the application files
COPY . /app

# Install dependencies
RUN pip install flask mlflow scikit-learn

# Expose the Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app/flask_app.py"]
