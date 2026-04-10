from flask import Flask, request, jsonify
import joblib
import numpy as np

# Create app
app = Flask(__name__)

# Load model
model = joblib.load("../saved_models/crop_model.pkl")
le = joblib.load("../saved_models/label_encoder.pkl")

#Home route
@app.route('/')
def home():
    return "Backend is running successfully 🚀"

# API route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Extract values
    N = data['N']
    P = data['P']
    K = data['K']
    temp = data['temperature']
    humidity = data['humidity']
    ph = data['ph']
    rainfall = data['rainfall']

    # Convert to array
    features = np.array([[N, P, K, temp, humidity, ph, rainfall]])

    # Predict
    probs = model.predict_proba(features)[0]
    top3 = probs.argsort()[-3:][::-1]

    crops = le.inverse_transform(top3)
    confidence = probs[top3]

    return jsonify({
        "top_crops": crops.tolist(),
        "confidence": confidence.tolist()
    })

# Run server
if __name__ == '__main__':
    app.run(debug=True)
