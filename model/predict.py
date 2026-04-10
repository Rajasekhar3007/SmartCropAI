import joblib
import numpy as np

model = joblib.load("../saved_models/crop_model.pkl")
le = joblib.load("../saved_models/label_encoder.pkl")

def predict_crop(N, P, K, temp, humidity, ph, rainfall):
    data = np.array([[N, P, K, temp, humidity, ph, rainfall]])

    probs = model.predict_proba(data)[0]
    top3 = probs.argsort()[-3:][::-1]

    crops = le.inverse_transform(top3)
    confidence = probs[top3]

    return crops, confidence

# Test
crops, conf = predict_crop(67,59,41,21.94766735,80.97384195,6.012632591,213.3560921)

print("Top 3 Crops:", crops)
print("Confidence:", conf)
