from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model, scaler, and label encoder
model = joblib.load('water_quality_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from the form
        features = [float(x) for x in request.form.values()]
        features = np.array(features).reshape(1, -1)
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Predict the class
        prediction = model.predict(features_scaled)
        
        # Decode the label
        predicted_class = label_encoder.inverse_transform(prediction)[0]
        
        return render_template('index.html', prediction_text=f'The predicted water quality is: {predicted_class}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
