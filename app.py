from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model_path = 'random_forest_model.joblib'
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json(force=True)
        # Assuming 'features' is the key for input features
        features = data['features']
        # Convert features to numpy array
        features = np.array(features).reshape(1, -1)
        
        # Make predictions
        prediction = model.predict(features)
        
        # Prepare the response
        response = {'prediction': int(prediction[0])}
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)