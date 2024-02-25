from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import pandas as pd
import numpy as np
from src.EEG_testing_features import gen_testing_matrix

app = Flask(__name__)
# CORS(app)
CORS(app)

# Load the pickled machine learning model
path = os.path.dirname(os.path.realpath(__file__)) # this is the directory that the code is being run from
model_file_path = path + "/models/60_confidence_rf_model_no_c.pkl"  # Replace with the actual file path
    
with open(model_file_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.get_json()
    data = pd.json_normalize(data)
    data = np.array(data).astype(float)
    print(data)
    # Assuming your model expects input in a certain format, preprocess the data if necessary
    
    matrix = gen_testing_matrix(data)

    # X = data['data']

    # # Perform prediction
    # probabilities = model.predict_proba(X)
    
    # # You may need to process the prediction output before returning it

    # focus_avg = probabilities[:, 1::2].flatten().mean()
    
    # return jsonify({'focus_probability': focus_avg})
    return jsonify({"result": matrix.to_json(orient="values")})

if __name__ == '__main__':
    app.run(debug=True)
