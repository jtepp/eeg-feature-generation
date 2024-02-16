from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# Load the pickled machine learning model
path = os.path.dirname(os.path.realpath(__file__)) # this is the directory that the code is being run from
model_file_path = path + "/../good_rf_model.pkl"  # Replace with the actual file path
    
with open(model_file_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.get_json()
    
    # Assuming your model expects input in a certain format, preprocess the data if necessary
    



    # Perform prediction
    probabilities = model.predict_proba()
    
    # You may need to process the prediction output before returning it

    focus_avg = probabilities[:, 1::2].flatten().mean()
    
    return jsonify({'focus_probability': focus_avg})

if __name__ == '__main__':
    app.run(debug=True)
