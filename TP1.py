from flask import Flask, request, jsonify
from joblib import load
model = load ('logistic_regression_model.pkl')
app = Flask(__name__)
@app.route('/predict', methods = ['POST'])
def predict ():
    data = request.get_json()
    prediction = model.predict([data['features']])
    return jsonify({'prediction': int (prediction [0])})
app.run (debug=True)
