from flask import Flask, request, jsonify
from joblib import load
model = load ('logistic_regression_model.pkl')
app = Flask(__name__)
@app.route('/predict', methods = ['POST'])
def predict ():
    data = request.get_json()
    prediction = model.predict([data['features']])
    return jsonify({'prediction': int (prediction [0])})
  if __name__ == '__main__':
    app.run (debug=True)
