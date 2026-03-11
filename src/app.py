from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder='templates')

# Load the model
model = joblib.load(os.path.join(os.path.dirname(__file__), "../models/churn_model.joblib"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_input = np.array([features])
    prediction = model.predict(final_input)[0]
    result = "Customer likely to Churn" if prediction == 1 else "Customer will Stay"
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
