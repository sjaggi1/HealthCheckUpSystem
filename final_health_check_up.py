from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained models
anemia_model = joblib.load('stacking_anemia_model.pkl')
heart_model = joblib.load('best_random_forest_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/bmi')
def bmi():
    return render_template('bmi.html')

@app.route('/anemia')
def anemia():
    return render_template('anemia.html')

@app.route('/heart')
def heart():
    return render_template('heart.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/calculate_bmi', methods=['POST'])
def calculate_bmi():
    weight = float(request.form['weight'])
    height = float(request.form['height'])
    bmi = weight / (height ** 2)
    return render_template('bmi.html', prediction_text=f'Your BMI is {round(bmi, 2)}')

@app.route('/predict_anemia', methods=['POST'])
def predict_anemia():
    features = [float(request.form[x]) for x in ['Age', 'Sex', 'RBC', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'TLC', 'PLT', 'HGB']]
    final_features = np.array(features).reshape(1, -1)
    
    prediction = anemia_model.predict(final_features)[0]
    probability = anemia_model.predict_proba(final_features)[0][prediction]
    
    output = "Anemia Detected" if prediction == 1 else "No Anemia Detected"
    confidence = round(probability * 100, 2)

    return render_template('anemia.html', prediction_text=f'Prediction: {output} (Confidence: {confidence}%)')

@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    gender = 1 if request.form['Gender'] == 'Male' else 0
    exercise = 1 if request.form['Exercise_Habits'] == 'High' else 0
    smoking = 1 if request.form['Smoking'] == 'Yes' else 0
    family = 1 if request.form['Family_Heart_Disease'] == 'Yes' else 0
    diabetes = 1 if request.form['Diabetes'] == 'Yes' else 0
    bmi = float(request.form['BMI'])   # <-- corrected: take BMI from form and convert to float
    high_bp = 1 if request.form['High_BP'] == 'Yes' else 0
    low_hdl = 1 if request.form['Low_HDL'] == 'Yes' else 0
    high_ldl = 1 if request.form['High_LDL'] == 'Yes' else 0

    alcohol_map = {'Low': 0, 'Medium': 1, 'High': 2}
    stress_map = {'Low': 0, 'Medium': 1, 'High': 2}
    sugar_map = {'Low': 0, 'Medium': 1, 'High': 2}

    features = [
        int(request.form['Age']),
        gender,
        int(request.form['Blood_Pressure']),
        int(request.form['Cholesterol']),
        exercise,
        smoking,
        family,
        diabetes,
        bmi,   # <-- corrected: comma instead of semicolon
        high_bp,
        low_hdl,
        high_ldl,
        alcohol_map[request.form['Alcohol']],
        stress_map[request.form['Stress']],
        float(request.form['Sleep_Hours']),
        sugar_map[request.form['Sugar']],
        int(request.form['Triglyceride']),
        int(request.form['Fasting_Blood_Sugar']),
        float(request.form['CRP']),
        float(request.form['Homocysteine']),
    ]

    final_features = np.array(features).reshape(1, -1)
    prediction = heart_model.predict(final_features)[0]
    probability = heart_model.predict_proba(final_features)[0][prediction]

    output = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"
    confidence = round(probability * 100, 2)

    return render_template('heart.html', prediction_text=f'Prediction: {output}')
# (Confidence: {confidence}%)
if __name__ == "__main__":
    app.run(debug=True)
