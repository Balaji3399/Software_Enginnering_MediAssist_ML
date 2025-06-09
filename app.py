from flask import Flask, request, jsonify
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from collections import Counter
import pandas as pd
import numpy as np

app = Flask(__name__)

# Number of iterations per model
NUM_ITERATIONS = 5

# Symptoms list (features)
l1 = ['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach','swelled_lymph_nodes',
'malaise','blurred_and_distorted_vision','phlegm','throat_irritation','redness_of_eyes','sinus_pressure',
'runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements',
'pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps','bruising',
'obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech',
'knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints','movement_stiffness',
'spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side','loss_of_smell',
'bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching',
'toxic_look_(typhos)','depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body',
'belly_pain','abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite',
'polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances',
'receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations',
'painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling','silver_like_dusting',
'small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze']

# Global reverse disease name mapping (used for decoding predictions)
disease_reverse_mapping = {}

# Load and process training data
def load_data():
    df = pd.read_csv("training.csv")

    # Clean prognosis text
    df['prognosis'] = df['prognosis'].str.strip()

    # Dynamically map diseases
    disease_mapping = {d: i for i, d in enumerate(sorted(df['prognosis'].unique()))}
    global disease_reverse_mapping
    disease_reverse_mapping = {v: k for k, v in disease_mapping.items()}

    df['prognosis'] = df['prognosis'].map(disease_mapping)

    X = df[l1]
    y = df['prognosis'].astype(int)
    return X, y

# Load data once
X, y = load_data()

def vectorize_symptoms(symptoms):
    symptoms = list(set(symptoms))  # remove duplicates
    return [[1 if symptom in symptoms else 0 for symptom in l1]]

def run_model_multiple_times(model_class, symptoms_vector):
    predictions = []
    for _ in range(NUM_ITERATIONS):
        clf = model_class()
        clf.fit(X, y)
        prediction = clf.predict(symptoms_vector)[0]
        predictions.append(prediction)
    return predictions

def calculate_probabilities(predictions):
    total = len(predictions)
    counts = Counter(predictions)
    return {
        disease_reverse_mapping[d]: round((count / total) * 100, 2)
        for d, count in counts.items()
    }

@app.route("/")
def index():
    return "✅ Disease Prediction Flask API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    symptoms = data.get("symptoms", [])

    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    symptoms_vector = vectorize_symptoms(symptoms)

    # Run predictions
    dt_preds = run_model_multiple_times(lambda: tree.DecisionTreeClassifier(random_state=np.random.randint(1000)), symptoms_vector)
    rf_preds = run_model_multiple_times(lambda: RandomForestClassifier(random_state=np.random.randint(1000)), symptoms_vector)
    nb_preds = run_model_multiple_times(lambda: GaussianNB(), symptoms_vector)

    all_predictions = dt_preds + rf_preds + nb_preds

    return jsonify({
        "input_symptoms": symptoms,
        "predicted_probabilities": calculate_probabilities(all_predictions),
        "individual_model_results": {
            "DecisionTree": calculate_probabilities(dt_preds),
            "RandomForest": calculate_probabilities(rf_preds),
            "NaiveBayes": calculate_probabilities(nb_preds)
        },
        "total_predictions": len(all_predictions),
        "iterations_per_model": NUM_ITERATIONS
    })

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host='0.0.0.0', port=port, debug=True)

