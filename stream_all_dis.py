import streamlit as st
import joblib
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from matplotlib import pyplot as plt

# Load trained models
diabetes_classifier = joblib.load('stacked_diabetes_model.pkl')
stacking_classifier_heart = joblib.load('stacked_heart_model.pkl')
stacking_classifier_lung = joblib.load('stacking_cancer_model.pkl')



# make diabetes explainer
X_train_diabetes = pd.read_pickle('X_train_diabetes.pkl')
y_train_diabetes = pd.read_pickle('y_train_diabetes.pkl')
diabetes_class_names = ['No Diabetes', 'Diabetes']
diabetes_explainer = LimeTabularExplainer(
    training_data=X_train_diabetes.values,
    training_labels=y_train_diabetes.values,
    feature_names=X_train_diabetes.columns,
    class_names=diabetes_class_names,
    mode='classification',
    random_state=42
)

diabetes_feature_names = ['BMI', 'Pregnancies', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction', 'Glucose',
                          'Age', 'BloodPressure']

X_train_lung = pd.read_pickle('X_train_lung.pkl')
y_train_lung = pd.read_pickle('y_train_lung.pkl')
lung_class_names = ['No Lung cancer', 'Lung Cancer']
lung_explainer = LimeTabularExplainer(
    training_data=X_train_lung.values,
    training_labels=y_train_lung.values,
    feature_names=X_train_lung.columns,
    class_names=lung_class_names,
    mode='classification',
    random_state=42
)

lung_feature_names = ['ALLERGY ', 'PEER_PRESSURE', 'ALCOHOL CONSUMING', 'AGE', 'SMOKING',
                      'CHRONIC DISEASE', 'SWALLOWING DIFFICULTY', 'WHEEZING', 'FATIGUE ',
                      'YELLOW_FINGERS', 'ANXIETY', 'SHORTNESS OF BREATH', 'COUGHING']

X_train_heart = pd.read_pickle('X_train_heart.pkl')
y_train_heart = pd.read_pickle('y_train_heart.pkl')
heart_class_names = ['No Heart Disease', 'Heart Disease']
heart_explainer = LimeTabularExplainer(
    training_data=X_train_heart.values,
    training_labels=y_train_heart.values,
    feature_names=X_train_heart.columns,
    class_names=heart_class_names,
    mode='classification',
    random_state=42
)

heart_feature_names = ['ST slope', 'max heart rate', 'cholesterol', 'age', 'resting bps',
                       'oldpeak', 'chest pain type', 'resting ecg', 'sex', 'exercise angina',
                       'fasting blood sugar']


# Preprocessing and prediction functions for Diabetes
def preprocess_input_diabetes(*args):
    return args


def predict_diabetes(*args):
    preprocessed_data = preprocess_input_diabetes(*args)
    explanation = diabetes_explanation(preprocessed_data)
    return diabetes_classifier.predict([preprocessed_data]), explanation


def diabetes_explanation(instance):
    series = pd.Series(instance, index=diabetes_feature_names)
    exp = diabetes_explainer.explain_instance(data_row=series, predict_fn=diabetes_classifier.predict_proba, top_labels=2)
    return exp


# Preprocessing and prediction functions for Heart Disease
def preprocess_input_heart(age, sex, *rest):
    sex_encoded = 1 if sex == 'M' else 0
    return (age, sex_encoded) + rest


def predict_heart_disease(age, sex, *rest):
    preprocessed_data = preprocess_input_heart(age, sex, *rest)
    explanation = heart_explanation(preprocessed_data)
    return stacking_classifier_heart.predict([preprocessed_data]), explanation


def heart_explanation(instance):
    series = pd.Series(instance, index=heart_feature_names)
    exp = heart_explainer.explain_instance(data_row=series, predict_fn=stacking_classifier_heart.predict_proba, top_labels=2)
    return exp


# Preprocessing and prediction functions for Lung Cancer
def preprocess_input_lung(age, *args):
    encoded = [1 if x == 'No' else (2 if x == 'Yes' else None) for x in args]
    if all(val is None for val in encoded):
        encoded = [1] * len(args)
    return (age,) + tuple(encoded)


def predict_lung_cancer(age, *args):
    preprocessed_data = preprocess_input_lung(age, *args)
    # have to rerun code to make classifier with svc(probability=True)
    explanation = lung_explanation(preprocessed_data)
    return stacking_classifier_lung.predict([preprocessed_data]), explanation


def lung_explanation(instance):
    series = pd.Series(instance, index=lung_feature_names)
    exp = lung_explainer.explain_instance(data_row=series, predict_fn=stacking_classifier_lung.predict_proba, top_labels=2)
    return exp


# Streamlit app
def main():
    st.markdown('<p style="font-size:20px;">⚠️ <strong>Warning:</strong> Please do not use as a doctor advice.</p>',
                unsafe_allow_html=True)
    st.title('Medical Prediction Systems')
    tab1, tab2, tab3= st.tabs(
        ["Diabetes Prediction", "Heart Disease Prediction", "Lung Cancer Prediction"])

    with tab1:
        st.header("Should I see an endocrinologist?")
        col1, col2, col3 = st.columns(3)
        with col1:
            Pregnancies = st.number_input('Pregnancies', min_value=0, step=1)
            Glucose = st.number_input('Glucose', min_value=0)
            BloodPressure = st.number_input('Blood Pressure', min_value=0)
        with col2:
            SkinThickness = st.number_input('Skin Thickness', min_value=0)
            Insulin = st.number_input('Insulin', min_value=0)
            BMI = st.number_input('BMI', min_value=0.0)
        with col3:
            DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0)
            Age = st.number_input('Age', min_value=0, step=1)
        if st.button('Predict Diabetes'):
            result, explanation = predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
                                                   DiabetesPedigreeFunction, Age)
            st.write('Please visit the endocrinologist' if result[0] == 1 else 'No need to see the endocrinologist')

            st.write('Explanations for prediction:')
            st.pyplot(explanation.as_pyplot_figure(result[0]))
            plt.clf()
    with tab2:
        st.header("Should I see a cardiologist?")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input('Age (Heart)', min_value=0, max_value=150)
            sex = st.selectbox('Sex', ['M', 'F'])
            chest_pain_type = st.number_input('Chest Pain Type', min_value=1, max_value=4)
            ST_slope = st.number_input('ST Slope', min_value=0, max_value=3)
        with col2:
            resting_bps = st.number_input('Resting BP', min_value=0)
            cholesterol = st.number_input('Cholesterol', min_value=0)
            fasting_blood_sugar = st.number_input('Fasting Blood Sugar', min_value=0, max_value=1)
            oldpeak = st.number_input('Oldpeak', min_value=0.0)
        with col3:
            resting_ecg = st.number_input('Resting ECG', min_value=0, max_value=2)
            max_heart_rate = st.number_input('Max Heart Rate', min_value=0)
            exercise_angina = st.number_input('Exercise Angina', min_value=0, max_value=1)
        if st.button('Predict Heart Disease'):
            result, explanation = predict_heart_disease(age, sex, chest_pain_type, resting_bps, cholesterol,
                                                        fasting_blood_sugar, resting_ecg, max_heart_rate,
                                                        exercise_angina, oldpeak, ST_slope)
            st.write('Please visit the cardiologist' if result[0] == 1 else 'No need to see the cardiologist')

            st.write('Explanations for prediction:')
            st.pyplot(explanation.as_pyplot_figure(result[0]))
            plt.clf()

    with tab3:
        st.header("Should I see an oncologist?")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.selectbox('Age (Lung)', list(range(121)))
            smoking = st.radio('Smoking', ['Yes', 'No'], index=1)
            yellow_fingers = st.radio('Yellow Fingers', ['Yes', 'No'], index=1)
            anxiety = st.radio('Anxiety', ['Yes', 'No'], index=1)
        with col2:
            peer_pressure = st.radio('Peer Pressure', ['Yes', 'No'], index=1)
            chronic_disease = st.radio('Chronic Disease', ['Yes', 'No'], index=1)
            fatigue = st.radio('Fatigue', ['Yes', 'No'], index=1)
            allergy = st.radio('Allergy', ['Yes', 'No'], index=1)
        with col3:
            wheezing = st.radio('Wheezing', ['Yes', 'No'], index=1)
            alcohol_consuming = st.radio('Alcohol Consuming', ['Yes', 'No'], index=1)
            coughing = st.radio('Coughing', ['Yes', 'No'], index=1)
            shortness_of_breath = st.radio('Shortness of Breath', ['Yes', 'No'], index=1)
            swallowing_difficulty = st.radio('Swallowing Difficulty', ['Yes', 'No'], index=1)
        if st.button('Predict Lung Cancer'):
            result, explanation = predict_lung_cancer(age, smoking, yellow_fingers, anxiety, peer_pressure,
                                                      chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
                                                      coughing, shortness_of_breath, swallowing_difficulty)
            st.write('Please visit the oncologist' if result[0] == 1 else 'No need to see the oncologist')

            st.write('Explanations for prediction:')
            st.pyplot(explanation.as_pyplot_figure(result[0]))
            plt.clf()

if __name__ == "__main__":
    main()
