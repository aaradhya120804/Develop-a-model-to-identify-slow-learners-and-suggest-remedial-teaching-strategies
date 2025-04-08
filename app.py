# app.py - Streamlit Dashboard Code

import streamlit as st
import pandas as pd
import joblib
import os
import warnings

# --- Page Configuration ---
st.set_page_config(
    page_title="Slow Learner Prediction Tool",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
SCALER_FILENAME = 'scaler.joblib'
MODEL_FILENAME = 'random_forest_model.joblib'
FEATURE_NAMES = ['Standardized_Test_Score', 'Average_Grade', 'Attendance_Percentage', 'Times_Late', 'Participation_Rating']

# --- Load Model and Scaler ---
@st.cache_resource
def load_artifacts(scaler_path, model_path):
    try:
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)
        return scaler, model
    except FileNotFoundError:
        st.error(f"Error: Could not find '{SCALER_FILENAME}' or '{MODEL_FILENAME}'. Ensure they are in the app directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None

script_dir = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(script_dir, SCALER_FILENAME)
model_path = os.path.join(script_dir, MODEL_FILENAME)
scaler, final_model = load_artifacts(scaler_path, model_path)

# --- Helper Functions ---
def predict_slow_learner(data, scaler_obj, model_obj):
    if scaler_obj is None or model_obj is None:
        return None, None
    input_data = pd.DataFrame([data], columns=FEATURE_NAMES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        input_data_scaled = scaler_obj.transform(input_data)
        prediction = model_obj.predict(input_data_scaled)[0]
        probability = model_obj.predict_proba(input_data_scaled)[0][1]
    return prediction, probability

def get_remedial_suggestions(prediction, probability, standardized_test_score, average_grade, attendance_percentage, participation_rating):
    suggestions = []
    if prediction == 1:
        suggestions.append(f"**Student identified as potentially needing support (Probability: {probability:.2%})**")
        suggestions.append("---")
        suggestions.append("**Recommended Actions:**")
        suggestions.append("* Schedule a one-on-one meeting to discuss challenges and learning style.")
        suggestions.append("* Provide simplified explanations and break down complex topics.")
        suggestions.append("* Offer extra practice exercises tailored to weak areas.")
        suggestions.append("* Utilize visual aids and hands-on activities.")
        suggestions.append("* Encourage questions in a supportive environment.")
        suggestions.append("* Recommend peer tutoring or study groups.")
        suggestions.append("\n**Observations & Targeted Suggestions:**")
        triggered_specific = False
        if standardized_test_score < 45:
            suggestions.append("* *Observation:* Low Standardized Test Score.")
            suggestions.append("    * *Suggestion:* Focus on foundational concepts for test topics.")
            triggered_specific = True
        if average_grade < 70:
            suggestions.append("* *Observation:* Low Average Grade.")
            suggestions.append("    * *Suggestion:* Review recent assignments for weaknesses.")
            suggestions.append("    * *Suggestion:* Offer re-assessment or extra credit opportunities.")
            triggered_specific = True
        if attendance_percentage < 85:
            suggestions.append("* *Observation:* Low Attendance.")
            suggestions.append("    * *Suggestion:* Discuss barriers to attendance.")
            triggered_specific = True
        if participation_rating <= 2:
            suggestions.append("* *Observation:* Low Class Participation.")
            suggestions.append("    * *Suggestion:* Create low-pressure participation opportunities.")
            suggestions.append("    * *Suggestion:* Reinforce participation attempts positively.")
            triggered_specific = True
        if not triggered_specific:
            suggestions.append("* No specific low metrics triggered; focus on general strategies.")
        suggestions.append("\n---")
        suggestions.append("**Follow Up:** Monitor progress and adjust strategies as needed.")
    return suggestions

# --- Streamlit App UI ---
st.title("🎓 Slow Learner Prediction Tool")
st.subheader("Identify students who might need additional support")

# Sidebar: About and Input Form
st.sidebar.header("About")
st.sidebar.info(
    "This tool uses a Random Forest model to predict if a student might need additional support "
    "based on academic and behavioral data. Enter the student’s details and click 'Predict'."
)

st.sidebar.header("Input Student Details")
with st.sidebar.form(key="input_form"):
    score = st.number_input(
        "Standardized Test Score",
        min_value=0.0, max_value=100.0, value=50.0, step=0.5, format="%.1f",
        help="Student’s most recent standardized test score (0-100)."
    )
    grade = st.number_input(
        "Average Grade (%)",
        min_value=0.0, max_value=100.0, value=75.0, step=0.5, format="%.1f",
        help="Student’s current average grade percentage (0-100)."
    )
    attendance = st.number_input(
        "Attendance Percentage (%)",
        min_value=0.0, max_value=100.0, value=90.0, step=0.5, format="%.1f",
        help="Student’s attendance rate (0-100)."
    )
    late = st.number_input(
        "Times Late to Class",
        min_value=0, value=2, step=1,
        help="Number of times the student has been late recently."
    )
    participation = st.slider(
        "Participation Rating",
        min_value=1, max_value=5, value=3, step=1,
        help="Rate the student’s class participation (1=Very Low, 5=Very High)."
    )
    predict_button = st.form_submit_button("✨ Predict Support Need", type="primary")

# Main Area: Prediction Results
st.header("Prediction Results")
if scaler is None or final_model is None:
    st.warning("Model artifacts not loaded. Please check file paths and logs.")
elif predict_button:
    with st.spinner("Analyzing student data..."):
        input_features = {
            'Standardized_Test_Score': score,
            'Average_Grade': grade,
            'Attendance_Percentage': attendance,
            'Times_Late': late,
            'Participation_Rating': participation
        }
        prediction, probability = predict_slow_learner(input_features, scaler, final_model)

    if prediction is not None:
        # Display prediction status
        if prediction == 1:
            st.warning("🚨 **Potential Need for Support Identified**")
        else:
            st.success("✅ **No Specific Support Need Identified by Model**")
        
        # Display probability
        st.metric("Model Confidence (Probability of Needing Support)", f"{probability:.1%}")

        # Display suggestions if available
        suggestions = get_remedial_suggestions(prediction, probability, score, grade, attendance, participation)
        if suggestions:
            with st.expander("💡 Recommended Remedial Suggestions", expanded=True):
                for suggestion in suggestions:
                    st.markdown(suggestion)
    else:
        st.error("Prediction could not be made. Check artifact loading.")
else:
    st.info("Enter student details in the sidebar and click 'Predict Support Need' to view results.")

# Footer
st.markdown("---")
st.caption("**Disclaimer:** This tool provides suggestions based on a predictive model. Always use professional judgment and consider individual student circumstances.")