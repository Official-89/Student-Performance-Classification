import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

# --- 1. UI SETUP (MUST BE THE VERY FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Student Performance Predictor", layout="wide", page_icon="🎓")

# --- 2. MODEL LOADING ---
@st.cache_resource 
def load_model():
    model = XGBClassifier()
    model.load_model('student_performance_model.json')
    return model

model = load_model()

# --- 3. MODEL COLUMNS ---
model_columns = [
    'gender_female', 'gender_male', 
    'race/ethnicity_group A', 'race/ethnicity_group B', 'race/ethnicity_group C', 'race/ethnicity_group D', 'race/ethnicity_group E',
    'parental level of education_associate\'s degree', 'parental level of education_bachelor\'s degree', 
    'parental level of education_high school', 'parental level of education_master\'s degree', 
    'parental level of education_some college', 'parental level of education_some high school',
    'lunch_free/reduced', 'lunch_standard', 
    'test preparation course_completed', 'test preparation course_none'
]

# --- 4. SIDEBAR INPUTS ---
st.sidebar.title("⚙️ Input Parameters")
st.sidebar.write("Adjust the student details below:")

math = st.sidebar.slider("Math Score", 0, 100, 50)
reading = st.sidebar.slider("Reading Score", 0, 100, 50)
writing = st.sidebar.slider("Writing Score", 0, 100, 50)

st.sidebar.markdown("---")
st.sidebar.subheader("Demographics")
gender = st.sidebar.selectbox("Gender", ["female", "male"])
race = st.sidebar.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
education = st.sidebar.selectbox("Parental Education", [
    "some high school", "high school", "some college", 
    "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.sidebar.selectbox("Lunch Type", ["standard", "free/reduced"])
prep = st.sidebar.selectbox("Test Prep Course", ["none", "completed"])

# Calculate Average
avg_score = (math + reading + writing) / 3

# --- 5. MAIN DASHBOARD ---
st.title("🎓 Student Academic Success Predictor")
st.markdown("Predict student outcomes and understand the driving factors using our XGBoost Classification Pipeline.")

# KPI Metrics
st.subheader("Current Academic Standing")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Math Score", math)
col2.metric("Reading Score", reading)
col3.metric("Writing Score", writing)
col4.metric("Calculated Average", f"{avg_score:.2f}", delta="Pass" if avg_score >= 40 else "Fail", delta_color="normal" if avg_score >= 40 else "inverse")

st.markdown("---")

# --- 6. PREDICTION LOGIC ---
if st.button("Predict Student Outcome", use_container_width=True):
    # Prepare data
    input_df = pd.DataFrame({
        'gender': [gender], 'race/ethnicity': [race],
        'parental level of education': [education], 'lunch': [lunch],
        'test preparation course': [prep]
    })
    
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    # Predict
    res = model.predict(input_encoded)
    prob = model.predict_proba(input_encoded)[0][1] * 100
    
    # Display Professional Status
    st.markdown("### Prediction Results")
    
    if res[0] == 1:
        st.success("Outcome: **PASS**")
        st.info(f"**Model Certainty:** {prob:.1f}%")
    else:
        st.error("Outcome: **FAIL**")
        st.info(f"**Model Certainty:** {100-prob:.1f}%")
    
    st.markdown("---")
    
    # Visualizations
    graph_col1, graph_col2 = st.columns(2)
    
    with graph_col1:
        st.subheader("📊 Performance vs Threshold")
        fig, ax = plt.subplots(figsize=(6, 4))
        scores_data = pd.DataFrame({
            'Subject': ['Math', 'Reading', 'Writing', 'Total Average'],
            'Score': [math, reading, writing, avg_score]
        })
        sns.barplot(x='Subject', y='Score', data=scores_data, palette="Blues_d", ax=ax, hue='Subject', legend=False)
        ax.axhline(40, ls='--', color='red', label='Passing Threshold (40)')
        plt.ylim(0, 105)
        st.pyplot(fig)

    with graph_col2:
        st.subheader("🧠 Feature Importance")
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'Feature': model_columns, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=False).head(5)
        feat_df['Feature'] = feat_df['Feature'].str.replace('_', ': ').str.title()
        
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Importance', y='Feature',
