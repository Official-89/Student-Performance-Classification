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
st.title("🎓 Student Academic Success Predictor
