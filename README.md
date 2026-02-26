# 🎓 Student Academic Success Predictor

### 🚀 [Launch the Live Predictor UI](https://student-performance-classification-wem3abymftd2ftghajaofg.streamlit.app/)

An interactive machine learning dashboard powered by **XGBoost** to predict student academic success (Pass/Fail) based on socioeconomic and demographic factors. This project demonstrates a complete end-to-end data science pipeline, from data preprocessing to web deployment.

---

## 🤝 Project Team

| NAME | REGISTRATION NO. |
| :--- | :--- |
| **HARSH SINGH** | 2401020462 |
| **ABHI RAJ** | 2401020434 |
| **AASTHA SINHA** | 2401020439 |
| **SNEHA MAITY** | 2401020422 |
| **SRINJONI MAPDAR** | 2401020421 |
| **MEDHA ROY GUPTA** | 2401020517 |

---

## 🛠️ Project Workflow & Methodology

### 1. Data Preprocessing
* Cleaned and processed a dataset containing student demographics and test scores.
* Calculated the aggregate average of Math, Reading, and Writing scores.
* Created a binary target variable: An average score $\ge$ 40 is classified as a **Pass (1)**, and $< 40$ is a **Fail (0)**.
* Applied One-Hot Encoding to categorical variables (Gender, Race/Ethnicity, Parental Education, Lunch Type, Test Prep).

### 2. Model Development (XGBoost)
* Implemented an **XGBoost Classifier** to capture complex, non-linear relationships in the demographic data.
* Trained the model to identify key socioeconomic drivers of academic success.
* Exported the trained model as a `.json` file for lightweight deployment.

### 3. Deployment & User Interface
The final model was deployed using **Streamlit**, featuring:
* **Real-time Prediction:** Adjust student scores and demographics to instantly see the predicted outcome.
* **Academic Safety Gate:** Hardcoded logic ensuring students below the mathematical threshold cannot artificially "pass" based on demographics alone.
* **Feature Importance Visualization:** A dynamic chart explaining exactly which demographic factors (e.g., Lunch Type, Parental Education) most heavily influenced the model's decision.
* **Certainty Scoring:** Displays the statistical confidence percentage of every model prediction.

---

## 💻 How to Run Locally

If you want to run this application on your own machine:

1. **Clone the repository:**
   ```bash

   git clone https://github.com/Official-89/Student-Performance-Classification.git
