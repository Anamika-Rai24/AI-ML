import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# ----------------------------
# Enhanced Background & Fonts
# ----------------------------
st.markdown("""
<style>

/* Main Background - Clear Student Theme */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1503676260728-1c00da094a0b");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* White Transparent Main Container */
.main {
    background-color: rgba(255, 255, 255, 0.95);
    padding: 40px;
    border-radius: 20px;
}

/* Title Styling */
h1 {
    font-size: 50px !important;
    color: #1f3c88;
    text-align: center;
    font-weight: bold;
}

/* Subtitle */
h3 {
    font-size: 28px !important;
    text-align: center;
    color: #2c3e50;
}

/* Increase Slider Label Size */
section[data-testid="stSidebar"] label {
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #1f3c88 !important;
}

/* Increase Slider Value Number Size */
section[data-testid="stSidebar"] .stSlider div[data-baseweb="slider"] + div {
    font-size: 18px !important;
    font-weight: 600 !important;
}

/* Result Box */
.result-box {
    padding: 25px;
    border-radius: 15px;
    font-size: 26px;
    text-align: center;
    font-weight: bold;
    margin-top: 20px;
}

/* Feature Importance Title */
h2 {
    font-size: 32px !important;
    color: #1f3c88;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Title Section
# ----------------------------
st.title("🎓 Student of the Year Predictor")
st.markdown("<h3>Predict Academic Excellence Using Machine Learning 📊</h3>", unsafe_allow_html=True)

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("SOTY1.csv")

X = df[["Attendance", "Extra_Curricular", "Study_Hours", "Marks"]]
y = df["SOTY"]

# Train Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# ----------------------------
# Sidebar Inputs
# ----------------------------rriculam
st.sidebar.header("📥 Enter Student Details")

attendance = st.sidebar.slider("Attendance (%)", 0, 100, 75)
extra = st.sidebar.slider("Extra Curricular Score", 0, 100, 60)
study = st.sidebar.slider("Study Hours per Day", 0, 10, 3)
marks = st.sidebar.slider("Marks", 0, 200, 120)

# ----------------------------
# Prediction
# ----------------------------
if st.sidebar.button("Predict 🎯"):

    new_data = pd.DataFrame(
        [[attendance, extra, study, marks]],
        columns=["Attendance", "Extra_Curricular", "Study_Hours", "Marks"]
    )

    prediction = model.predict(new_data)

    if prediction[0] == 1:
        st.markdown(
            '<div class="result-box" style="background-color:#d4edda; color:#155724;">🏆 Student of the Year: YES 🎉</div>',
            unsafe_allow_html=True
        )
        st.balloons()
    else:
        st.markdown(
            '<div class="result-box" style="background-color:#f8d7da; color:#721c24;">❌ Student of the Year: NO</div>',
            unsafe_allow_html=True
        )

# ----------------------------
# Feature Importance
# ----------------------------
st.markdown("---")
st.markdown("<h2>📊 Feature Importance</h2>", unsafe_allow_html=True)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance.set_index("Feature"))

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("<h4 style='text-align:center;'>Made with ❤️ using Streamlit</h4>", unsafe_allow_html=True)