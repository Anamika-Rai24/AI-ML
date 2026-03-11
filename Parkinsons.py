# Parkinson's Disease Prediction Web App

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("parkinsons_reduced.csv")

X = data.drop(columns="status")
y = data["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = RandomForestClassifier(n_estimators=200)

model.fit(X_train,y_train)

accuracy = model.score(X_test,y_test)

print("Model Accuracy:",accuracy)

pickle.dump(model,open("parkinsons_model.pkl","wb"))

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# -----------------------
# Page Config
# -----------------------

st.set_page_config(
    page_title="Parkinson Disease Prediction",
    page_icon="🧠",
    layout="wide"
)

# -----------------------
# Animated Background
# -----------------------

page_bg = """
<style>

/* Background Image */

.stApp{
background-image:url("https://images.unsplash.com/photo-1530497610245-94d3c16cda28");
background-size:cover;
background-position:center;
background-attachment:fixed;
}

/* MAIN PAGE TITLES */

[data-testid="stAppViewContainer"] h1{
color:white !important;
text-align:center;
font-size:45px;
font-weight:bold;
text-shadow:2px 2px 10px black;
}

[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3{
color:white !important;
text-shadow:1px 1px 8px black;
}

/* MAIN PAGE TEXT */

[data-testid="stAppViewContainer"] p{
color:white !important;
}

/* Slider Labels */

div[data-testid="stSlider"] label{
color:#f5f5f5 !important;
font-weight:bold;
text-shadow:1px 1px 4px black;
}

/* SIDEBAR (Project Information) */

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label{
color:black !important;
text-shadow:none !important;
}

/* Button */

.stButton>button{
background-color:#ff4b4b;
color:white;
font-size:18px;
border-radius:10px;
padding:10px;
}

</style>
"""

st.markdown(page_bg,unsafe_allow_html=True)

# -----------------------
# Load Model
# -----------------------

model = pickle.load(open("parkinsons_model.pkl","rb"))

# -----------------------
# Title
# -----------------------

st.title("🧠 Parkinson Disease Prediction System")

st.write(
"""This AI system predicts whether a patient may have **Parkinson's Disease** based on voice measurement features."""
)

# -----------------------
# Sidebar
# -----------------------

st.sidebar.header("Project Information")

st.sidebar.write("""
Dataset: Parkinson's UCI Dataset  
Algorithm: Random Forest  
Features: 10 Voice Measurements
""")

st.sidebar.success("Model Accuracy ≈ 94%")

# -----------------------
# Slider Inputs
# -----------------------

st.header("Patient Voice Measurements")

col1,col2 = st.columns(2)

with col1:

    fo = st.slider("Fo (Hz)",80.0,250.0,120.0)
    fhi = st.slider("Fhi (Hz)",100.0,300.0,150.0)
    flo = st.slider("Flo (Hz)",60.0,200.0,80.0)
    jitter = st.slider("Jitter (%)",0.0,0.05,0.01)
    shimmer = st.slider("Shimmer",0.0,0.1,0.03)

with col2:

    nhr = st.slider("NHR",0.0,0.5,0.02)
    hnr = st.slider("HNR",10.0,35.0,20.0)
    rpde = st.slider("RPDE",0.0,1.0,0.5)
    dfa = st.slider("DFA",0.0,1.0,0.7)
    ppe = st.slider("PPE",0.0,1.0,0.2)

# -----------------------
# Prediction
# -----------------------

if st.button("Predict Parkinson Disease"):

    input_data = pd.DataFrame([[fo,fhi,flo,jitter,shimmer,
                                nhr,hnr,rpde,dfa,ppe]],

    columns=[
        "MDVP:Fo(Hz)",
        "MDVP:Fhi(Hz)",
        "MDVP:Flo(Hz)",
        "MDVP:Jitter(%)",
        "MDVP:Shimmer",
        "NHR",
        "HNR",
        "RPDE",
        "DFA",
        "PPE"
    ])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Parkinson Disease Detected")
    else:
        st.success("✅ Person is Healthy")

# -----------------------
# Feature Importance
# -----------------------

st.subheader("Feature Importance")

features = ["Fo","Fhi","Flo","Jitter","Shimmer","NHR","HNR","RPDE","DFA","PPE"]

importance = model.feature_importances_

fig,ax = plt.subplots()

ax.barh(features,importance)

ax.set_xlabel("Importance")

st.pyplot(fig)

# -----------------------
# CSV Upload Prediction
# -----------------------

st.subheader("Batch Prediction using CSV")

file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:

    data = pd.read_csv(file)

    # Remove target column if present
    if "status" in data.columns:
        data = data.drop(columns=["status"])

    features = [
        "MDVP:Fo(Hz)",
        "MDVP:Fhi(Hz)",
        "MDVP:Flo(Hz)",
        "MDVP:Jitter(%)",
        "MDVP:Shimmer",
        "NHR",
        "HNR",
        "RPDE",
        "DFA",
        "PPE"
    ]

    data = data[features]

    predictions = model.predict(data)

    data["Prediction"] = predictions

    st.write(data)

    st.download_button(
        "Download Result CSV",
        data.to_csv(index=False),
        "prediction_results.csv"
    )


# -----------------------
# Footer
# -----------------------

st.markdown("---")

st.write("AI / ML Certification Project")