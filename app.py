import pathlib
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Genre KNN Explorer Oh", page_icon="🎧", layout="centered")

st.markdown(
    """
    <style>
    body {
        background: #ffd6e7;
    }
    .stApp {
        background: #ffd6e7;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 780px;
    }
    .stButton>button {
        background-color: #ff77b3;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff4f91;
    }
    .stSlider>div>div>div>div {
        color: #333333;
    }
    .css-1d391kg {
        background-color: rgba(255,255,255,0.65);
        border-radius: 24px;
        padding: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Music Genre Predictor")
st.markdown(
    "Discover which genre your track best matches using a KNN model trained on tempo, energy, danceability, and acousticness."
)

model_path = pathlib.Path(__file__).resolve().parent / "genre_knn.pkl"

@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    model = load_model(model_path)
except Exception as exc:
    st.error("Could not load the genre model. Make sure genre_knn.pkl is present.")
    st.stop()

with st.form(key="genre_form"):
    st.subheader("Track attributes")
    tempo = st.slider("Tempo (BPM)", min_value=40, max_value=220, value=120, step=1)
    energy = st.slider("Energy", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    danceability = st.slider("Danceability", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
    acousticness = st.slider("Acousticness", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    submit = st.form_submit_button("Predict genre")

if submit:
    features = np.array([[tempo, energy, danceability, acousticness]])
    try:
        prediction = model.predict(features)
        proba = model.predict_proba(features)
        if hasattr(model, "classes_"):
            labels = model.classes_
        else:
            labels = [f"Class {i}" for i in range(proba.shape[1])]
        genre = prediction[0]

        st.markdown("### Prediction")
        st.success(f"**{genre}**")

        proba_df = pd.DataFrame(proba, columns=labels)
        proba_df = proba_df.T
        proba_df.columns = ["Probability"]
        proba_df["Probability"] = proba_df["Probability"].round(3)

        st.markdown("### Prediction probabilities")
        st.bar_chart(proba_df)
        st.write(proba_df)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
