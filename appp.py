# app.py
# Streamlit app - loads the trained model and lets you test headlines/articles
# Run train_model.py first to generate the .pkl files this needs

import os
import re
import string
import streamlit as st



def clean_text(text):
    # same cleaning as train_model.py, has to match or predictions will be off
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_model():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer


st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detector")
st.write("Paste a news headline or article below and it'll tell you if it looks Real or Fake.")

if not os.path.exists("fake_news_model.pkl"):
    st.error("Model not found - run train_model.py first (check README for dataset setup).")
    st.stop()

model, vectorizer = load_model()

user_text = st.text_area("Paste text here:", height=180)

if st.button("Check"):
    if not user_text.strip():
        st.warning("Paste something first")
    else:
        cleaned = clean_text(user_text)
        vec = vectorizer.transform([cleaned])

        prediction = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]
        confidence = round(max(probs) * 100, 2)

        if prediction == 1:
            st.success(f"This looks like REAL news ({confidence}% confidence)")
        else:
            st.error(f"This looks like FAKE news ({confidence}% confidence)")

        st.progress(int(confidence))

        st.caption(
            "Note: trained on a general news dataset, mostly US-based. "
            "Treat this as a demo project, not an actual fact-checker."
        )
