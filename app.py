import streamlit as st
import pickle

# Load saved vectorizer and model
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# App UI
st.set_page_config(page_title="Spam Detector", page_icon="📩")
st.title("📩 Spam Message Detector")
st.write("Enter a message below to check whether it is **Spam** or **Not Spam**.")

# User input
message = st.text_area("Message", height=150)

# Prediction
if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        vector = tfidf.transform([message])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.error("🚫 Spam Message")
        else:
            st.success("✅ Not Spam")