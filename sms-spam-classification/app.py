import pickle
import streamlit as st

# Load model
with open("train_model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit UI
st.title("SMS Spam Classifier")
user_input = st.text_area("Enter a message:")

if st.button("Predict"):
    prediction = model.predict([user_input])[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    st.write(f"Prediction: {result}")
