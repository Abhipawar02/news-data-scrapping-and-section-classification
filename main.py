import streamlit as st
import joblib
from sklearn.feature_extraction.text import CountVectorizer

# Load the logistic regression model

lr_model = joblib.load('model/TheIndianExpress-C4-random_forest_model.pkl', mmap_mode=None)


# Load the CountVectorizer used during training
try:
    cv = CountVectorizer(decode_error="replace", vocabulary=joblib.load('model/count_vectorizer.pkl'))
except Exception as e:
    st.error(f"Error loading CountVectorizer: {e}")
    st.stop()

# Streamlit UI
st.title("News Article Classification App")

# User input text
user_input = st.text_area("Enter your Article text here:")

if st.button("Predict"):
    if user_input:
        # Transform the user input using the loaded CountVectorizer
        user_input_transformed = cv.transform([user_input])

        # Make predictions using the loaded logistic regression model
        prediction = lr_model.predict(user_input_transformed)

        # Mapping the prediction to labels
        labels = {0: "Entertainment", 1: "Sports", 2: "World", 3: "Business"}
        result = labels.get(prediction[0], "Unknown")

        # Display the result
        st.success(f"The predicted category is: {result}")
    else:
        st.warning("Please enter text for prediction.")
