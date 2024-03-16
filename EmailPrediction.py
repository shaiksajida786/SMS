import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download NLTK data if not already downloaded
# nltk.download('punkt')
# nltk.download('stopwords')

# Initialize Porter Stemmer
ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the TF-IDF vectorizer and the model
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Set background color
st.markdown(
    """
    <style>
    body {
        background-color: #ADD8E6; /* light blue */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("Email Prediction")

# Text area for input
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Preprocess
    transformed_sms = transform_text(input_sms)
    # Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # Predict
    result = model.predict(vector_input)[0]
    # Display result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

# Navigation to other pages
st.write("Want to go back to [Home](#Home)?")
st.write("Or check out our [About Us](#About-Us) page.")
st.write("Already have an account? [Login](#Login) here.")
st.write("New user? [Sign up](#Sign-up) now.")
