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

# Navigation bar
#st.sidebar.title("Navigation")
nav_selection = st.sidebar.radio("", ["Home", "Sign up","Login","Email Prediction","About Us"])

if nav_selection == "Home":
    # Set background image for homepage only with moving letters
    st.markdown(
        """
        <style>
        @keyframes moveContent {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        .moving-content {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 25px;
            font-weight: bold;
            color: #ffffff; /* White color */
            font-family: 'Arial', sans-serif; /* Choose a suitable font */
            animation: moveContent 10s linear infinite;
        }

        .main-content {
            padding: 50px;
            text-align: center;
            color: #ffffff; /* White color */
            font-family: 'Arial', sans-serif; /* Choose a suitable font */
            font-size: 20px;
            line-height: 1.5;
            
        }

        .floating-image img {
            animation: moveContent 10s linear infinite;
        }

        .spacer {
            height: 50px; /* Adjust as needed */
        }

        body {
            background-image: url("https://media.istockphoto.com/id/1257404830/vector/email-marketing-icon-vector-graphics.jpg?s=612x612&w=0&k=20&c=0O_ldByYxp4qyQQszBIt08ErjP11Jf9ujCh-UZQS40Q=");
            background-size: cover;
            position: relative;
            overflow: auto;
            height: 100vh;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display moving letters and main content
    st.markdown('<div class="moving-content"><b>EMAIL SPAM CLASSIFIER</b></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-content">Welcome to our spam-free oasis! Our email spam classification project harnesses the power of machine learning to keep your inbox clutter-free. Say goodbye to endless junk mail and hello to efficient communication. Explore how we\'re revolutionizing email management one classification at a time.</div>', unsafe_allow_html=True)
    
    # Display floating image using HTML markup with .floating-image class
    st.markdown('<div class="floating-image"><img src="https://t3.ftcdn.net/jpg/01/75/84/66/360_F_175846607_4s5DLFCO1YMvx6gxfvIAH5F1hMs1BMS6.jpg" width="550" /></div>', unsafe_allow_html=True)

elif nav_selection == "Sign up":
    st.title("Sign Up")
    st.write("""
    Create a new account:
    """)

    new_username = st.text_input("New Username")
    email = st.text_input("Email")
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if new_password != confirm_password:
        st.warning("Passwords do not match")
    else:
        if st.button("Sign Up"):
            # Add sign-up logic here
            st.success("Account created successfully")

            # Add CSS animation for success message
            st.markdown(
                """
                <style>
                @keyframes fadeInOut {
                    0% { opacity: 0; }
                    25% { opacity: 1; }
                    75% { opacity: 1; }
                    100% { opacity: 0; }
                }

                .animated {
                    animation: fadeInOut 3s;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

elif nav_selection == "Login":
    st.title("Login")
    st.write("""
    
    """)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "your_username" and password == "your_password":
            st.success("Logged in as {}".format(username))
            # Redirect to Email Prediction page upon successful login
            st.experimental_reroute("/EmailPrediction.py")
        else:
            st.error("Invalid username or password")

    st.write("Don't have an account? [Go to Email Prediction](#EmailPrediction.py)")

elif nav_selection == "Email Prediction":
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


elif nav_selection == "About Us":
    st.title("About Us")
    st.write("""
    Welcome to our Spam Classifier App! We are dedicated to helping users identify spam emails and messages efficiently. Our app utilizes state-of-the-art machine learning algorithms to predict with high accuracy whether a message is spam or not, ensuring that you can sift through your inbox with ease and confidence. With our user-friendly interface and robust backend technology, we strive to provide a seamless experience that prioritizes your security and peace of mind. Feel empowered to take control of your digital communication and stay protected from the nuisance of spam. Join our growing community of users who rely on our app to safeguard their inboxes and streamline their online experience. Together, let's combat spam and foster a cleaner, safer digital environment. Welcome aboard, and enjoy a spam-free journey with us!
    """)

