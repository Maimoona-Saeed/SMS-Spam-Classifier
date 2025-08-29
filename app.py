import streamlit as st
import pickle
import string
import nltk
import os
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Configure NLTK
nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Download required NLTK data
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Initialize stemmer
ps = PorterStemmer()

def transform_text(text):
    try:
        text = text.lower()
        text = word_tokenize(text)
        text = [word for word in text if word.isalnum()]
        text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
        text = [ps.stem(word) for word in text]
        return " ".join(text)
    except Exception as e:
        st.error(f"Text processing error: {str(e)}")
        return ""

# Load models with error handling
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"Model loading error: {str(e)}")
    st.stop()

# Streamlit UI
st.title("📩 SMS Spam Classifier")
input_sms = st.text_input("Enter the message")

if input_sms:
    transformed_sms = transform_text(input_sms)
    if transformed_sms:
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        st.header("🚫 Spam" if result == 1 else "✅ Not Spam")