import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()

# ----------------- Background + Styling -----------------
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-attachment: fixed;
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("background.jpg")

#-----------------Text preprocesing -----------------------
def transform_text(sample):
    sample = sample.lower()
    sample = nltk.word_tokenize(sample)

    y=[]
    for i in sample:
        if i.isalnum():
            y.append(i)

    sample = y[:]
    y.clear()

    for i in sample:
        if i not in stopwords.words('english') and i not in  string.punctuation:
             y.append(i)

    sample = y[:]
    y.clear()

    for i in sample:
        y.append(ps.stem(i))

    return " ".join(y)
# ----------------------Extracting model & input text-------------------------
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.title('SMS/Email Spam Classifier')

input_text = st.text_area('Enter your message')
#-----------------------pipeline for classification---------------------------
if st.button('Submit'):
     processed_text = transform_text(input_text)
     Vector_input = tfidf.transform([processed_text]).toarray()
     result = model.predict(Vector_input)[0]
     if result == 1:
          st.header('Spam')
     else:
          st.header('Not Spam')

