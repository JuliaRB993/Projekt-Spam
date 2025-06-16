import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from load_data_safe import load_data_safe

# Nie wymaga punkt, działa offline
tokenizer = TreebankWordTokenizer()

st.set_page_config(page_title="Klasyfikator Spamu", layout="centered")
st.title("📧 Klasyfikator Emaili: Spam czy Nie-Spam")
st.write("Wprowadź treść wiadomości email, a aplikacja powie, czy to spam.")

data = load_data_safe()

@st.cache_resource
def build_model():
    try:
        stop_words = stopwords.words('english')
    except:
        stop_words = ['the', 'and', 'is', 'in', 'to', 'of']

    def preprocess(text):
        text = text.lower()
        text = ''.join([char for char in text if char not in string.punctuation])
        tokens = tokenizer.tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        return " ".join(tokens)

    data['text'] = data['text'].astype(str).apply(preprocess)
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])

    pipeline.fit(X_train, y_train)
    return pipeline

model = build_model()

email_text = st.text_area("✉️ Treść wiadomości email:", height=200)

if st.button("🔍 Sprawdź, czy to spam"):
    if email_text.strip() == "":
        st.warning("Wprowadź tekst wiadomości.")
    else:
        prediction = model.predict([email_text])[0]
        prob = model.predict_proba([email_text])[0].max()
        if prediction.lower() == 'spam':
            st.error(f"⚠️ To jest SPAM (pewność: {prob:.2%})")
        else:
            st.success(f"✅ To nie jest spam (pewność: {prob:.2%})")
