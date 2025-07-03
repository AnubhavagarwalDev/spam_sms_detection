import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()


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


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model (1).pkl', 'rb'))

st.set_page_config(page_title="Spam Classifier", page_icon="✉️", layout="centered")
st.title("✉️ Email & SMS Spam Classifier")
st.write("Detect whether a message is **Spam** or **Not Spam** using Machine Learning.")

input_sms = st.text_area("Enter your message here:")
if st.button("Predict"):

    if input_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        with st.spinner("Analyzing message..."):
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]
        if result == 1:
            st.error("⚠️ This message is likely **Spam**.")
        else:
            st.success("✅ This message is **Not Spam**.")

st.markdown("---")
