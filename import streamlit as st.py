import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

st.title('Fake News Detection App')
st.sidebar.header('Upload your datasets')

uploaded_file = st.sidebar.file_uploader("Upload Training Dataset", type=["csv"])
validation_file = st.sidebar.file_uploader("Upload Validation Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Training Data Sample")
    st.write(df.head())

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)

    df['cleaned_text'] = (df['title'] + " " + df['text']).apply(clean_text)

    X = df['cleaned_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=62)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=500)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        st.write(f"### {name} Accuracy: {accuracy * 100:.2f}%")
        st.write(classification_report(y_test, y_pred))

        ConfusionMatrixDisplay.from_estimator(model, X_test_tfidf, y_test, cmap='Blues')
        st.pyplot(plt)

    st.write("## Model Comparison")
    plt.figure(figsize=(8, 4))
    plt.bar(results.keys(), results.values(), color=['blue', 'green', 'orange'])
    plt.title("Model Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0.5, 1.0)
    st.pyplot(plt)

    # Word Frequency Analysis
    real_words = " ".join(df[df['label'] == 1]['cleaned_text']).split()
    fake_words = " ".join(df[df['label'] == 0]['cleaned_text']).split()
    real_counts = Counter(real_words).most_common(15)
    fake_counts = Counter(fake_words).most_common(15)

    st.write("## Most Common Words in Real vs. Fake News")
    fig, ax = plt.subplots()
    ax.bar(*zip(*real_counts), color='blue', label='Real News')
    ax.bar(*zip(*fake_counts), color='red', alpha=0.7, label='Fake News')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(fig)

    # Word Clouds
    real_wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(real_words))
    fake_wc = WordCloud(width=800, height=400, background_color="black").generate(" ".join(fake_words))

    st.write("## Word Clouds")
    st.image(real_wc.to_array(), caption="Real News Word Cloud")
    st.image(fake_wc.to_array(), caption="Fake News Word Cloud")

    # Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    df["sentiment"] = df["cleaned_text"].apply(lambda text: sia.polarity_scores(text)["compound"])

    st.write("## Sentiment Distribution")
    fig, ax = plt.subplots()
    ax.hist(df[df["label"] == 1]["sentiment"], bins=20, alpha=0.7, label="Real News", color="blue")
    ax.hist(df[df["label"] == 0]["sentiment"], bins=20, alpha=0.7, label="Fake News", color="red")
    plt.legend()
    st.pyplot(fig)

    if validation_file is not None:
        val_df = pd.read_csv(validation_file)
        val_df['cleaned_text'] = (val_df['title'] + " " + val_df['text']).apply(clean_text)
        X_val_tfidf = vectorizer.transform(val_df['cleaned_text'])
        val_df['label'] = models["Random Forest"].predict(X_val_tfidf)
        st.write("### Validation Data Predictions")
        st.write(val_df)
