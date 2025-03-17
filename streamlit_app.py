import streamlit as st
import pandas as pd
import gdown
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# ✅ Ensure required NLTK resources are downloaded
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# ✅ Google Drive File IDs
DATA_FILE_ID = "1AsdUWNsA981I0GXty9r345IBC4Ly_D1X"
VALIDATION_FILE_ID = "1Wj8Z4FrhVvYZtfhLvaaiyoTYdtWFFRn9"

# ✅ Function to download CSV files from Google Drive
@st.cache_data
def download_csv_from_gdrive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, fuzzy=True, quiet=False)
    return pd.read_csv(output_path)

# ✅ Download datasets
df = download_csv_from_gdrive(DATA_FILE_ID, "data.csv")
val_df = download_csv_from_gdrive(VALIDATION_FILE_ID, "validation_data.csv")

# ✅ Ensure column names exist
if "title" not in df.columns or "text" not in df.columns:
    st.error("❌ Error: The dataset does not have 'title' or 'text' columns!")
    st.stop()

# ✅ Text Preprocessing Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# ✅ Initialize NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ✅ Apply preprocessing (Fix missing "cleaned_text" column)
df["cleaned_text"] = df["title"] + " " + df["text"]
df["cleaned_text"] = df["cleaned_text"].apply(clean_text)

# ✅ Display dataset overview
st.title("📰 Fake News Detection App")
st.write("### Dataset Overview:")
st.dataframe(df[['label', 'title', 'text']].head())

# ✅ Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000)
X_tfidf = vectorizer.fit_transform(df["cleaned_text"])
y = df["label"]

# ✅ Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_tfidf, y)

# ✅ User Input Section
user_input = st.text_area("Enter a news headline or article:")

if st.button("Check News"):
    cleaned_input = clean_text(user_input)
    input_vector = vectorizer.transform([cleaned_input])
    prediction = model.predict(input_vector)[0]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success("📰 This is **Real News**!")
    else:
        st.error("🚨 This is **Fake News**!")
