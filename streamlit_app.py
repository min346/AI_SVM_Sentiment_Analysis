import streamlit as st
import re, string
import nltk
import contractions
from joblib import load
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---- One-time NLTK setup (downloads if missing; cached between runs) ----
@st.cache_resource
def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4")
    return True

_ = _ensure_nltk()

# ---- Load artifacts (model + vectorizer) ----
@st.cache_resource
def load_artifacts():
    mdl = load("SVM.joblib")
    vect = load("tfidf_vectorizer.joblib")
    return mdl, vect

model, vectorizer = load_artifacts()

# ---- NLP tools ----
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
APPLY_SPELL_CORRECTION = False  # toggle if you later add TextBlob

# ---- Clean function (copy of your notebook version; “Method 2 flexible”) ----
def clean_text(text: str) -> str:
    text = contractions.fix(str(text))                                # expand "don't" -> "do not"
    text = text.encode("ascii", errors="ignore").decode()              # drop non-ascii/garbled
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)   # remove punctuation
    text = text.lower()                                                # lowercase
    text = re.sub(r"\d+", "", text)                                    # remove numbers
    words = text.split()                                               # tokenize by space
    # optional spell correction (off by default)
    if APPLY_SPELL_CORRECTION:
        from textblob import TextBlob
        words = [str(TextBlob(w).correct()) for w in words]
    cleaned = [lemmatizer.lemmatize(w) for w in words
               if w not in stop_words and len(w) > 1]
    return " ".join(cleaned)

# ---- UI ----
st.set_page_config(page_title="Review Sentiment Classifier", page_icon="🤖", layout="centered")
st.title("🍟 McDonald's Review Sentiment Classifier")
st.write("Enter a review, and the model will predict **Positive** or **Negative**.")

# Text input
user_text = st.text_area("✍️ Type a review here:", height=160, placeholder="e.g., The fries were crispy and the staff were super friendly!")

col1, col2 = st.columns(2)
with col1:
    predict_btn = st.button("🔮 Predict")

if predict_btn:
    if not user_text.strip():
        st.warning("Please enter a review first.")
    else:
        cleaned = clean_text(user_text)
        features = vectorizer.transform([cleaned])
        pred = model.predict(features)[0]  # e.g., "positive" or "negative"

        if pred.lower() == "positive":
            st.success("✅ Predicted Sentiment: **Positive** 😃")
        else:
            st.error("❌ Predicted Sentiment: **Negative** 😞")

        # Show the cleaned text (useful for demo)
        with st.expander("See cleaned text used for prediction"):
            st.code(cleaned)

# ---- Optional: Batch predictions from a CSV upload ----
st.divider()
st.subheader("📁 Predict from CSV (optional)")
st.caption("Upload a CSV with a column named **Review**.")
file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    import pandas as pd
    try:
        df_up = pd.read_csv(file)
    except UnicodeDecodeError:
        # try common alternative encodings
        df_up = pd.read_csv(file, encoding="cp1252")

    if "Review" not in df_up.columns:
        st.error("CSV must have a column named 'Review'.")
    else:
        df_up["cleaned_review"] = df_up["Review"].astype(str).apply(clean_text)
        X_up = vectorizer.transform(df_up["cleaned_review"])
        df_up["predicted_sentiment"] = model.predict(X_up)
        st.success("Predictions completed.")
        st.dataframe(df_up[["Review", "cleaned_review", "predicted_sentiment"]].head(20), use_container_width=True)

        # Download results
        from io import BytesIO
        buf = BytesIO()
        df_up.to_csv(buf, index=False)
        st.download_button("⬇️ Download results CSV",
                           data=buf.getvalue(),
                           file_name="predicted_reviews.csv",
                           mime="text/csv")
