import streamlit as st
import pandas as pd
import re, string
import nltk
import contractions
from joblib import load
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import chardet
import io

# ---- One-time NLTK setup (downloads if missing; cached between runs) ----
@st.cache_resource
def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")
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

# ---- Clean function ----
def clean_text(text: str) -> str:
    text = contractions.fix(str(text))                                 # expand "don't" -> "do not"
    text = text.encode("ascii", errors="ignore").decode()              # drop non-ascii/garbled
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)   # remove punctuation
    text = text.lower()                                                # lowercase
    text = re.sub(r"\d+", "", text)                                    # remove numbers
    words = word_tokenize(text)                                        # tokenize
    # optional spell correction (off by default)
    if APPLY_SPELL_CORRECTION:
        from textblob import TextBlob
        words = [str(TextBlob(w).correct()) for w in words]
    cleaned = [lemmatizer.lemmatize(w) for w in words
               if w not in stop_words and len(w) > 1]
    return " ".join(cleaned)

# ---- CSV Reading function with encoding detection ----
def read_csv_with_encoding(file):
    """Read CSV file with automatic encoding detection"""
    try:
        # First, try to detect encoding
        raw_data = file.read()
        file.seek(0)  # Reset file pointer
        
        # Detect encoding
        detected = chardet.detect(raw_data)
        encoding = detected['encoding'] if detected['confidence'] > 0.7 else 'utf-8'
        
        # Try detected encoding first
        try:
            file.seek(0)
            df = pd.read_csv(io.StringIO(raw_data.decode(encoding)))
            return df, encoding
        except (UnicodeDecodeError, LookupError):
            pass
        
        # Try common encodings
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
        
        for encoding in encodings_to_try:
            try:
                file.seek(0)
                df = pd.read_csv(io.StringIO(raw_data.decode(encoding)))
                return df, encoding
            except (UnicodeDecodeError, LookupError):
                continue
        
        # If all encodings fail, try with error handling
        try:
            file.seek(0)
            df = pd.read_csv(io.StringIO(raw_data.decode('utf-8', errors='replace')))
            return df, 'utf-8 (with error replacement)'
        except Exception:
            raise ValueError("Could not decode the file with any common encoding")
            
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")

@st.cache_data
def predict_batch(texts):
    """Predict sentiment for a list of texts"""
    results = []
    for text in texts:
        if pd.isna(text) or str(text).strip() == '':
            results.append('unknown')
        else:
            cleaned = clean_text(str(text))
            if cleaned.strip() == '':
                results.append('unknown')
            else:
                features = vectorizer.transform([cleaned])
                pred = model.predict(features)[0]
                results.append(pred.lower())
    return results

def find_review_column(df):
    """Find review column by looking for variations of 'review'"""
    possible_names = ['review', 'Review', 'REVIEW']
    for col in df.columns:
        if col in possible_names:
            return col
    return None

# ---- UI Configuration ----
st.set_page_config(page_title="Sentiment Analysis", page_icon="📊", layout="wide")

# ---- Initialize session state ----
if "total_reviews" not in st.session_state:
    st.session_state.total_reviews = 0
    st.session_state.positive = 0
    st.session_state.negative = 0
if "csv_results" not in st.session_state:
    st.session_state.csv_results = None
# Initialize session state for preserving inputs and results
if "user_text_input" not in st.session_state:
    st.session_state.user_text_input = ""
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_cleaned_text" not in st.session_state:
    st.session_state.last_cleaned_text = ""
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "csv_analysis_done" not in st.session_state:
    st.session_state.csv_analysis_done = False

# ---- Sidebar Navigation ----
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox("Choose a page:", ["✍️ Review Prediction", "📁 CSV Analysis"])

# ---- Review Prediction Page ----
if page == "✍️ Review Prediction":
    st.title("🍟 McDonald's Review Sentiment Classifier")
    st.write("Here's a demo of how the model classifies reviews into **Positive** or **Negative**.")

    # Demo Sentiment Table
    st.subheader("🧾 Demo Sentiment Table")
    
    # Create demo data
    demo_reviews = [
        'The fries were cold and soggy, very disappointed.',
        'Fast delivery, the burger was still hot when it arrived!',
        'Customer service was rude and unhelpful.',
        'The app is easy to use, and ordering was smooth.',
        'The drink spilled inside the bag, really bad experience.',
        'Great value for the price, I will definitely order again.',
        'Long waiting time and the order was still wrong',
        'The new menu item is delicious, highly recommended!',
        'Packaging was terrible, food looked messy when it arrived.',
        'Friendly staff, clean restaurant, and tasty food.'
        
    ]
    
    actual_sentiments = [
        'Negative', 'Positive', 'Negative', 'Positive', 'Negative', 
        'Positive', 'Negative', 'Positive', 'Negative', 'Positive'
    ]
    
    # Generate dynamic predictions using the loaded model
    predicted_sentiments = []
    for review in demo_reviews:
        cleaned = clean_text(review)
        if cleaned.strip() == '':
            predicted_sentiments.append('Unknown')
        else:
            features = vectorizer.transform([cleaned])
            pred = model.predict(features)[0]
            predicted_sentiments.append(pred.capitalize())  # Capitalize to match format
    
    demo_data = {
        'Review': demo_reviews,
        'Actual Sentiment': actual_sentiments,
        'Predicted Sentiment': predicted_sentiments
    }
    
    demo_df = pd.DataFrame(demo_data)
    
    # Display the table with styling
    st.dataframe(
        demo_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Review": st.column_config.TextColumn("Review", width="large"),
            "Actual Sentiment": st.column_config.TextColumn("Actual Sentiment", width="medium"),
            "Predicted Sentiment": st.column_config.TextColumn("Predicted Sentiment (Dynamic)", width="medium")
        }
    )
    
    st.markdown("---")

    # Text input
    st.write("Now try typing your own review below to see the prediction in action!")
    user_text = st.text_area("✍️ Type a review here:", value=st.session_state.user_text_input, height=160, placeholder="e.g., The fries were crispy and the staff were super friendly!", key="review_text_area")
    
    # Update session state when text changes
    st.session_state.user_text_input = user_text

    col1, col2 = st.columns(2)
    with col1:
        predict_btn = st.button("🔮 Predict")
    
    # Show last prediction if it exists and no new prediction is being made
    if predict_btn:
        if not user_text.strip():
            st.warning("Please enter a review first.")
        else:
            cleaned = clean_text(user_text)
            features = vectorizer.transform([cleaned])
            pred = model.predict(features)[0]  # e.g., "positive" or "negative"

            # Store prediction in session state
            st.session_state.last_prediction = pred
            st.session_state.last_cleaned_text = cleaned

            if pred.lower() == "positive":
                st.success("✅ Predicted Sentiment: **Positive** 😃")
                st.session_state.positive += 1
            else:
                st.error("❌ Predicted Sentiment: **Negative** 😞")
                st.session_state.negative += 1
            
            st.session_state.total_reviews += 1

            # Show the cleaned text
            with st.expander("See cleaned text used for prediction"):
                st.code(cleaned)
    
    # Display last prediction if it exists (when returning to page)
    elif st.session_state.last_prediction is not None and st.session_state.user_text_input.strip():
        if st.session_state.last_prediction.lower() == "positive":
            st.success("✅ Predicted Sentiment: **Positive** 😃")
        else:
            st.error("❌ Predicted Sentiment: **Negative** 😞")
        
        # Show the cleaned text (useful for demo)
        with st.expander("See cleaned text used for prediction"):
            st.code(st.session_state.last_cleaned_text)
