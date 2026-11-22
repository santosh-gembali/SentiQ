
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
import bcrypt
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SentiQ | AI Review Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #080D1B 0%, #10172A 55%, #11152B 100%);
    color: #F1F5F9;
}

[data-testid="stSidebar"] {
    background: #0B1120;
    border-right: 1px solid #26324A;
}

/* Keep enough room at the bottom so sidebar content does not sit behind Logout */
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-bottom: 5.5rem;
}

/* Pin the Logout area to the bottom of the sidebar */
.st-key-logout_container {
    position: fixed;
    left: 1rem;
    bottom: 1rem;
    width: min(17rem, calc(100vw - 2rem));
    z-index: 1000;
    padding: 0.65rem;
    background: #0B1120;
    border-top: 1px solid #26324A;
}

[data-testid="stSidebar"] * {
    color: #DCE5F5;
}

.main-title {
    font-size: 37px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -1.5px;
}

.gradient-text {
    background: linear-gradient(90deg, #8B5CF6, #38BDF8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #94A3B8;
    font-size: 14px;
    line-height: 1.7;
}

.section-title {
    color: #F8FAFC;
    font-size: 22px;
    font-weight: 700;
    margin-top: 15px;
}

.metric-card {
    background: linear-gradient(145deg, #17233A, #111B30);
    border: 1px solid #293B5A;
    border-radius: 15px;
    padding: 20px;
    min-height: 120px;
}

.metric-label {
    color: #94A3B8;
    font-size: 13px;
}

.metric-value {
    color: #FFFFFF;
    font-size: 30px;
    font-weight: 800;
    margin-top: 8px;
}

.metric-icon {
    font-size: 20px;
}

.result-card {
    border-radius: 16px;
    padding: 24px;
    border: 1px solid #34415B;
    background: #17233A;
    margin-top: 15px;
}

.result-positive {
    border-left: 5px solid #22C55E;
}

.result-negative {
    border-left: 5px solid #EF4444;
}

.result-neutral {
    border-left: 5px solid #F59E0B;
}

.result-label {
    font-size: 25px;
    font-weight: 800;
    color: white;
}

.stButton > button {
    background: linear-gradient(90deg, #7C3AED, #2563EB);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    min-height: 45px;
    transition: 0.2s;
}

.stButton > button:hover {
    border: 1px solid #A78BFA;
    color: white;
    filter: brightness(1.15);
}

div[data-testid="stTextArea"] textarea {
    background: #121D32;
    color: #FFFFFF;
    border: 1px solid #34415B;
    border-radius: 12px;
}

div[data-testid="stTabs"] button {
    color: #A5B4CC;
    font-weight: 600;
}

hr {
    border-color: #29364E;
}

.footer {
    text-align: center;
    color: #64748B;
    font-size: 12px;
    padding: 25px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# AUTHENTICATION SYSTEM
# ==========================================




def login_page():

    st.markdown("""
    <div style="text-align:center; padding:60px 10px 20px;">
        <h1 style="font-size:42px;">
            🧠 Senti<span style="color:#8B5CF6;">Q</span>
        </h1>
        <p style="color:#94A3B8;">
            AI-Powered Customer Review Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)

    left, center, right = st.columns([1, 1.2, 1])

    with center:

        st.markdown("### Welcome Back")
        st.write("Sign in to access your dashboard.")

        with st.form("login_form"):

            selected_role = st.selectbox(
                "Select Role",
                ["User", "Admin"],
                index=None,
                placeholder="Choose your role"
            )

            username = st.text_input(
                "Username",
                placeholder="Enter username"
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter password"
            )

            submitted = st.form_submit_button(
                "Sign In",
                use_container_width=True
            )

        if submitted:

            username = username.strip()

            if not selected_role:
                st.warning("Please select your role.")

            elif not username or not password:
                st.warning("Please enter username and password.")

            # USER LOGIN: Allow multiple users without registration
            elif selected_role == "User":

                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.role = "user"

                st.success("Welcome! Opening User Dashboard...")
                st.rerun()

            # ADMIN LOGIN: Verify credentials using bcrypt
            elif selected_role == "Admin":

                accounts = st.secrets["accounts"]
                admin_account = accounts.get("admin")

                if admin_account is None:
                    st.error("Admin account configuration not found.")

                elif username != admin_account["username"]:
                    st.error("Invalid admin username.")

                else:

                    try:
                        password_valid = bcrypt.checkpw(
                            password.encode("utf-8"),
                            admin_account["password_hash"].encode("utf-8")
                        )

                    except ValueError:
                        st.error("Invalid admin password configuration.")
                        return

                    if password_valid:

                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.role = "admin"

                        st.success("Welcome Admin!")
                        st.rerun()

                    else:
                        st.error("Incorrect admin password.")

        st.caption(
            "Authorized access only. Contact your administrator "
            "if you need an admin account."
        )


# Initialize authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login_page()
    st.stop()

# ==========================================
# LOAD MACHINE LEARNING MODELS
# ==========================================

@st.cache_resource
def load_models():
    model = joblib.load(BASE_DIR / "sentiment_model.pkl")
    vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
    return model, vectorizer

try:
    model, tfidf = load_models()
except Exception as e:
    st.error(f"Unable to load model files: {e}")
    st.stop()

# ==========================================
# TEXT PREPROCESSING
# MUST MATCH TRAINING PREPROCESSING
# ==========================================

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))

stop_words -= {"not", "no", "nor"}

stemmer = PorterStemmer()

domain_stopwords = {
    "samsung", "mobile", "phone", "product", "amazon"
}

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    words = [
        word for word in words
        if word not in domain_stopwords
    ]

    return " ".join(words)

# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_sentiment(review):

    cleaned = clean_text(review)

    if not cleaned.strip():
        return None

    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]

    return prediction

# ==========================================
# SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("""
    <div style="padding:15px 5px 25px">
        <div style="font-size:30px;font-weight:800">
            🧠 Senti<span style="color:#8B5CF6">Q</span>
        </div>
        <div style="color:#64748B;font-size:11px">
            CUSTOMER INTELLIGENCE PLATFORM
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# ROLE-BASED NAVIGATION
# ==========================================

with st.sidebar:

    st.markdown("### ✦ SentiQ")

    st.write(
        f"Welcome, **{st.session_state.username}**"
    )

    role = st.session_state.role

    if role == "admin":

        st.caption("ADMINISTRATOR")

        pages = [
            "Overview",
            "Analyze Review",
            "Batch Analysis",
            "Dataset Explorer",
            "Model Performance"
        ]

    else:

        st.caption("USER ACCOUNT")

        pages = [
            "Analyze Review"
        ]

    page = st.radio(
        "MAIN MENU",
        pages
    )

    st.markdown("### MODEL INFORMATION")

    st.markdown("""
    **Algorithm:** Linear SVM

    **Feature Extraction:** TF-IDF

    **Framework:** Scikit-learn

    **NLP:** NLTK
    """)

    st.divider()

    st.caption("NLP Sentiment Analysis Project")
    st.caption("Built with Python & Streamlit")

    # Logout stays fixed at the bottom of the sidebar
    with st.container(key="logout_container"):
        if st.button("🚪 Logout", use_container_width=True, key="sidebar_logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.role = None
            st.rerun()

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div>
    <div class="main-title">
        Senti<span class="gradient-text">Q</span>
    </div>
    <div class="subtitle">
        AI-powered customer review intelligence.
        Understand customer opinions through Natural Language Processing.
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# ==========================================
# LOAD DATASET
# ==========================================

@st.cache_data
def load_dataset():
    path = BASE_DIR / "Dataset (2).xlsx"

    if not path.exists():
        return None

    data = pd.read_excel(path)

    def label_sentiment(rating):
        if rating <= 2:
            return "Negative"
        elif rating == 3:
            return "Neutral"
        return "Positive"

    data["sentiment"] = data["rating"].apply(label_sentiment)

    return data

df = load_dataset()

# ==========================================
# OVERVIEW PAGE
# ==========================================

if page == "Overview":

    st.markdown('<div class="section-title">Dashboard Overview</div>',
                unsafe_allow_html=True)

    st.write("Your customer review intelligence at a glance.")

    if df is not None:

        total = len(df)
        positive = int((df["sentiment"] == "Positive").sum())
        negative = int((df["sentiment"] == "Negative").sum())
        neutral = int((df["sentiment"] == "Neutral").sum())

        c1, c2, c3, c4 = st.columns(4)

        cards = [
            (c1, "TOTAL REVIEWS", total, "✦"),
            (c2, "POSITIVE", positive, "↗"),
            (c3, "NEGATIVE", negative, "↘"),
            (c4, "NEUTRAL", neutral, "◉")
        ]

        for col, label, value, icon in cards:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value:,}</div>
                </div>
                """, unsafe_allow_html=True)

        st.write("")

        left, right = st.columns(2)

        with left:
            st.markdown("### Sentiment Distribution")

            counts = df["sentiment"].value_counts().reindex(
                ["Positive", "Negative", "Neutral"],
                fill_value=0
            )

            st.bar_chart(counts)

        with right:
            st.markdown("### Sentiment Proportion")

            st.write("Share of customer opinions")

            st.dataframe(
                pd.DataFrame({
                    "Sentiment": counts.index,
                    "Reviews": counts.values,
                    "Percentage": (
                        counts.values / total * 100
                    ).round(2)
                }),
                hide_index=True,
                use_container_width=True
            )

        st.write("")

        st.markdown("### Recent Review Samples")

        st.dataframe(
            df[["title", "rating", "sentiment"]].head(8),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning("Dataset file not found.")

# ==========================================
# SINGLE REVIEW ANALYSIS
# ==========================================

elif page == "Analyze Review":

    st.markdown(
        '<div class="section-title">AI Review Analyzer</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter any customer review and let the trained SVM model "
        "identify its sentiment."
    )

    review = st.text_area(
        "CUSTOMER REVIEW",
        height=170,
        placeholder="Example: The camera quality is excellent "
                    "and battery performance is amazing..."
    )

    if st.button("✦ Analyze Sentiment", use_container_width=True):

        if not review.strip():

            st.warning("Please enter a customer review.")

        else:

            prediction = predict_sentiment(review)

            if prediction is None:

                st.warning("Please enter meaningful review text.")

            else:

                styles = {
                    "Positive": (
                        "😊", "POSITIVE SENTIMENT",
                        "result-positive",
                        "Customers express a favourable opinion."
                    ),
                    "Negative": (
                        "😞", "NEGATIVE SENTIMENT",
                        "result-negative",
                        "Customers express dissatisfaction."
                    ),
                    "Neutral": (
                        "😐", "NEUTRAL SENTIMENT",
                        "result-neutral",
                        "The review is classified as neutral."
                    )
                }

                emoji, title, css_class, description = styles[prediction]

                st.markdown(f"""
                <div class="result-card {css_class}">
                    <div style="font-size:38px">{emoji}</div>
                    <div class="result-label">{title}</div>
                    <p style="color:#A5B4CC">{description}</p>
                </div>
                """, unsafe_allow_html=True)

                st.write("")

                st.markdown("### Analysis Details")

                st.write("**Original review:**", review)
                st.write("**Predicted sentiment:**", prediction)

                st.session_state.history.insert(
                    0,
                    {
                        "Review": review,
                        "Sentiment": prediction
                    }
                )

    st.divider()

    st.markdown("### Example Reviews")

    examples = [
        "Excellent camera and amazing battery life.",
        "Very poor quality. Totally disappointed.",
        "The phone has 128 GB storage."
    ]

    for example in examples:
        st.code(example)

# ==========================================
# BATCH ANALYSIS
# ==========================================

elif page == "Batch Analysis":

    st.markdown(
        '<div class="section-title">Batch Review Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a CSV file containing customer reviews "
        "to analyze multiple reviews together."
    )

    uploaded = st.file_uploader(
        "Upload customer review CSV",
        type=["csv"]
    )

    if uploaded:

        batch_df = pd.read_csv(uploaded)

        review_column = (
            "review" if "review" in batch_df.columns
            else "body" if "body" in batch_df.columns
            else None
        )

        if review_column is None:

            st.error(
                "Your CSV must contain a 'review' or 'body' column."
            )

        else:

            if st.button("Analyze All Reviews"):

                valid_reviews = batch_df[review_column].fillna("").astype(str)

                batch_df["Predicted Sentiment"] = [
                    predict_sentiment(text) or "Unclassified"
                    for text in valid_reviews
                ]

                st.success(
                    f"Analyzed {len(batch_df)} uploaded records."
                )

                st.dataframe(
                    batch_df,
                    use_container_width=True
                )

                csv_data = batch_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Analysis Results",
                    data=csv_data,
                    file_name="sentiment_results.csv",
                    mime="text/csv"
                )

# ==========================================
# DATASET EXPLORER
# ==========================================

elif page == "Dataset Explorer":

    st.markdown(
        '<div class="section-title">Dataset Explorer</div>',
        unsafe_allow_html=True
    )

    if df is not None:

        st.write("Explore the original customer review dataset.")

        search = st.text_input("Search customer reviews")

        filtered = df.copy()

        if search:
            filtered = filtered[
                filtered["body"].astype(str).str.contains(
                    search,
                    case=False,
                    na=False
                )
            ]

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Dataset",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="customer_sentiment_dataset.csv",
            mime="text/csv"
        )

    else:
        st.warning("Dataset file not found.")

# ==========================================
# MODEL PERFORMANCE
# ==========================================

elif page == "Model Performance":

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Performance recorded during the original held-out test evaluation."
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">TEST ACCURACY</div>
            <div class="metric-value">78.13%</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">MACRO F1-SCORE</div>
            <div class="metric-value">65.64%</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    comparison = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Naive Bayes",
            "Original SVM",
            "Balanced SVM"
        ],
        "Accuracy": [
            79.51, 75.00, 78.13, 76.39
        ],
        "Macro F1": [
            60.86, 53.11, 65.64, 64.65
        ]
    })

    st.markdown("### Algorithm Comparison")

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Important Model Limitation")

    st.warning(
        "Neutral sentiment detection needs improvement. "
        "The original SVM achieved only 23% recall for the "
        "Neutral class on the held-out test set. "
        "The displayed metrics are historical test results, "
        "not live application accuracy."
    )

    st.markdown("### Model Architecture")

    st.code("""
Customer Review
      |
Text Preprocessing
      |
TF-IDF Vectorization
      |
Linear Support Vector Machine
      |
Positive / Negative / Neutral
    """, language="text")

# ==========================================
# REVIEW HISTORY
# ==========================================

st.divider()

with st.expander("🕘 Recent Prediction History"):

    if st.session_state.history:

        st.dataframe(
            pd.DataFrame(st.session_state.history),
            use_container_width=True,
            hide_index=True
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

    else:
        st.info("No review predictions yet.")

# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    🧠 SentiQ | AI Customer Review Intelligence<br>
    Built with Python · NLP · Scikit-learn · Streamlit
</div>
""", unsafe_allow_html=True)
