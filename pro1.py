import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
@st.cache_data

def load_data():
    df = pd.read_csv("news.csv")
    df.drop(columns=["Unnamed: 0"], inplace=True)
    df["content"] = df["title"] + " " + df["text"]
    return df

# Page Config
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake vs Real News Classifier")
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    .stTextArea>textarea {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load and show data
df = load_data()
with st.expander("📊 Show Dataset"):
    st.dataframe(df.head())

# Preprocess
X = df["content"]
y = df["label"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Choose prediction source
st.sidebar.title("🛠️ Options")
option = st.sidebar.radio("Choose dataset for prediction:", ("Training", "Testing"))

if option == "Training":
    data, labels = X_train, y_train
else:
    data, labels = X_test, y_test

# Predict
predictions = model.predict(data)

# Evaluation
st.subheader("📈 Model Evaluation")
st.code(classification_report(labels, predictions))

# Bar graph for confusion matrix
cm = confusion_matrix(labels, predictions)
labels_names = ['FAKE', 'REAL']
cm_df = pd.DataFrame(cm, index=labels_names, columns=labels_names)
fig, ax = plt.subplots(figsize=(5, 4))
cm_df.plot(kind='bar', ax=ax, color=['skyblue', 'lightgreen'])
plt.title("Confusion Matrix - Bar Chart")
plt.ylabel("True Count")
plt.xlabel("Class")
st.pyplot(fig)

# Custom prediction
st.subheader("✍️ Enter A News")
user_input = st.text_area("Enter news text (title + article):")
if st.button("🔍 Predict"):
    if user_input.strip():
        input_vec = vectorizer.transform([user_input])
        result = model.predict(input_vec)[0]
        st.success(f"✅ The model predicts this news is: **{result}**")
    else:
        st.warning("⚠️ Please enter some text to predict.")
