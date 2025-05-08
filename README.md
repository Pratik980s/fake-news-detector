# 📰 Fake News Detector

This is a Streamlit-based machine learning web app that classifies news articles as **Real** or **Fake** using a Logistic Regression model and TF-IDF vectorization.

## 🚀 Features

- Uploads and processes a dataset of news articles
- Trains a Logistic Regression classifier
- Interactive UI using Streamlit
- Evaluation metrics and confusion matrix visualization
- Custom text input for real-time prediction

## 🗃️ Dataset

The app uses a CSV file (`news.csv`) with the following columns:
- `title`
- `text`
- `label` (REAL or FAKE)

> Ensure your dataset file is named `news.csv` and placed in the project root.

## 🛠️ How to Run

### Prerequisites

Install the required Python packages:

```bash
pip install -r requirements.txt


**Run the App**
streamlit run app.py


📊 Model Info
Vectorizer: TfidfVectorizer

Classifier: LogisticRegression

Evaluation: classification_report, confusion_matrix


📁 Project Structure
fake-news-detector/
│
├── app.py             # Streamlit app
├── news.csv           # Dataset
└── README.md          # Project overview


Author
Pratik Sakhare
