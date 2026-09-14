# train_model.py
# Trains a fake news classifier - compares Logistic Regression vs Naive Bayes
# and saves whichever one does better.
#
# Before running this, download Fake.csv and True.csv from the Kaggle
# "Fake and Real News Dataset" and put them in this same folder.

import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib


def clean_text(text):
    # basic cleaning - lowercase, remove urls, punctuation, numbers
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


print("loading data...")
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = 0
true_df["label"] = 1

# combining title + text, title alone didn't seem like enough signal
fake_df["content"] = fake_df["title"].fillna("") + " " + fake_df["text"].fillna("")
true_df["content"] = true_df["title"].fillna("") + " " + true_df["text"].fillna("")

data = pd.concat([fake_df[["content", "label"]], true_df[["content", "label"]]])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"total rows: {len(data)}")
print(f"fake: {(data.label == 0).sum()}, real: {(data.label == 1).sum()}")

print("cleaning text (this takes a bit)...")
data["clean_content"] = data["content"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    data["clean_content"], data["label"], test_size=0.2, random_state=42
)

print("vectorizing...")
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, max_features=50000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
}

results = {}

for name, model in models.items():
    print(f"\ntraining {name}...")
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"accuracy: {acc:.4f}")
    print(f"precision: {prec:.4f}")
    print(f"recall: {rec:.4f}")
    print(f"f1: {f1:.4f}")
    print("confusion matrix:")
    print(confusion_matrix(y_test, preds))

    results[name] = {"model": model, "f1": f1}

# pick whichever model got the better f1 score
best_name = max(results, key=lambda k: results[k]["f1"])
best_model = results[best_name]["model"]
print(f"\nbest model: {best_name}")

joblib.dump(best_model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

with open("best_model_name.txt", "w") as f:
    f.write(best_name)

print("saved fake_news_model.pkl and tfidf_vectorizer.pkl")
print("now run: streamlit run app.py")
