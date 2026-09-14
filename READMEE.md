# Fake News Detector

A small ML project that checks whether a news headline/article is real
or fake. Made this as part of my placement prep to get hands-on with
text classification (my earlier project - resume screener - was more
about similarity matching, so wanted to try an actual classifier this
time).

## What it does
You paste in a headline or article text, and it tells you if the model
thinks it's Real or Fake, along with a confidence %.

## Dataset
Using the Fake and Real News Dataset from Kaggle (link below), has
around 45k articles already labeled. Didn't build my own dataset for
this one, would've taken too long to collect and label manually.

Link: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

You need a free Kaggle account to download it. After downloading you
get two files - Fake.csv and True.csv - put them in the same folder
as app.py.

## How it works
1. Load Fake.csv and True.csv, label them 0 and 1
2. Clean the text - lowercase everything, strip out urls/punctuation/numbers
3. Convert text to TF-IDF vectors
4. Train two models and compare - Logistic Regression and Naive Bayes
   (picked these two since we covered them, wanted to see which does
   better on this kind of data)
5. Whichever gets a better F1 score gets saved and used in the app

Ended up going with Logistic Regression in my run, got around 98-99%
accuracy which honestly surprised me at first, but that's apparently
normal for this dataset since it's pretty cleanly separable.

## Setup

Install requirements:
```
pip install -r requirements.txt
```

Train the model (run this first, only needs to be done once unless you
change something):
```
python train_model.py
```
This will take a few minutes since there's a lot of rows. It prints out
accuracy/precision/recall/F1 for both models so you can see the
comparison.

Then run the app:
```
streamlit run app.py
```

## Note for deployment
Streamlit Cloud doesn't run train_model.py on its own - it just serves
whatever files are in the repo. So after training locally, you need to
push the .pkl files it generates (fake_news_model.pkl,
tfidf_vectorizer.pkl, best_model_name.txt) to GitHub too, not just the
python scripts. Learned this the hard way after my first deploy attempt
didn't work.

## Things I'd want to improve if I get time
- test it on some actual Indian news headlines instead of just the
  Kaggle test set, since the dataset is mostly US news
- try adding a model that shows *why* it flagged something as fake
  (which words mattered most) - saw this is usually done with SHAP or LIME
- maybe extend this later to handle Hinglish/WhatsApp-forward style text,
  which is honestly a bigger problem in India than English fake news

## Author
Shalu Patel
B.Tech CSE (AIML), GGITS Jabalpur
