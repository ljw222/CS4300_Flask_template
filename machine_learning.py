from sklearn.linear_model import LogisticRegression
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score, make_scorer, f1_score
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import nltk
import re

def train():
  stop_words = ["floor", "restaurant", "owner", "food", "counter", "windy", "radius", "ingredients", "hours", "person",
             "review", "people", "everybody", "eat", "ate", "plate", "plated", "order", "ordered", "today"]
  stop_words = nltk.corpus.stopwords.words("english") + stop_words
  ambiance_labels = ['touristy', 'classy', 'romantic', 'casual', 'hipster', 'divey', 'intimate', 'trendy', 'upscale']
  # use this for training/saving models
  with open("finalData2.json", "r") as f:
      data = json.load(f)
  # df = pd.DataFrame(columns=['restaurant', 'reviews', 'price', 'stars'] + ambiance_labels)
  df = pd.DataFrame(columns=['restaurant', 'reviews'] + ambiance_labels)
  for restaurant, restaurant_dict in data['BOSTON'].items():
    ambiances = restaurant_dict['ambience']
    review_texts = [r['text'] for r in restaurant_dict['reviews']]
    review_combined = '\n'.join(review_texts) # join the two reviews into one
    new_row = dict()
    new_row['restaurant'] = restaurant
    new_row['reviews'] = review_combined
    for label in ambiance_labels:
        if label in ambiances:
            new_row[label] = 1
        else:
            new_row[label] = 0
    if len(ambiances) > 0:
        df = df.append(new_row, ignore_index=True)
  print("df finished")
  manual = pd.read_csv("manual_data.csv") # add in this manual data to boost some features we want
  df = pd.concat([df, manual, manual, manual], axis=0, ignore_index=True)
  for col in ambiance_labels:
    df[col] = df[col].astype(int)

  df['reviews'] = df['reviews'].apply(lambda x: " ".join([t.lower() for t in x.split() if t.lower() not in stop_words and bool(re.match(r"^[a-zA-Z]+$", t))]))
  X = df['reviews']
  Y = df[ambiance_labels]
  vec = TfidfVectorizer()
  X_features = pd.DataFrame(vec.fit_transform(X).toarray())
  with open("ml_vectorizer.sav", "wb") as f:
    pickle.dump(vec, f)
  names = vec.get_feature_names()
  # commented out code below was for when we did cross validation

  # param_grid = {'C': [1, 10, 100],
  #             'gamma': [1, 0.1, 0.01],
  #             'kernel': ['rbf', 'linear']}
  # param_grid = {'C': [10, 1, 0.5, 0.25, 0.1]}

  models = dict()

  # grid = GridSearchCV(LogisticRegression(), param_grid, refit = True, verbose = 0, scoring="f1_weighted")
  for col in ambiance_labels:
    X = X_features
    y = Y[col]
    print(col)
    X_sm, y_sm = SMOTE(random_state=42).fit_resample(X, y)
    X_train, X_test, y_train, y_test =  train_test_split(X_sm, y_sm, test_size=0.3, random_state=42)
    print("Positive: ", len(y_train[y_train==1]))
    print("Negative: ", len(y_train[y_train==0]))
    # grid.fit(X_train, y_train)
    # print(grid.best_score_)
    # print(grid.best_estimator_)
    # model = grid.best_estimator_
    model = LogisticRegression(C=1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))
    models[col] = model
    importance = model.coef_[0]
    importances = sorted([(i, v) for i, v in enumerate(importance)], key=lambda x: -x[1])
    for i,v in importances[:10]:
        print(f'Feature: {names[i]}, Score: {v}')
    print("")
    pickle.dump(model, open(f"model_{col}.sav", "wb"))


def predict(review, ambiance_label):
  # predicts if an ambiance label should be applied for some review
  vec = pickle.load( open( "ml_vectorizer.sav", "rb" ) )
  model = pickle.load( open( f"model_{ambiance_label}.sav", "rb" ) )
  review_features = vec.transform([review])
  label = model.predict(review_features)
  return label


if __name__ == '__main__':
  train()
