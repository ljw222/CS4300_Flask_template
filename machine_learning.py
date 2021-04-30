from sklearn.svm import SVC
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

ambiance_labels = ['touristy', 'classy', 'romantic', 'casual', 'hipster', 'divey', 'intimate', 'trendy', 'upscale']

def train():
  # use this for training/saving models
  with open("finalData2.json", "r") as f:
      data = json.load(f)
  # df = pd.DataFrame(columns=['restaurant', 'reviews', 'price', 'stars'] + ambiance_labels)
  df = pd.DataFrame(columns=['restaurant', 'reviews'] + ambiance_labels)
  for restaurant, restaurant_dict in data['BOSTON'].items():
    ambiances = restaurant_dict['ambience']
    review_texts = [r['text'] for r in restaurant_dict['reviews']]
    # review_stars = [r['stars'] for r in restaurant_dict['reviews']]
    review_combined = '\n'.join(review_texts)
    # stars_avg = sum(review_stars)/len(review_stars)
    new_row = dict()
    new_row['restaurant'] = restaurant
    new_row['reviews'] = review_combined
    # new_row['price'] = restaurant_dict['price']
    # new_row['stars'] = stars_avg
    for label in ambiance_labels:
        if label in ambiances:
            new_row[label] = 1
        else:
            new_row[label] = 0
    if len(ambiances) > 0:
        df = df.append(new_row, ignore_index=True)
  # for col in ambiance_labels + ['price']:
  for col in ambiance_labels:
    df[col] = df[col].astype(int)

  # X = df[['reviews', 'stars', 'price']]
  X = df['reviews']
  Y = df[ambiance_labels]
  print(Y.shape)
  vec = TfidfVectorizer()
  X_features = pd.DataFrame(vec.fit_transform(X).toarray())
  with open("ml_vectorizer.sav", "wb") as f:
    pickle.dump(vec, f)
  # Xtrain_features['stars'] = X_train['stars'].reset_index()['stars']
  # Xtrain_features['price'] = X_train['price'].reset_index()['price']
  # Xtest_features['stars'] = X_test['stars'].reset_index()['stars']
  # Xtest_features['price'] = X_test['price'].reset_index()['price']

  svm = SVC()
  # param_grid = {'C': [1, 10, 100], 
  #             'gamma': [1, 0.1, 0.01],
  #             'kernel': ['rbf', 'linear']} 
  param_grid = {'C': [20, 10, 1, 0.5, 0.25, 0.1]}

  models = dict()
    
  grid = GridSearchCV(LogisticRegression(), param_grid, refit = True, verbose = 0, scoring="f1_weighted")
  for col in ambiance_labels:
    print(col)
    X_train, X_test, y_train, y_test = train_test_split(X_features, Y[col], test_size=0.3, random_state=42)
    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
    print("Positive: ", len(y_train[y_train==1]))
    print("Negative: ", len(y_train[y_train==0]))
    grid.fit(X_train, y_train)
    print(grid.best_score_)
    print(grid.best_estimator_)
    model = grid.best_estimator_
    preds = model.predict(X_test)
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))
    models[col] = model

  print(models)
  for col in ambiance_labels:
    pickle.dump(models[col], open(f"model_{col}.sav", "wb"))


def predict(review, ambiance_label):
  # predicts if an ambiance label should be applied for some review
  vec = pickle.load( open( "ml_vectorizer.sav", "rb" ) )
  model = pickle.load( open( f"model_{ambiance_label}.sav", "rb" ) )
  review_features = vec.transform([review])
  label = model.predict(review_features)
  return label


if __name__ == '__main__':
  train()
