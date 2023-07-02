import os
from itertools import chain

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
from sklearn.naive_bayes import MultinomialNB

import preprocessing as pp


def initial_train():
    train_path = os.path.join(os.path.dirname(__file__), 'dataset', 'banking-train.parquet')
    train_df = pd.read_parquet(train_path)

    X_train = train_df['text']
    y_train = train_df['label']
    train_user_path = os.path.join(os.path.dirname(__file__), 'dataset', 'banking-training-user.csv')
    data = pd.read_csv(train_user_path)
    y_user = data['label']
    y_train = pd.concat([y_train, y_user])
    series_preprocessed = apply_preprocessing(X_train)
    series_user_suggested_preprocessed = apply_preprocessing(data['text'])
    series_preprocessed = pd.concat([series_preprocessed, series_user_suggested_preprocessed])
    vectorizer = TfidfVectorizer()

    X_train_transformed = vectorizer.fit_transform(series_preprocessed)
    return vectorizer, X_train_transformed, y_train


def train_model_svm():
    vectorizer, X_train_transformed, y_train = initial_train()

    model = svm.SVC(probability=True)
    model.fit(X_train_transformed, y_train)

    return model, vectorizer



def train_model_logistic_regression():
    vectorizer, X_train_transformed, y_train = initial_train()

    model = LogisticRegression()
    model.fit(X_train_transformed, y_train)
    return model, vectorizer


def train_model_naive_bayes():
    vectorizer, X_train_transformed, y_train = initial_train()

    classifier = MultinomialNB()
    classifier.fit(X_train_transformed, y_train)
    return classifier, vectorizer


def apply_preprocessing(x):
    text_preprocessed = []
    for phrase in x:
        text_preprocessed.append(pp.preprocess(phrase))
    array_preprocessed = []
    for phrase_preprocessed in text_preprocessed:
        array_preprocessed.append([' '.join(phrase_preprocessed["message"][0])])
    series_preprocessed = pd.Series(list(chain.from_iterable(array_preprocessed)))
    return series_preprocessed


def calculate_accuracy(model, vectorizer):
    test_path = os.path.join(os.path.dirname(__file__), 'dataset', 'banking-test.parquet')
    test_df = pd.read_parquet(test_path)
    X_test = test_df['text']
    y_test = test_df['label']
    series_preprocessed = apply_preprocessing(X_test)
    text_to_predict_transformed = vectorizer.transform(series_preprocessed)
    y_pred = model.predict(text_to_predict_transformed)
    return "{:.4f}".format(accuracy_score(y_test, y_pred))


def generate_report(model, vectorizer):
    test_path = os.path.join(os.path.dirname(__file__), 'dataset', 'banking-test.parquet')
    test_df = pd.read_parquet(test_path)
    X_test = test_df['text']
    y_test = test_df['label']
    series_preprocessed = apply_preprocessing(X_test)
    text_to_predict_transformed = vectorizer.transform(series_preprocessed)
    y_pred = model.predict(text_to_predict_transformed)
    return classification_report(y_test, y_pred, output_dict=True)


def dump_model(model, model_filename):
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)


def load_model(model_filename):
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
        return model


def dump_vectorizer(vectorizer, vectorizer_filename):
    with open(vectorizer_filename, 'wb') as file:
        pickle.dump(vectorizer, file)


def load_vectorizer(vectorizer_filename):
    with open(vectorizer_filename, 'rb') as file:
        vectorizer = pickle.load(file)
        return vectorizer
