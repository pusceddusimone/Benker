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
    """
Function which prepares the training set for the model training, it also fits the vectorizer
    :return: the vectorizer trained, the x vectorized and the labels of the classes
    """
    train_path = os.path.join(os.path.dirname(__file__), 'dataset', 'banking-train.parquet')
    train_df = pd.read_parquet(train_path)

    X_train = train_df['text']
    y_train = train_df['label']
    # Finds the file with the user's suggestions
    train_user_path = os.path.join(os.path.dirname(__file__), 'dataset', 'banking-training-user.csv')
    data = pd.read_csv(train_user_path)
    y_user = data['label']
    y_train = pd.concat([y_train, y_user])
    # Preprocess the training data
    series_preprocessed = apply_preprocessing(X_train)
    series_user_suggested_preprocessed = apply_preprocessing(data['text'])
    series_preprocessed = pd.concat([series_preprocessed, series_user_suggested_preprocessed])
    vectorizer = TfidfVectorizer()

    X_train_transformed = vectorizer.fit_transform(series_preprocessed)
    return vectorizer, X_train_transformed, y_train


def train_model_svm():
    """
Trains the svm model
    :return: the vectorizer and svm trained
    """
    vectorizer, X_train_transformed, y_train = initial_train()

    model = svm.SVC(probability=True, C=2, gamma='scale')
    model.fit(X_train_transformed, y_train)

    return model, vectorizer


def train_model_logistic_regression():
    """
Trains the logistic regression model
    :return: the vectorizer and logistic regression trained
    """
    vectorizer, X_train_transformed, y_train = initial_train()

    model = LogisticRegression(penalty='l2', solver='lbfgs', C=15)
    model.fit(X_train_transformed, y_train)
    return model, vectorizer


def train_model_naive_bayes():
    """
Trains the naive bayes model
    :return: the vectorizer and naive bayes trained
    """
    vectorizer, X_train_transformed, y_train = initial_train()

    classifier = MultinomialNB(alpha=0.1)
    classifier.fit(X_train_transformed, y_train)
    return classifier, vectorizer


def apply_preprocessing(x):
    """
Applies the preprocessing to the text
    :param x: text to preprocess
    :return: text preprocessed
    """
    text_preprocessed = []
    for phrase in x:
        text_preprocessed.append(pp.preprocess(phrase))
    array_preprocessed = []
    for phrase_preprocessed in text_preprocessed:
        array_preprocessed.append([' '.join(phrase_preprocessed["message"][0])])
    series_preprocessed = pd.Series(list(chain.from_iterable(array_preprocessed)))
    return series_preprocessed


def calculate_accuracy(model, vectorizer):
    """
Calculates the model's accuracy
    :param model: the model to calculate the accuracy
    :param vectorizer:
    :return: the accuracy of the model
    """
    test_path = os.path.join(os.path.dirname(__file__), 'dataset', 'banking-test.parquet')
    test_df = pd.read_parquet(test_path)
    X_test = test_df['text']
    y_test = test_df['label']
    series_preprocessed = apply_preprocessing(X_test)
    text_to_predict_transformed = vectorizer.transform(series_preprocessed)
    y_pred = model.predict(text_to_predict_transformed)
    return "{:.4f}".format(accuracy_score(y_test, y_pred))


def generate_report(model, vectorizer):
    """
Generate a detailed report of the model's performance
    :param model:
    :param vectorizer:
    :return: the report of the model
    """
    test_path = os.path.join(os.path.dirname(__file__), 'dataset', 'banking-test.parquet')
    test_df = pd.read_parquet(test_path)
    X_test = test_df['text']
    y_test = test_df['label']
    series_preprocessed = apply_preprocessing(X_test)
    text_to_predict_transformed = vectorizer.transform(series_preprocessed)
    y_pred = model.predict(text_to_predict_transformed)
    return classification_report(y_test, y_pred, output_dict=True)


# This functions are used during development, they are used not to train the model every time
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
