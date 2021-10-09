import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
import pickle
from sklearn.naive_bayes import GaussianNB

encoder = LabelEncoder()
def label_encoder(df,columns):
    for i in columns:
        df[i] = encoder.fit_transform(df[i])
    df.head()
    return df

def split_dataset(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    return X_train, X_test, y_train, y_test

sc = StandardScaler()
def scaling(X_train,X_test):
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train,X_test

def Classifier(X_train, X_test, y_train, y_test):
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    filename = 'model.pkl'
    pickle.dump(classifier, open(filename, 'wb'))
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy: " + str(acc))
    return acc

df = pd.read_csv('heart.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = split_dataset(X,y)
X_train,X_test = scaling(X_train,X_test)
accuracy = Classifier(X_train, X_test, y_train, y_test)