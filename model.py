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
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    filename = 'model.pkl'
    pickle.dump(classifier, open(filename, 'wb'))
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy: " + str(acc))
    return acc

df = pd.read_csv('healthcare-dataset-stroke-data.csv')
bmi = round(df['bmi'].mean(),1)
df['bmi'].fillna(bmi, inplace=True)
bins = [0,18,36,54,72,90]
labels = ['0-18','18-36','36-54','54-72','72-90']
df['age'] = pd.cut(df['age'], bins=bins, labels=labels)
df = df[(df['bmi'] > 10.3) & (df['bmi'] < 43)]
del df['id']
columns = ['gender', 'age', 'hypertension', 'ever_married','work_type', 'Residence_type', 'smoking_status']
df = label_encoder(df,columns)
majority = df[df['stroke'] == 0]
minority = df[df['stroke'] == 1]
upsampled = resample(minority, replace=True, n_samples=len(majority))
df = pd.concat([majority,upsampled])
df = df.sample(frac=1).reset_index(drop=True)
# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values
# X_train, X_test, y_train, y_test = split_dataset(X,y)
# X_train,X_test = scaling(X_train,X_test)
# accuracy = Classifier(X_train, X_test, y_train, y_test)
print(df.head())