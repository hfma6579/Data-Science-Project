#!/usr/bin/env python3
"""
An machine learning exercise with scikit-learn
"""
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def parse(src):
    """
    Parse input csv and performs data cleaning, converting, etc
    """
    df = pd.read_csv(src, index_col=0)

    # Sex
    df.loc[df.Sex == 'male', 'Sex'] = 0
    df.loc[df.Sex == 'female', 'Sex'] = 1

    # Pclass and embarked
    df = df.join(pd.get_dummies(df.Pclass, prefix='Pclass'))
    df = df.join(pd.get_dummies(df.Embarked, prefix='Embarked'))

    # cabin
    cabin = df.Cabin.str.extract('([A-Z])', expand=False)
    cabin.fillna('U', inplace=True)
    df = df.join(pd.get_dummies(cabin, prefix='Cabin'))

    # SibSp
    df.loc[df.SibSp > 1, 'SibSp'] = 2
    df = df.join(pd.get_dummies(df.SibSp, prefix='SibSp'))

    # Parch
    df.loc[df.Parch > 1, 'Parch'] = 2
    df = df.join(pd.get_dummies(df.Parch, prefix='Parch'))

    # Age
    scaler = StandardScaler()
    imputer = Imputer(strategy="median")
    age_arr = imputer.fit_transform(df['Age'].values.reshape(-1, 1))
    df['Age'] = scaler.fit_transform(age_arr)

    # Fare
    fare_arr = imputer.fit_transform(df['Fare'].values.reshape(-1, 1))
    df['Fare'] = scaler.fit_transform(fare_arr)

    # Names
    keys = ['Major.', 'Master.', 'Capt.', 'Dr.']
    for key in keys:
        df[key] = df.Name.str.contains(key).astype(int)

    X = df[['Sex', 'Age', 'Pclass_1', 'Pclass_2', 'Pclass_3',
            'SibSp_0', 'SibSp_1', 'SibSp_2', 'Parch_0', 'Parch_1', 
            'Parch_2', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
            'Cabin_A', 'Cabin_B', 'Cabin_C', 'Cabin_D', 'Cabin_E',
            'Cabin_F', 'Cabin_G', 'Cabin_U'] + keys]
    X = X.astype('float').values

    if 'Survived' in df.columns:
        return X,  df.Survived.values
    else:
        return X


def train(X, y):
    """
    Trainning the model. 
    """
    # split the data set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    classifier = SVC(gamma=0.2)
    result = cross_validate(classifier, X_train, y_train,
                            cv=10, return_train_score=False)
    classifier.fit(X_train, y_train)
    test_score = classifier.score(X_test, y_test)

    print('avg score in cross validation: {:0.4f}'.format(
        np.mean(result['test_score'])))
    print('score in test set: {:0.4f}'.format(test_score))

    return classifier


def main():
    INPUT = 'input/input.csv'
    TEST = 'input/test.csv'

    X, y = parse(INPUT)
    classfier = train(X, y)
    X_test = parse(TEST)
    y_test = classfier.predict(X_test)

    df = pd.read_csv(TEST, index_col=0)
    df['Survived'] = y_test
    df[['Survived']].to_csv('submission.csv')


if __name__ == '__main__':
    main()
