"""
An machine learning exercise with scikit-learn
"""
import pandas as pd
import numpy as np
import re
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.model_selection
import sklearn.feature_selection
import sklearn.tree


N = 20  # Num of repetitions in cross validation


def parse(src):
    """
    Parse input csv and performs data cleaning, converting, etc
    """
    df = pd.read_csv(src)

    df.loc[df['Sex'] == 'male', 'Sex'] = 0
    df.loc[df['Sex'] == 'female', 'Sex'] = 1

    # Pclass and embarked
    df = df.join(pd.get_dummies(df.Pclass, prefix='Pclass'))
    df = df.join(pd.get_dummies(df.Embarked, prefix='Embarked'))

    # cabin
    cabin = df.Cabin.str.extract('([A-Z])', expand=False)
    cabin.fillna('U', inplace=True)
    s = pd.get_dummies(cabin, prefix='Cabin')
    df = df.join(s)


    return df


def fill_missing(X):
    """
    Fill in the missing value
    """
    imputer = sklearn.preprocessing.Imputer(strategy="most_frequent")
    return imputer.fit_transform(X)


def create_vec(df):
    """
    Create a feature vectors
    """
    # features: Sex, Age, Pclass, SibSp, Parch, Fare, Embarked
    features = ['Sex', 'Age', 'Pclass_1', 'Pclass_2', 'Pclass_3',
                'SibSp', 'Parch', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
                'Fare', 'Cabin_A', 'Cabin_B', 'Cabin_C', 'Cabin_D', 'Cabin_E', 
                'Cabin_F', 'Cabin_G', 'Cabin_U']
    return df[features]


def train(df):
    """
    Trainning the model. 
    """
    # using Logistic regression
    # model = sklearn.linear_model.LogisticRegression()

    # using Decision Tree
    model = sklearn.tree.DecisionTreeClassifier()

    X = create_vec(df)
    y = np.array(df['Survived'])

    # prepocessing
    X = fill_missing(X) 
    
    # get important features
    selector = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2, 4)
    selector.fit(X, y)
    X = selector.transform(X)

    # Evaluation
    evaluation = np.mean([train_single_split(X, y, model)
                          for i in range(N)], axis=0)
    print('Evaluation with {:d} repetitions:'.format(N))
    print('Accuracy: {:f}'.format(evaluation[0]))

    # Using the whole dataset when predicting
    model.fit(X, y)
    return model, selector


def train_single_split(X, y, model):
    """
    Trainning data with random split
    """
    # split data into training set and testing set
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y)

    # fit model
    model.fit(X_train, y_train)

    # get the accuracy
    y_guess = model.predict(X_test)
    y_correct = y_guess == y_test

    # get measurement
    accuracy = np.sum(y_correct) / len(y_correct)
    precision = np.sum(y_correct[y_guess == 1]) / len(y_test[y_guess == 1])
    recall = np.sum(y_correct[y_test == 1]) / len(y_test[y_test == 1])

    return accuracy, precision, recall


def predict(df, model, selector):
    X = create_vec(df)
    X = fill_missing(X)
    X = selector.transform(X)
    return model.predict(X)


def main():
    input_df = parse('input/input.csv')
    test_df = parse('input/test.csv')
    model, selector = train(input_df)
    ans = predict(test_df, model, selector)
    ans_df = test_df[['PassengerId']].assign(Survived=ans)
    ans_df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
