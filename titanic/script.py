"""
An machine learning exercise with scikit-learn
"""
import pandas as pd
import numpy as np
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.model_selection

N = 10  # Num of repetitions in cross validation


def parse(src):
    """
    Parse input csv and performs data cleaning, converting, etc
    """
    df = pd.read_csv(src)

    df.loc[df['Sex'] == 'male', 'Sex'] = 0
    df.loc[df['Sex'] == 'female', 'Sex'] = 1

    df.loc[df['Embarked'] == 'S', 'Embarked'] = 0
    df.loc[df['Embarked'] == 'C', 'Embarked'] = 1
    df.loc[df['Embarked'] == 'Q', 'Embarked'] = 2

    # df.dropna(inplace=True)
    return df


def fill_missing(X):
    imputer = sklearn.preprocessing.Imputer(strategy="median")
    return imputer.fit_transform(X)


def create_vec(df):
    """
    Create a feature vectors
    """
    # features: Sex, Age, Pclass, SibSp, Parch, Fare, Embarked
    return np.array([df['Sex'], df['Age'], df['Pclass'], df['SibSp'], df['Parch'], df['Fare'], df['Embarked']]).T


def train(df):
    """
    Trainning the model. The 
    """
    # using Logistic regression
    model = sklearn.linear_model.LogisticRegression()
    model = sklearn.
    X = create_vec(df)
    y = np.array(df['Survived'])

    # prepocessing
    X = fill_missing(X)

    # Evaluation
    evaluation = np.mean([train_single_split(X, y, model)
                          for i in range(N)], axis=0)
    print('Evaluation with {:d} repetitions:'.format(N))
    print('Accuracy: {:f}\nPrecision: {:f}\nRecall: {:f}'.format(*evaluation))

    # Using the whole dataset when predicting
    model.fit(X, y)
    return model


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


def predict(df, model):
    X = create_vec(df)
    X = fill_missing(X)
    return model.predict(X)


def main():
    input_df = parse('input/input.csv')
    test_df = parse('input/test.csv')
    model = train(input_df)
    ans = predict(test_df, model)
    ans_df = test_df[['PassengerId']].assign(Survived=ans)
    ans_df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
