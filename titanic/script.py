"""
An machine learning exercise with scikit-learn
"""
import pandas as pd
import numpy as np
import sklearn.linear_model
import sklearn.model_selection

N = 20  # Num of repetitions in cross validation


def parse(src):
    """
    Parse input csv and performs data cleaning, converting, etc
    """
    df = pd.read_csv(src)

    df.loc[df['Sex'] == 'male', 'Sex'] = 0
    df.loc[df['Sex'] == 'female', 'Sex'] = 1

    # Currectly null values cannot be handled properly
    df.dropna(inplace=True)
    return df


def create_vec(df):
    """
    Create a feature vectors
    """
    # only use two features
    return np.array([df['Sex'], df['Age']]).T


def train(df):
    """
    Trainning the model. The 
    """
    # using Logistic regression
    model = sklearn.linear_model.LogisticRegression()
    X = create_vec(df)
    y = np.array(df['Survived'])

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
    x = create_vec(df)
    return model.predict(x)


def main():
    input_df = parse('input/input.csv')
    test_df = parse('input/test.csv')
    model = train(input_df)
    ans = predict(test_df, model)


if __name__ == '__main__':
    main()
