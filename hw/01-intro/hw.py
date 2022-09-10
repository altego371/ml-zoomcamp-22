import numpy as np
import pandas as pd


def hw():
    print("Q1: ", q1())
    print("Q2: ", q2())
    print("Q3: ", q3())
    print("Q4: ", q4())
    print("Q5: ", q5())
    print("Q6: ", q6())
    print("Q7: ", q7())


def q1():
    """What's the version of NumPy that you installed?"""
    return np.__version__


def q2():
    """How many records are in the dataset?"""
    df = pd.read_csv('data.csv')
    return df.shape[0]


def q3():
    """Who are the most frequent car manufacturers (top-3) according to the dataset?"""
    df = pd.read_csv('data.csv')
    return df.groupby('Make').size().sort_values(ascending=False).index.tolist()[:3]


def q4():
    """What's the number of unique Audi car models in the dataset?"""
    df = pd.read_csv('data.csv')
    return len(df[df['Make'] == 'Audi']['Model'].unique())


def q5():
    """How many columns in the dataset have missing values?"""
    df = pd.read_csv('data.csv')
    return sum(df.isnull().any())


def q6():
    """
    Find the median value of "Engine Cylinders" column in the dataset.
    Next, calculate the most frequent value of the same "Engine Cylinders".
    Use the fillna method to fill the missing values in "Engine Cylinders" with the most frequent value from the previous step.
    Now, calculate the median value of "Engine Cylinders" once again.
    Has it changed?
    """
    df = pd.read_csv('data.csv')
    median_0 = df['Engine Cylinders'].median()
    # most_frequent = df.groupby('Engine Cylinders').size().sort_values(ascending=False).index.tolist()[0]
    most_frequent = df['Engine Cylinders'].mode()[0]
    median_1 = df['Engine Cylinders'].fillna(most_frequent).median()
    return median_0 != median_1


def q7():
    """
    Select all the "Lotus" cars from the dataset.
    Select only columns "Engine HP", "Engine Cylinders".
    Now drop all duplicated rows using drop_duplicates method (you should get a dataframe with 9 rows).
    Get the underlying NumPy array. Let's call it X.
    Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
    Invert XTX.
    Create an array y with values [1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800].
    Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
    What's the value of the first element of w?
    """
    df = pd.read_csv('data.csv')
    X = df[df['Make'] == 'Lotus'][['Engine HP', 'Engine Cylinders']].drop_duplicates()
    XTX = X.T.dot(X)
    XTX_inv = pd.DataFrame(np.linalg.pinv(XTX.values), XTX.columns, XTX.index)
    XTX_inv.dot(XTX)  # check XTX_inv
    y = pd.array([1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800])
    w = XTX_inv.dot(X.T).dot(y)
    return w.tolist()[0]
