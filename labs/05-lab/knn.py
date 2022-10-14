import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.spatial import distance_matrix

class KNN:

    def __init__(self, k, X_train, y_train):
        self.k = k
        self.X_train = X_train
        self.y_train = y_train
        self.distance_matrix = None
    
    def train(self):
        # https://scikit-learn.org/0.24/modules/generated/sklearn.neighbors.DistanceMetric.html
        self.distance_matrix = distance_matrix(self.X_train, self.y_train)

    def predict(self, example):
        return ...

    def get_error(self, predicted, actual):
        return sum(map(lambda x : 1 if (x[0] != x[1]) else 0, zip(predicted, actual))) / len(predicted)

    def test(self, test_input, labels):
        actual = labels
        predicted = (self.predict(test_input))
        print("error = ", self.get_error(predicted, actual))

# Add the dataset here
# from https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
iris = datasets.load_iris()
X = iris.data
y = iris.target

print(iris)
# Split the data 70:30 and predict.
# 70 for testing, 30 for training 
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, train_size=70 )
print(" x_train", len(X_train))
print(" x_test", len(X_test))
# create a new object of class KNN
k = KNN(3, X_train, y_train)
print(KNN.train(k))

# plot a boxplot that is grouped by Species. 
# You may have to ignore the ID column

# predict the labels using KNN

# use the test function to compute the error
