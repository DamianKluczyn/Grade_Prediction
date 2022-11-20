import numpy as np
import tensorflow as tf
import keras
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


#import data
data = pd.read_csv("student-mat.csv", sep=";")

"""
#replace non-integer values to integers
data[["paid"]]=data[["paid"]].replace("yes", 1)
data[["paid"]]=data[["paid"]].replace("no", 0)"""

data = data[["G1", "G2", "G3", "age", "studytime", "freetime", "health", "failures", "absences"]]

#establish data
predict = "G3"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

#divide train and test data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#pick best model and save it using pickle
"""
best = 0
while(best < 0.95):
    # divide train and test data
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    #train model
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    #take accuracy
    accuracy = linear.score(x_test, y_test)
    print("Accuracy: ", accuracy)

    if accuracy > best:
        best = accuracy
        #save model
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
"""

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Co (m): ", linear.coef_)
print("Intercept (b): ", linear.intercept_)

#predict final grade
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

#show model
p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data[predict])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()