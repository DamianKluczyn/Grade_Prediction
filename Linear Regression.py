import numpy as np
import tensorflow as tf
import keras
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

#import data
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "age", "studytime", "freetime", "goout", "health", "famrel"]]

#establish data
predict = "G3"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

#divide train and test data
x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.15)
