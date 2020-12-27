# Python script for ISM
# SVM: Support Vector Machines (Are non-probabilistic binary linear classifier)

# ========================================================================================================================================================== #
# ========================================================================================================================================================== #
# imports
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import GridSearchCV

cancer = pd.read_csv('./breast_cancer_data.csv') #/Users/albertoperdomogarcia/Documents/00-Documents_A/00-Dev/Python/Python_MASTER/ISM/breast_cancer
cancer = cancer.drop('id', axis=1) # id not relevant
cancer = cancer.drop('Unnamed: 32', axis=1) # pandas issue
# print(cancer.head(5))
# sns.distplot(cancer['symmetry_worst']) # print whatever variable you want to plot to do some exploratory analisis

# split the data in X and y
X = cancer.drop('diagnosis', axis=1) # X does not contain labels
decoded_y = cancer['diagnosis'] # only labels

# encode class values as integers (0==B or 1==M)
encoder = LabelEncoder()
encoder.fit(decoded_y)
y = encoder.transform(decoded_y)

# split the data in training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# model
svm = SVC()
svm.fit(X_train, y_train) # train
predictions = svm.predict(X_test)

print("================================== FIRST RESULTS ==================================")
print('CONFUSION MATRIX: \n', confusion_matrix(y_test, predictions))
print('\n CLASSIFICATION REPORT: \n', classification_report(y_test, predictions))
print('\n ACCURACY OF THE MODEL: ', accuracy_score(y_test, predictions))
print("\n===================================================================================")

# grid search for finding the best parameters for the model
param_grid = {'C':[0.1, 1, 10, 100, 1000], 'gamma':[1, 0.1, 0.001, 0.0001]} # this test parameters are set randomly
grid = GridSearchCV(SVC(), param_grid, verbose=0)
grid.fit(X_train, y_train)

results = pd.Series(grid.cv_results_)

print('\n BEST SET OF PARAMETERS FOR THE MODEL: \n', grid.best_params_)
print('\n')

# make the predictions with the best parameters
grid_predictions = grid.predict(X_test)

print("================================== FINAL RESULTS ==================================")
print('CONFUSION MARTRIX (For the best set of parameters): \n', confusion_matrix(y_test, grid_predictions))
print('\n CLASSIFICATION REPORT (For the best set of parameters): \n', classification_report(y_test, grid_predictions))
print('\n ACCURACY OF THE MODEL (For the best set of parameters): ', accuracy_score(y_test, grid_predictions))
print('\n')

# plot non-normalized confusion matrix
titles_options = [("CONFUSION MATRIX", None),
                  ("CONFUSION MATRIX (Normalized)", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(grid, X_test, y_test, normalize=normalize, cmap='viridis')
    disp.ax_.set_title(title)

# print the plot
plt.show()