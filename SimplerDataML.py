import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as TTS, KFold

import sklearn.metrics as met


def GetModelMetric(model, X, y):
    #ROC - Receiver Operating Characteristics
    # When predicting values, model bases on probability value from 0 ~ 1. for logistic regression the threshold is 0.5
    # higher threshold means better precision (confident with +ve accuracy) but lower recall. vice versa is true
    # with lowering the threshold, high recall means catching all the positive cases, but lower precision.

    # Define the threshold for target's 0 or 1, and thing lower than threshold will round to false and higher to true.
    # Determine predicted results

    y_predicted = model.predict_proba(X)[:, 1]
    #y_predicted = model.predict(X)
    one_minus_Specificity, Sensitivity, thresholds = met.roc_curve(y, y_predicted)

    #print(one_minus_Specificity)
    plt.plot(one_minus_Specificity,Sensitivity)
    plt.show()
    # Now using the determined model formula, we are testing the data inputs and comparing results with y
    # y_predicted = model.predict(X)
    print(model.score(X, y))  # same as sum(y == y_predicted)/y.shape[0]
    y_predicted = y_predicted>0.75
    [[TN, FP], [FN, TP]] = met.confusion_matrix(y, y_predicted)  # result format : [[TN, FN], [FP, TP]]
    Specificity = TN / (TN + FP) # Specificity = met.precision_recall_fscore_support(y,y_predicted)

    metric = {"Accuracy": met.accuracy_score(y, y_predicted), "Precision": met.precision_score(y, y_predicted),
              "Recall/Sensitivity": met.recall_score(y, y_predicted),
              "Specificity": Specificity, "F1 Score": met.f1_score(y, y_predicted)}  # note accuracy is the same score
    return metric

    print("Accuracy: {0:.2%} \n"
          "Precision: {1:.2%} \n"
          "Recall/Sensitivity: {2:.2%} \n"
          "Specificity: {3:.2%} \n"
          "F1 Score: {4:.2%}".format(metric['Accuracy'], metric['Precision'], metric['Recall/Sensitivity'],
                                     metric['Specificity'], metric['F1 Score']))

    print("      Act+          Act- \n"
          "Pred+ {}          {} \n"
          "Pred- {}          {}".format(TP, FP,
                                        FN, TN))

#_______________________________________________________________________________________



TitanicData = "https://sololearn.com/uploads/files/titanic.csv"

df = pd.read_csv(TitanicData)

df['male'] = df['Sex'] == "male"
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values
model = LogisticRegression()
#_________Using Sklearn's Test & Train Model_________
Xtrain, Xtest, ytrain, ytest = TTS(X,y)

#__________Using K-Fold cross validation approach____________________

X = df[['Age', 'Fare']].values[:6]
y = df[['Survived']].values[:6]
kFold = KFold(n_splits=3,shuffle=True) # good practice to shuffle the data, n split is the # of chunks to cluster data
# KFold is a generator
print(list(kFold.split(X)))

for trainset, testset in kFold.split(X):
    print("Train data: {}   <>   Test data: {}".format(trainset,testset))
model.fit(Xtrain,ytrain)
metric = GetModelMetric(model,Xtest,ytest)

#print(df.head())