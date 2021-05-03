# Note: This is a practice on using scikit-learn, panda and numpy
# scikit-learn has a build in breast cancer dataset

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as met
from sklearn.model_selection import train_test_split as TTS
import pandas as pd
from matplotlib import pyplot as plt

def GetModelMetric(model, X, y):
    #ROC - Receiver Operating Characteristics
    # When predicting values, model bases on probability value from 0 ~ 1. for logistic regression the threshold is 0.5
    # higher threshold means better precision (confident with +ve accuracy) but lower recall. vice versa is true
    # with lowering the threshold, high recall means catching all the positive cases, but lower precision.

    # Define the threshold for target's 0 or 1, and thing lower than threshold will round to false and higher to true.
    # Determine predicted results

    y_predicted = model.predict_proba(X)[:, 1]
    Specificity, Sensitivity, thresholds = met.roc_curve(y, y_predicted)
    plt.plot(Specificity,Sensitivity)
    # Now using the determined model formula, we are testing the data inputs and comparing results with y
    # y_predicted = model.predict(X)
    print(model.score(X, y))  # same as sum(y == y_predicted)/y.shape[0]
    [[TN, FP], [FN, TP]] = met.confusion_matrix(y, y_predicted)  # result format : [[TN, FN], [FP, TP]]
    Specificity = TN / (TN + FP) # Specificity = met.precision_recall_fscore_support(y,y_predicted)

    metric = {"Accuracy": met.accuracy_score(y, y_predicted), "Precision": met.precision_score(y, y_predicted),
              "Recall/Sensitivity": met.recall_score(y, y_predicted),
              "Specificity": Specificity, "F1 Score": met.f1_score(y, y_predicted)}  # note accuracy is the same score

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


cancer_data = load_breast_cancer()  # This is a dictionary with all sorts of information for the dataset
print(cancer_data.keys())  # {'data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'}

otherinfo = []
for i in cancer_data.keys():
    try:
        print("{0} has shape {1}".format(i, cancer_data[i].shape))
    except:
        if cancer_data[i] is not None:
            otherinfo.append(i + " ------ " + cancer_data[i])


#print(cancer_data['target_names'])
#   Features Titles:
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#      'mean smoothness' 'mean compactness' 'mean concavity'
#      'mean concave points' 'mean symmetry' 'mean fractal dimension'
#      'radius error' 'texture error' 'perimeter error' 'area error'
#      'smoothness error' 'compactness error' 'concavity error'
#      'concave points error' 'symmetry error' 'fractal dimension error'
#      'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#      'worst smoothness' 'worst compactness' 'worst concavity'
#      'worst concave points' 'worst symmetry' 'worst fractal dimension']

cancer_df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])  # columns are titles
print(cancer_data['feature_names'])
cancer_df['target'] = cancer_data['target']
print(cancer_df.head())

X = cancer_df[cancer_data.feature_names].values  # remember to say values (we want the values stored under X).
# ie. the above reads the dict cancer data >> element features_names which is a list of all column header
# and creates an another mini dataframe. TIP: newDF = oldDF[[column_A_name, column_C_name, column_G_name]] note the [[]]

y = cancer_df['target'].values
# -----------------------------------------
# model = LogisticRegression()
# model.fit(X, y)
# print(model.coef_, model.intercept_)

# When we run the above code we get a Convergence Warning.
# This means that the model needs more time to find the optimal solution.
# One option is to increase the number of iterations.
# You can also switch to a different solver.
# The solver is the algorithm that the model uses to find the equation of the line.
# You can see the possible solvers in the Logistic Regression documentation

# -----------------------------------------


model = LogisticRegression(solver='liblinear')
model.fit(X, y)

print(["{:.2f}".format(coef) for coef in model.coef_[0]], model.intercept_)
# Now using the determined model formula, we compare the predicted results with y and get the metrics
GetModelMetric(model, X, y)
# _____________Split data into training and test data __________________
# train_size can be used to define % split between test and train
# random_stat can be used to permit the code to use the same test data set (int mean the state, keep same for same data)
[Xtrain, Xtest, ytrain, ytest] = TTS(X, y, train_size=0.65, random_state=12)
print(X.shape, y.shape, Xtrain.shape, ytrain.shape, Xtest.shape, ytest.shape)  # checking the size of shapes

model.fit(Xtrain, ytrain)
GetModelMetric(model, Xtest, ytest)
print(model.score(Xtest, ytest))  # testing the model against the test data
