''' Note: This is a practice on using scikit-learn, panda and numpy

scikit-learn has a build in breast cancer dataset
'''

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import pandas as pd


cancer_data = load_breast_cancer()  # This is a dictionary with all sorts of information for the dataset
print(cancer_data.keys())           # {'data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'}

otherinfo=[]
for i in cancer_data.keys():
    try:
        print("{0} has shape {1}".format(i, cancer_data[i].shape))
    except:
        if cancer_data[i] is not None:
            otherinfo.append(i + " [*******] " + cancer_data[i])
print(cancer_data['DESCR'])

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

cancer_df = pd.DataFrame(cancer_data['data'], columns = cancer_data['feature_names']) # columns are titles

print(cancer_df.head())
print(cancer_data['feature_names'])

cancer_df['target'] = cancer_data['target']
print(cancer_df.head())

X = cancer_df[cancer_data.feature_names].values  # remember to say values (we want the values stored under X).
# ie. the above reads the dict cancer data >> element features_names which is a list of all column header
# and creates an another mini dataframe. TIP: newDF = oldDF[[column_A_name, column_C_name, column_G_name]] note the [[]]

y= cancer_df['target'].values
#-----------------------------------------
# model = LogisticRegression()
# model.fit(X, y)
# print(model.coef_, model.intercept_)

# When we run the above code we get a Convergence Warning.
# This means that the model needs more time to find the optimal solution.
# One option is to increase the number of iterations.
# You can also switch to a different solver.
# The solver is the algorithm that the model uses to find the equation of the line.
# You can see the possible solvers in the Logistic Regression documentation

#-----------------------------------------


model = LogisticRegression(solver='liblinear')
model.fit(X, y)

print(model.coef_, model.intercept_)

# Now using the determined model formula, we are testing the data inputs and comparing results with y
y_predicted = model.predict(X)

print(sum(y == y_predicted))


print(type(cancer_df['target']))

