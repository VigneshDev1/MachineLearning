''' Note: This is a practice on using scikit-learn, panda and numpy

scikit-learn has a build in breast cancer dataset
'''

from sklearn.datasets import load_breast_cancer
import pandas as pd


cancer_data = load_breast_cancer()

print(cancer_data.keys())
otherinfo=[]
for i in cancer_data.keys():
    try:
        print("{0} has shape {1}".format(i, cancer_data[i].shape))
    except:
        if cancer_data[i] is not None:
            otherinfo.append(i + " [*******] " + cancer_data[i])

