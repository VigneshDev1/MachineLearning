import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as TTS
import sklearn.metrics as met

TitanicData = "https://sololearn.com/uploads/files/titanic.csv"

df = pd.read_csv(TitanicData)

df['male'] = df['Sex'] == "male"
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values
model = LogisticRegression()
Xtrain, Xtest, ytrain, ytest = TTS(X,y)
model.fit(Xtrain,ytrain)

print("Predict Proba:")
print(model.predict_proba(Xtest))
ypred = model.predict_proba(Xtest)[:,1]
print(ypred)
fpr, tpr, thresholds = met.roc_curve(ytest,ypred) #fpr -> false positive rates, true positive rates
plt.plot(fpr,tpr)
plt.show()
#print(df.head())