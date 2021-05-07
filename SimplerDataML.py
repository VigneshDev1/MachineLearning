import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as met

TitanicData = "https://sololearn.com/uploads/files/titanic.csv"

df = pd.read_csv(TitanicData)

df['male'] = df['Sex'] == "male"
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values
print(y)
print(df.head())