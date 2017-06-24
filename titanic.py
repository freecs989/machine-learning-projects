import pandas as pd
import numpy as  np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('C:/Users/hai/Desktop/matplotlib/kaggle/titanic/train.csv')
test = pd.read_csv('C:/Users/hai/Desktop/matplotlib/kaggle/titanic/test.csv')
combine = [train,test]



print(train.columns.values)

train.info()
print('_'*40)
test.info()
print(train.describe())
print(train.describe(include = ["O"]))

g = sns.FacetGrid(train,col="Survived")
g.map(plt.hist,"Age",bins=10)
plt.show()
print(train[["Pclass","Survived"]].groupby(['Pclass']).mean())

##print(train.columns.values)
##print(test.columns.values)
inr = pd.read_html(r'https://raw.githubusercontent.com/MOOCDataAnalysis/datasets/master/ex103x/q1/requests.csv')
print(inr)

