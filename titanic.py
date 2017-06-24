#import necessary modules
import pandas as pd
import numpy as  np
import random as rnd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier as rfc

#read data from csv files using pandas and combine train and test datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
combine = [train,test]

#view and analyze data in train and test set
print(train.columns.values)
train.info()
test.info()
print(train.describe())
print(train.describe(include = ["O"]))
print('_'*90)
#look for correlations in  between columns of data
g = sns.FacetGrid(train,col="Survived")
g.map(plt.hist,"Age",bins=10)
plt.show()
print(train[["Pclass","Survived"]].groupby(['Pclass']).mean())

#Modify and create feautures to be used during training the randomforest model
#fill NaN values using mode or median or fillna methods
mode_embarked = train['Embarked'].dropna().mode()[0]
for each in combine:
	each['Embarked'] = each['Embarked'].fillna(mode_embarked)
	each['Embarked'] = each['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)

# Using Titles in Name column 
for each in combine:
	each['Title'] = each['Name'].str.extract('([A-Za-z+])\.',expand = False)

for each in combine:
    each['Title'] = each['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    each['Title'] = each['Title'].replace('Mlle', 'Miss')
    each['Title'] = each['Title'].replace('Ms', 'Miss')
    each['Title'] = each['Title'].replace('Mme', 'Mrs')

for each in combine:
	each['Title'] = each['Title'].map({'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Rare':5})
	each['Title'] = each['Title'].fillna(0)
	each['Sex'] = each['Sex'].map({'female':1,'male':0}).astype(int)

#Filling NaN in Age Column
guess_ages = np.zeros((2,3))
for each in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = each[(each['Sex'] == i) & (each['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            each.loc[ (each.Age.isnull()) & (each.Sex == i) & (each.Pclass == j+1),'Age'] = guess_ages[i,j]

    each['Age'] = each['Age'].astype(int)
train['Ageband']= pd.cut(train['Age'],5)
for each in combine:   
    each.loc[ each['Age'] <= 16, 'Age'] = 0
    each.loc[(each['Age'] > 16) & (each['Age'] <= 32), 'Age'] = 1
    each.loc[(each['Age'] > 32) & (each['Age'] <= 48), 'Age'] = 2
    each.loc[(each['Age'] > 48) & (each['Age'] <= 64), 'Age'] = 3
    each.loc[ each['Age'] > 64, 'Age']



test['Fare'] = test['Fare'].fillna(test['Fare'].dropna().median())

#remove feautures that are irrelevant to final result
train = train.drop(['PassengerId','Name','Ticket','Cabin','Age'],axis=1)
test = test.drop(['Name','Ticket','Cabin'],axis=1)
combine = [train,test]

#create final feautures for model training and testing
xtrain = train.drop(['Survived'], axis=1)
ytrain = train['Survived']
xtest = test.drop(['PassengerId'],axis=1)
xtrain.info()
xtest.info()
print(ytrain.head())

#train model using scikit learn
random_forest = rfc(n_estimators=100)
random_forest.fit(xtrain, ytrain)

#predict for the test file using trained model
ypred = random_forest.predict(xtest)

#score your model
acc_random_forest = round(random_forest.score(xtrain, ytrain) * 100, 2)
print(acc_random_forest)

#create submission DataFrame
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':ypred})

#write the output to csv file in present directory
submission.to_csv('submission.csv',index = False)
