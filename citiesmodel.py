#import necessary modules
import pandas as pd
import numpy as  np
import random as rnd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier as rfc

world = pd.read_csv('world-cities.csv')
world.info()

print(world.head())
print('-'*90)
#citiescou = cities[['name','country']].groupby('country')
Country_city = world[['name','country']].groupby('country')['name'].count().reset_index().sort_values(by='name',ascending=False).reset_index(drop=True)
print(Country_city)
#plt.figure(figsize=(13,13))
#sns.barplot(x='name',y='country',data = Country_city)

India= world.loc[world['country']== 'India']
India_states = India.groupby('subcountry')['name'].count().reset_index().sort_values(by='name',ascending=False).reset_index(drop=True)
print(India_states)
#plt.figure(figsize=(13,13))
#sns.barplot(x='name',y='subcountry',data = India_states)


UnitedStates= world.loc[world['country']== 'United States']
UnitedStates_states = UnitedStates.groupby('subcountry')['name'].count().reset_index().sort_values(by='name',ascending=False).reset_index(drop=True)
print(UnitedStates_states)
print('-'*90)
plt.figure(figsize=(13,13))
sns.barplot(x='name',y='subcountry',data = UnitedStates_states)

plt.show()