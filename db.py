import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabeties_dataset = pd.read_csv('diabetes.csv')

diabeties_dataset.head()

diabeties_dataset.shape


diabeties_dataset.describe()
diabeties_dataset['Outcome'].value_counts()
diabeties_dataset.groupby('Outcome').mean()
x=diabeties_dataset.drop(columns = 'Outcome', axis=1)
y=diabeties_dataset['Outcome']
print(x)
print(y)
scaler = StandardScaler()
scaler.fit(x)
