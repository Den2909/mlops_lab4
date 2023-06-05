import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


#Modification of the dataset Binarization of the "Age" attribute by creating a new "IsAdult" attribute (adult), which will have a value of 1 if the passenger's age is greater than or equal to 18,
# and 0 otherwise. This will highlight the influence of age on survival.
df = pd.read_csv('./datasets/test.csv')
df.head()
df['IsAdult'] = np.where(df['Age'] >= 18, 1, 0)
df.to_csv("./datasets/test.csv")

#Filling in the missing values in the "Age" field with the average value
imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df['Age'].values.reshape(-1, 1))

df.head()

df.to_csv("./datasets/test.csv")
