import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


#modify
df = pd.read_csv('./datasets/test.csv')
df.head()
df['IsAdult'] = np.where(df['Age'] >= 18, 1, 0)
df.to_csv("./datasets/test.csv")
