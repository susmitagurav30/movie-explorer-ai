import pandas as pd

data = pd.read_csv("Movie_Review.csv")

print(data.head())
print(data.isnull().sum())