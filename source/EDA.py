import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# caminhos
dataset_path = "input/census_income_dataset.csv"

# lendo
df = pd.read_csv(dataset_path)

# initial EDA
# info(), head(), tail() e describe() from data
# We can see from info() that there are no missing values on the dataset
print(df.info())
print(df.shape)
#print(df.head())
#print(df.tail())
#print(df.describe())

# How many single values do we have per (object) column?
for c in df.columns:
    print('\nColumn:',c)
    print(df[c].value_counts())
    print(df[c].value_counts().shape)
    print(df.groupby([c,'income_level'])['income_level'].count())

# feature capital gain and captain loss apper not to make sense on the model due to their number of zeros.