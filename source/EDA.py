import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import LabelEncoder
# caminhos
dataset_path = "input/census_income_dataset.csv"

# lendo
df = pd.read_csv(dataset_path)
df.drop(['capital_loss', 'capital_gain'], inplace=True, axis=1)

# Preprocessing
encoder = LabelEncoder()
df['income_level_label'] = encoder.fit_transform(df.income_level)

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
    print(df.groupby([c])['income_level_label'].sum())

# All possible pairplots
#sns.pairplot(df, hue='income_level_label')
#plt.show()

# sex
#sns.catplot(x='sex', y='income_level_label',kind='bar', data=df)

# race
#sns.catplot(x='race', y='income_level_label',kind='bar', data=df)
#sns.catplot(x='race', y='income_level_label', hue='sex', kind='bar', data=df)

# relationship
#sns.catplot(x='relationship', y='income_level_label',kind='bar', data=df)
#sns.catplot(x='relationship', y='income_level_label', hue='race', kind='bar', data=df)

# occupation
#sns.catplot(x='occupation', y='income_level_label',kind='bar', data=df)

# education
#sns.catplot(x='education', y='income_level_label',kind='bar', data=df)

# workclass
#sns.catplot(x='workclass', y='income_level_label',kind='bar', data=df)

# age
#sns.catplot(x='income_level_label', y='age', kind='box', data=df)

# Correlation between hours_per_week and income_level
sns.catplot(x='income_level', y='hours_per_week', kind='violin',data=df)

plt.show()

# Conclusions
# feature capital gain and captain loss apper not to make sense on the model due to their number of zeros.
# Do something with country, but excluding for now. Added to Catboost and
# Looks like good features race, sex, relationship, occupation(maybe), education, workclass, age. They are!
# Trying hours_per_week, sounds like a decent part of the >50k ate on the 50 hours per weak. Worked!
# Becareful with unbalanced target class
