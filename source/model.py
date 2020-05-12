# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import plot_confusion_matrix
# paths
dataset_path = "input/census_income_dataset.csv"

# reading
df = pd.read_csv(dataset_path)

# variables
RANDOM_STATE = 123
df = pd.read_csv(dataset_path)

# PREPROCESSING

# FEATURE ENGENIRING
# Do something with the countries

# FEATURE SELECTION
features = ['race', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age']
X = df[features]
y = df.income_level
print(y.value_counts())

X_dummies = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_dummies, y=='>50K', stratify=y, test_size=0.3, random_state=RANDOM_STATE)
print(X_train)
print(y_train)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# MODEL

# HistGradientBoostingClassifier
param_distributions_hgb = {
    'learning_rate' : np.logspace(-3, -1, 25),
    'max_iter' : np.arange(100, 300, 50),
    'min_samples_leaf' : np.arange(10, 50, 10),
    'random_state' : [RANDOM_STATE]
}
hgb = HistGradientBoostingClassifier()
hgb_CV = RandomizedSearchCV(hgb, param_distributions=param_distributions_hgb, cv=10, n_jobs=-1, random_state=RANDOM_STATE)
hgb_CV.fit(X_train, y_train)
print('-' * 100)
print('HGB train score: {:.3f}'.format(hgb_CV.score(X_train, y_train)))
print('HGB test score: {:.3f}'.format(hgb_CV.score(X_test, y_test)))
print('HGB best params: {0}'.format(hgb_CV.best_params_))

# Classfication metrics
plot_confusion_matrix(hgb_CV.best_estimator_, X_train, y_train)
plt.show()
plot_confusion_matrix(hgb_CV.best_estimator_, X_test, y_test)
plt.show()