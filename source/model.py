# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import plot_confusion_matrix, classification_report

# variables
RANDOM_STATE = 123

# paths
dataset_path = "input/census_income_dataset.csv"

# reading
df = pd.read_csv(dataset_path)

# PREPROCESSING

# FEATURE ENGENIRING
# Do something with the countries

# FEATURE SELECTION
features = ['race', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age', 'hours_per_week']
X = df[features]
y = df.income_level

X_dummies = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_dummies, y=='>50K', stratify=y, test_size=0.3, random_state=RANDOM_STATE)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
# MODEL

# Catboost
cat_features = ['race', 'sex', 'relationship', 'occupation', 'education', 'workclass']

cat = CatBoostClassifier(cat_features=cat_features, random_seed=RANDOM_STATE)
#cat.load_model('cat')

cat.fit(X_train_cat, y_train_cat)
print('CatBoost train score: {:.3f}'.format(cat.score(X_train_cat, y_train_cat)))
print('CatBoost test score: {:.3f}'.format(cat.score(X_test_cat, y_test_cat)))
print(classification_report(y_test_cat, cat.predict(X_test_cat)))
#cat.save_model('cat',pool=X_train_cat)

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
print(classification_report(y_test, hgb_CV.best_estimator_.predict(X_test)))
#plot_confusion_matrix(hgb_CV.best_estimator_, X_train, y_train)
#plt.show()
#plot_confusion_matrix(hgb_CV.best_estimator_, X_test, y_test)
#plt.show()