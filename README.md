# Census Income
### Input
Instances: 48842

Attributes: 15

Tasks: Classification

Year Published: 1996

Link: [census income](https://www.mldata.io/dataset-details/census_income/)


### Objective
Predict if an individual makes greater or less than $50000 per year

### Output
source/EDA.py has the exploratory data analysis done in the dataset. A look inside each feature and how they are correlated with the target variable. Plots that show the relation between features and the target variable. And some conclusions about the feature selection after running the model.

source/model.py has the feature selection and the modelling for both CatBoostClassifier and HistGradientBoostingClassifier.

source/models.csv has results for both models, selected features, chosen parameters and their detalied scores.
