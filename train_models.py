from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
from scipy.stats import randint

# Using decision tree
def train_decision_tree(X_train, y_train):
    # Using search for the best hyperparameters for the decision tree
    param_dist = {
                "criterion": ["gini", "entropy"],
                "max_depth": randint(1, 100),
                "min_samples_leaf": randint(1, 100),
                "min_samples_split": randint(2, 100),
                "max_features": randint(1, 100),
                "ccp_alpha": [0, 0.01, 0.1, 1, 10, 100]
                }

    dt_classifier = DecisionTreeClassifier()

    rand_search = RandomizedSearchCV(dt_classifier, param_distributions=param_dist, n_iter=5, cv=5, scoring='accuracy')

    dt_classifier = rand_search.fit(X_train, y_train)
    return dt_classifier.best_estimator_

# Using random forest
def train_random_forest(X_train, y_train):
    # Using search for the best hyperparameters for the random forest
    param_dist = {"n_estimators": randint(50, 1000),
                "max_depth": randint(1, 100),
                "min_samples_leaf": randint(1, 100),
                "min_samples_split": randint(2, 100),
                "max_features": randint(1, 100),
                "ccp_alpha": [0, 0.01, 0.1, 1, 10, 100]
                }

    rf_classifier = RandomForestClassifier()
    
    rand_search = RandomizedSearchCV(rf_classifier, param_distributions=param_dist, n_iter=5, cv=5, scoring='accuracy')
    
    rand_search.fit(X_train, y_train)
    return rand_search.best_estimator_

# Using gradient boosting
def train_xgboost(X_train, y_train):
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    return xgb
