import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo
from imblearn.over_sampling import SMOTE

def handle_missing_values(X):
    # Using mean imputation for missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_imputed = imputer.fit_transform(X)
    return X_imputed

def load_and_preprocess_data(binary=True):
    # Fetch dataset
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    # Handle missing values
    X = handle_missing_values(X)

    # Reshape y to a 1D array
    if binary == True:
        y = np.where(y > 0, 1, 0)
    else:
        y = y.values.ravel()

    return X, y

def split_and_scale_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train):
    smote = SMOTE()
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote

def save_models(best_models):
    for model in best_models:
        model_name = model[2]
        model_estimator = model[0]
        filename = f"models/model_{model_name}.joblib"
        joblib.dump(model_estimator, filename)


def load_models(filename):
    model_estimator = joblib.load(filename)
    return model_estimator

