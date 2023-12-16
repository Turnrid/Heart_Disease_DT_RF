from data_preprocessing import load_and_preprocess_data, split_and_scale_data, apply_smote
from train_models import train_decision_tree, train_random_forest, train_xgboost
from evaluate_models import evaluate_model



def top_three_models(accuracies):
    best_models = [[0],[0],[0]]
    for accuracy in accuracies:
        if accuracy[1] == "decision_tree":
            if accuracy[0] > best_models[0][0]:
                best_models[0] = accuracy
        elif accuracy[1] == "random_forest":
            if accuracy[0] > best_models[1][0]:
                best_models[1] = accuracy
        elif accuracy[1] == "xgboost":
            if accuracy[0] > best_models[2][0]:
                best_models[2] = accuracy

    return best_models


def main():
    # Load and preprocess data
    # if you want to use the multivariable output, set binary=False for load_and_preprocess_data()
    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = split_and_scale_data(X, y)

    # Apply SMOTE to handle class imbalance
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    accuracies = []

    for _ in range(5):
        # Tune Decision Tree and train models
        dt_model = train_decision_tree(X_train_smote, y_train_smote)
        rf_model = train_random_forest(X_train_smote, y_train_smote)
        xgb_model = train_xgboost(X_train_smote, y_train_smote) 

        # Evaluate models
        evaluate_model(dt_model, X_test, y_test, accuracies, "decision_tree")
        evaluate_model(rf_model, X_test, y_test, accuracies, "random_forest")
        evaluate_model(xgb_model, X_test, y_test, accuracies, "xgboost")
    
    best_models = top_three_models(accuracies)

    for best_model in best_models:
        evaluate_model(best_model[2], X_test, y_test, accuracies, best_model[1], debug=True)




if __name__ == "__main__":
    main()
