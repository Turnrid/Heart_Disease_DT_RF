import data_preprocessing as dp
import train_models as tm
from evaluate_models import evaluate_model


def top_three_models(accuracies):
    best_models = [[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    for accuracy in accuracies:
        if accuracy[2] == "decision_tree":
            if accuracy[3] > best_models[0][3]:
                best_models[0] = accuracy
        elif accuracy[2] == "random_forest":
            if accuracy[3] > best_models[1][3]:
                best_models[1] = accuracy
        elif accuracy[2] == "grad_boost":
            if accuracy[3] > best_models[2][3]:
                best_models[2] = accuracy

    return best_models


def main(filename=None, totalRuns=1):
    # Load and preprocess data
    X, y = dp.load_and_preprocess_data()

    if filename == None:

        accuracies = []
        X_train, X_test, y_train, y_test = dp.split_and_scale_data(X, y)

        # Apply SMOTE to handle class imbalance
        X_train_smote, y_train_smote = dp.apply_smote(X_train, y_train)

        for i in range(totalRuns):
            print(f"Run {i+1} of {totalRuns}")

            # Tune Decision Tree and train models
            dt_model = tm.train_decision_tree(X_train_smote, y_train_smote)
            rf_model = tm.train_random_forest(X_train_smote, y_train_smote)
            xgb_model = tm.train_xgboost(X_train_smote, y_train_smote)

            # Evaluate models
            evaluate_model(dt_model, X_test, y_test, accuracies, "decision_tree")
            evaluate_model(rf_model, X_test, y_test, accuracies, "random_forest")
            evaluate_model(xgb_model, X_test, y_test, accuracies, "grad_boost")

        best_models = top_three_models(accuracies)

        dp.save_models(best_models)

        for best_model in best_models:
            evaluate_model(best_model, X_test, y_test, accuracies, best_model[2], debug=True)
            print(f"Best Parameters for {best_model[2]}: {best_model[1]}\n")
    else:
        model = dp.load_models(filename)
        model = [model, {}, filename.split("_")[1].split(".")[0]]
        evaluate_model(model, X, y, [], filename, debug=True)


        



if __name__ == "__main__":
    models = ["best_models/model_decision_tree.joblib", "best_models/model_random_forest.joblib", "best_models/model_grad_boost.joblib"]

    main(totalRuns=3)
    # for model in models:
    #     main(model)

