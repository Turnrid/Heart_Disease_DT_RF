from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def evaluate_model(model, X_test, y_test, accuracies, model_name="Model", debug=False):
    predictions = model[0].predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    clf_report = classification_report(y_test, predictions, zero_division=1)
    conf_matrix = confusion_matrix(y_test, predictions)


    accuracies.append([model[0], model[1], model_name, accuracy])

    if debug:
        print(f"\n==== {model_name} Evaluation ====")
        print("Accuracy:", accuracy)
        print("\nClassification Report:\n", clf_report)
        print("\nConfusion Matrix:\n", conf_matrix)
        print("\n")
