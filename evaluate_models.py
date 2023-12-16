from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sn

def evaluate_model(model, X_test, y_test, accuracies, model_name="Model", debug=False):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    clf_report = classification_report(y_test, predictions, zero_division=1)
    conf_matrix = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(10, 7))
    sn.heatmap(conf_matrix, annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("Truth")

    save_path = f"confusion_matrix_{model_name}.png"
    plt.savefig(save_path)

    accuracies.append([accuracy, model_name, model])

    if debug:
        print(f"==== {model_name} Evaluation ====")
        print("Accuracy:", accuracy)
        print("\nClassification Report:\n", clf_report)
        print("\nConfusion Matrix:\n", conf_matrix)
        print("\n")
