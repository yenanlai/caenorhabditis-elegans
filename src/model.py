import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold


# Define logistic regression classifiers
def train_and_evaluate_logistic_regression_classifiers(data_features_train_prepared, data_train_targets,
                                                       data_features_test_prepared, data_test_targets):
    # Set up hyperparameter grid for tuning
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['newton-cg']
    }

    # Initialize classifier
    log_reg = LogisticRegression(max_iter=1000, random_state=42)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(log_reg, param_grid, cv=StratifiedKFold(n_splits=10), scoring='accuracy', verbose=1,
                               n_jobs=-1)

    # Fit model
    grid_search.fit(data_features_train_prepared, data_train_targets.values.ravel())

    # Find the best model
    print(f"LogisticRegression classifier optimal parameters: {grid_search.best_params_}")
    best_clf = grid_search.best_estimator_

    # Predictions on the test set
    clf_preds = best_clf.predict(data_features_test_prepared)

    # Results
    accuracy = accuracy_score(data_test_targets, clf_preds)
    print(f"Logistic Regression Classifier Accuracy: {accuracy * 100: .1f}%\n")

    print(f"Logistic Regression Classifier Confusion Matrix:")
    class_names = data_test_targets['behaviour'].unique()
    confusion_matrix_figure(confusion_matrix(data_test_targets, clf_preds), class_names, "Confusion Matrix - Test Set")

    print("\n Logistic Regression Classifier Classification Report:")
    print(classification_report(data_test_targets, clf_preds))

    return best_clf


# define random forest classifier
def train_and_evaluate_random_forest_classifiers(data_features_train_prepared, data_train_targets,
                                                 data_features_test_prepared, data_test_targets):
    # Set up a hyperparameter grid search using GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize classifier
    rf = RandomForestClassifier(random_state=42)
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

    # Fit the model to the training data
    grid_search.fit(data_features_train_prepared, data_train_targets.values.ravel())

    # Use the best estimator from the grid search to predict the classes of the test set
    best_clf = grid_search.best_estimator_
    print(f"Random Forest classifier optimal parameters: {grid_search.best_params_}")
    clf_preds = best_clf.predict(data_features_test_prepared)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(data_test_targets, clf_preds)
    print(f"Random Forest Classifier Accuracy: {accuracy * 100: .1f}%\n")

    # Calculate the confusion matrix
    print(f"Random Forest Classifier Confusion Matrix:")
    class_names = data_test_targets['behaviour'].unique()
    confusion_matrix_figure(confusion_matrix(data_test_targets, clf_preds), class_names, "Confusion Matrix - Test Set")

    # Print the classification report
    print("\n Random Forest Classifier Classification Report:")
    print(classification_report(data_test_targets, clf_preds))

    return best_clf


# plot confusion matrix
def confusion_matrix_figure(confusion_matrix, class_names, title):
    fig, ax = plt.subplots(figsize=(18, 9))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=class_names,
                yticklabels=class_names)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(title)
    plt.show()
