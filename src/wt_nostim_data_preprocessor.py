import mat73
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 1000)

wt_nostim_dict = mat73.loadmat('../data/WT_NoStim.mat')
wt_nostim = wt_nostim_dict.get('WT_NoStim')
print(f"WT_NoStim dict keys: {wt_nostim.keys()}")


def process_wt_nostim():
    index = 0
    neuron_ids = []
    identified_neuron_ids = []
    states = []
    neuron_trace_data = []

    for x, y in wt_nostim.items():
        if x == 'NeuronNames':
            max = 0
            for idx, v in enumerate(y):
                if len(v) > max:
                    max = len(v)
                    index = idx
                    neuron_ids = v
            identified_neuron_ids = [k for k in neuron_ids if not k.isnumeric()]
            print(f"Using C. elegans GCaMP data with the highest number of neurons monitored, "
                  f"detected neurons len: {len(neuron_ids)}, identified neurons len: {len(identified_neuron_ids)}, IDs: {identified_neuron_ids}")

        if x == 'States':
            states_dict = y[index]
            print(f"state types: {states_dict.keys()}")
            state_dt = states_dict.get("dt")
            state_fwd = states_dict.get("fwd")
            state_nostate = states_dict.get("nostate")
            state_rev1 = states_dict.get("rev1")
            state_rev2 = states_dict.get("rev2")
            state_revsus = states_dict.get("revsus")
            state_slow = states_dict.get("slow")
            state_vt = states_dict.get("vt")

            for idx, v in enumerate(state_dt):
                if v == 1:
                    states.append("dt")
                elif state_fwd[idx] == 1:
                    states.append("fwd")
                elif state_nostate[idx] == 1:
                    states.append("nostate")
                elif state_rev1[idx] == 1:
                    states.append("rev1")
                elif state_rev2[idx] == 1:
                    states.append("rev2")
                elif state_revsus[idx] == 1:
                    states.append("revsus")
                elif state_slow[idx] == 1:
                    states.append("slow")
                elif state_vt[idx] == 1:
                    states.append("vt")
                else:
                    states.append("none")

        if x == 'deltaFOverF_bc':
            neuron_trace_data = y[index]

    return states, neuron_trace_data, neuron_ids, identified_neuron_ids


wt_nostim_behaviours, wt_nostim_neuron_trace_data, wt_nostim_neuron_ids, wt_nostim_identified_neuron_ids = process_wt_nostim()

wt_nostim_behaviour_df = pd.DataFrame({'behaviour': wt_nostim_behaviours})
print(f"\n wt_nostim behaviours data:")
print(wt_nostim_behaviour_df.info())
print(wt_nostim_behaviour_df)

wt_nostim_neuron_trace_df = pd.DataFrame(wt_nostim_neuron_trace_data, columns=wt_nostim_neuron_ids)
wt_nostim_neuron_trace_df = wt_nostim_neuron_trace_df[wt_nostim_identified_neuron_ids]
print(f"wt_nostim neuron trace data:")
print(wt_nostim_neuron_trace_df.info())
print(wt_nostim_neuron_trace_df)


def pre_process_data(df):
    # Create the pipeline
    num_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())])

    # Fit the pipeline to the training data
    data_prepared = num_pipeline.fit_transform(df)

    return data_prepared


wt_nostim_neuron_trace_train, wt_nostim_neuron_trace_test, wt_nostim_behaviour_train, wt_nostim_behaviour_test = train_test_split(
    wt_nostim_neuron_trace_df,
    wt_nostim_behaviour_df,
    test_size=0.3,
    random_state=42
)

wt_nostim_neuron_trace_train_prepared = pre_process_data(wt_nostim_neuron_trace_train)
print(f"wt_nostim neuron trace train prepared data:")
print(wt_nostim_neuron_trace_train_prepared.shape)
print(wt_nostim_neuron_trace_train_prepared)

wt_nostim_neuron_trace_test_prepared = pre_process_data(wt_nostim_neuron_trace_test)
print(f"wt_nostim neuron trace test prepared data:")
print(wt_nostim_neuron_trace_test_prepared.shape)
print(wt_nostim_neuron_trace_test_prepared)


def confusion_matrix_figure(confusion_matrix, class_names, title):
    fig, ax = plt.subplots(figsize=(18, 9))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=class_names,
                yticklabels=class_names)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(title)
    plt.show()


def find_relevant_features_from_behaviours(model, X, X_test, y_test):
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

    feature_importance = pd.DataFrame({'Neuron': X.columns,
                                       'Importance': result.importances_mean,
                                       'Standard Deviation': result.importances_std})
    feature_importance = feature_importance.sort_values('Importance', ascending=False).head(30)
    feature_importance_copy = feature_importance.sort_values('Importance', ascending=True)

    ax = feature_importance_copy.plot(x='Neuron', y='Importance', kind='barh', figsize=(1280 / 96, 720 / 96))
    ax.set_title('Neuronal importance of all behaviors')

    return feature_importance


def find_relevant_features_from_one_behaviour(model, X, behaviour):
    classes = model.classes_

    index = list(classes).index(behaviour)

    coefficients = model.coef_[index]

    feature_importance = pd.DataFrame({'Neuron': X.columns, 'Importance': np.abs(coefficients)})
    feature_importance = feature_importance.sort_values('Importance', ascending=False).head(30)

    feature_importance_copy = feature_importance.sort_values('Importance', ascending=True)

    ax = feature_importance_copy.plot(x='Neuron', y='Importance', kind='barh', figsize=(1280 / 96, 720 / 96))
    ax.set_title(f'Neuronal importance of {behaviour} behavior')

    return feature_importance


def find_relevant_features_from_random_forest_classifier(model, df):
    # Get the feature importances from the best Random_Forest classifier
    feature_importances = model.feature_importances_

    # Sort the features in descending order of importance
    sorted_indices = np.argsort(-feature_importances)

    # Get the top 5 feature names
    top_5_features = df.columns[sorted_indices[:5]]

    # Print the top 5 features and their corresponding coefficients
    print("Top 5 most important features from the Random Forest classifier (in descending order of importance):")
    for idx, feature in enumerate(top_5_features):
        print(f"{idx + 1}. {feature} ({feature_importances[sorted_indices[idx]]:.4f})")


def figure_wt_nostim(df, feature_importance_df):
    neurons = feature_importance_df['Neuron'].head(15).tolist()
    neuron_dict = {}
    for k in neurons:
        neuron_dict[k] = df[k].tolist()
    filtered_df = pd.DataFrame(neuron_dict)
    col = len(filtered_df.columns)

    fig, axes = plt.subplots(nrows=col, ncols=1, sharex=True, figsize=(15, 30))
    fig.suptitle('Neuronal traces of permutation importance')

    for c, ax in zip(filtered_df, axes):
        filtered_df[c].plot(ax=ax, label=f'{c}')
        ax.legend(loc='upper right')


def cross_val(clf, data_features_test_prepared, data_test_targets):
    k_folds = KFold(n_splits=10)

    scores = cross_val_score(clf, data_features_test_prepared, data_test_targets, cv=k_folds)
