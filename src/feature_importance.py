import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# define random forest classifier feature importance
def find_relevant_features_from_rf_clf_all_behaviours(model, X):
    feature_importances = model.feature_importances_
    feature_importance = pd.DataFrame({'Neuron': X.columns,
                                       'Importance': feature_importances})
    feature_importance = feature_importance.sort_values('Importance', ascending=False).head(30)
    feature_importance_copy = feature_importance.sort_values('Importance', ascending=True)

    ax = feature_importance_copy.plot(x='Neuron', y='Importance', kind='barh', figsize=(1280 / 96, 720 / 96))
    ax.set_title('Neuronal importance of all behaviours')

    return feature_importance


# define logistic regression classifier feature importance
def find_relevant_features_from_lr_clf_one_behaviour(model, X, behaviour):
    classes = model.classes_

    index = list(classes).index(behaviour)

    coefficients = model.coef_[index]

    feature_importance = pd.DataFrame({'Neuron': X.columns, 'Importance': np.abs(coefficients)})
    feature_importance = feature_importance.sort_values('Importance', ascending=False).head(30)

    feature_importance_copy = feature_importance.sort_values('Importance', ascending=True)

    ax = feature_importance_copy.plot(x='Neuron', y='Importance', kind='barh', figsize=(1280 / 96, 720 / 96))
    ax.set_title(f'Neuronal importance of {behaviour} behaviour')

    return feature_importance


# plot wt_nostim neuron trace
def figure_wt_nostim_neuron_trace(df, feature_importance_df):
    neurons = feature_importance_df['Neuron'].head(15).tolist()
    neuron_dict = {}
    for k in neurons:
        neuron_dict[k] = df[k].tolist()
    filtered_df = pd.DataFrame(neuron_dict)

    col = len(filtered_df.columns)

    fig, axes = plt.subplots(nrows=col, ncols=1, sharex=True, figsize=(15, 15))
    fig.suptitle('Neuronal traces of permutation importance')

    for c, ax in zip(filtered_df, axes):
        filtered_df[c].plot(ax=ax, label=f'{c}')
        ax.legend(loc='upper right')


# plot wt_stim neuron trace
def figure_wt_stim_neuron_trace(df, feature_importance_df):
    neurons = feature_importance_df['Neuron'].head(15).tolist()
    neuron_dict = {}
    for k in neurons:
        neuron_dict[k] = df[k].tolist()[:1000]
    filtered_df = pd.DataFrame(neuron_dict)

    col = len(filtered_df.columns)

    fig, axes = plt.subplots(nrows=col, ncols=1, sharex=True, figsize=(15, 15))
    fig.suptitle('Neuronal traces of permutation importance')

    for c, ax in zip(filtered_df, axes):
        filtered_df[c].plot(ax=ax, label=f'{c}')
        ax.set_xticks(np.arange(0, 1000, step=30))
        ax.legend(loc='upper right')
