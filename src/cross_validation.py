from sklearn.model_selection import KFold, cross_val_score

import wt_nostim_data_preprocessing
import wt_nostim_train_validation
import wt_stim_data_preprocessing
import wt_stim_train_validation


# define cross-validation function
def cross_val(best_clf, data_features_prepared, data_targets):
    # K-Fold
    k_folds = KFold(n_splits=10)
    scores = cross_val_score(best_clf, data_features_prepared, data_targets.values.ravel(), cv=k_folds)
    return scores.mean()


wt_nostim_neuron_trace_prepared = wt_nostim_data_preprocessing.pre_process_data(
    wt_nostim_data_preprocessing.wt_nostim_neuron_trace_df)
wt_stim_neuron_trace_prepared = wt_stim_data_preprocessing.pre_process_data(
    wt_stim_data_preprocessing.wt_stim_neuron_trace_df)

wt_nostim_lr_clf_cv = cross_val(wt_nostim_train_validation.wt_nostim_lr_clf, wt_nostim_neuron_trace_prepared,
                                wt_nostim_data_preprocessing.wt_nostim_behaviour_df)
wt_nostim_rf_clf_cv = cross_val(wt_nostim_train_validation.wt_nostim_rf_clf, wt_nostim_neuron_trace_prepared,
                                wt_nostim_data_preprocessing.wt_nostim_behaviour_df)
wt_stim_lr_clf_cv = cross_val(wt_stim_train_validation.wt_stim_lr_clf, wt_stim_neuron_trace_prepared,
                              wt_stim_data_preprocessing.wt_stim_behaviour_df)
wt_stim_rf_clf_cv = cross_val(wt_stim_train_validation.wt_stim_rf_clf, wt_stim_neuron_trace_prepared,
                              wt_stim_data_preprocessing.wt_stim_behaviour_df)

print(f"wt_nostim logistic regression classifier cv avg score: {wt_nostim_lr_clf_cv}")
print(f"wt_nostim random forest classifier cv avg score: {wt_nostim_rf_clf_cv}")
print(f"wt_stim logistic regression classifier cv avg score: {wt_stim_lr_clf_cv}")
print(f"wt_stim random forest classifier cv avg score: {wt_stim_rf_clf_cv}")
