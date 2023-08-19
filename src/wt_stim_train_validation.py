import feature_importance
import model
import wt_stim_data_preprocessing

wt_stim_neuron_trace_df = wt_stim_data_preprocessing.wt_stim_neuron_trace_df
wt_stim_neuron_trace_train_prepared, wt_stim_neuron_trace_test_prepared, wt_stim_behaviour_train, wt_stim_behaviour_test = wt_stim_data_preprocessing.get_wt_stim_data_prepared()

# train and validate logistic regression classifier
wt_stim_lr_clf = model.train_and_evaluate_logistic_regression_classifiers(
    wt_stim_neuron_trace_train_prepared,
    wt_stim_behaviour_train,
    wt_stim_neuron_trace_test_prepared,
    wt_stim_behaviour_test
)

# train and validate random forest classifier
wt_stim_rf_clf = model.train_and_evaluate_random_forest_classifiers(
    wt_stim_neuron_trace_train_prepared,
    wt_stim_behaviour_train,
    wt_stim_neuron_trace_test_prepared,
    wt_stim_behaviour_test
)

# neuron feature importance of all behaviours by random forest classifier
wt_stim_behaviours_feature_importance_df = feature_importance.find_relevant_features_from_rf_clf_all_behaviours(
    wt_stim_rf_clf,
    wt_stim_neuron_trace_df
)

# neuron feature importance of all behaviours
feature_importance.figure_wt_stim_neuron_trace(
    wt_stim_neuron_trace_df,
    wt_stim_behaviours_feature_importance_df
)

# Neuron feature importance of rev behaviour by logistic regression classifier
wt_stim_one_behaviour_feature_importance_df = feature_importance.find_relevant_features_from_lr_clf_one_behaviour(
    wt_stim_lr_clf,
    wt_stim_neuron_trace_df,
    'rev'
)

# neuron trace of revsus behaviour
feature_importance.figure_wt_stim_neuron_trace(
    wt_stim_neuron_trace_df,
    wt_stim_one_behaviour_feature_importance_df
)

# Neuron feature importance of turn behaviour by logistic regression classifier
wt_stim_one_behaviour_feature_importance_df = feature_importance.find_relevant_features_from_lr_clf_one_behaviour(
    wt_stim_lr_clf,
    wt_stim_neuron_trace_df,
    'turn'
)

# neuron trace of turn behaviour
feature_importance.figure_wt_stim_neuron_trace(
    wt_stim_neuron_trace_df,
    wt_stim_one_behaviour_feature_importance_df
)
