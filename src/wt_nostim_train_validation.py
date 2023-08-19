import feature_importance
import model
import wt_nostim_data_preprocessing

wt_nostim_neuron_trace_df = wt_nostim_data_preprocessing.wt_nostim_neuron_trace_df
wt_nostim_neuron_trace_train_prepared, wt_nostim_neuron_trace_test_prepared, wt_nostim_behaviour_train, wt_nostim_behaviour_test = wt_nostim_data_preprocessing.get_wt_nostim_data_prepared()

# train and validate logistic regression classifier
wt_nostim_lr_clf = model.train_and_evaluate_logistic_regression_classifiers(
    wt_nostim_neuron_trace_train_prepared,
    wt_nostim_behaviour_train,
    wt_nostim_neuron_trace_test_prepared,
    wt_nostim_behaviour_test
)

# train and validate random forest classifier
wt_nostim_rf_clf = model.train_and_evaluate_random_forest_classifiers(
    wt_nostim_neuron_trace_train_prepared,
    wt_nostim_behaviour_train,
    wt_nostim_neuron_trace_test_prepared,
    wt_nostim_behaviour_test
)

# neuron feature importance of all behaviours by random forest classifier
wt_nostim_behaviours_feature_importance_df = feature_importance.find_relevant_features_from_rf_clf_all_behaviours(
    wt_nostim_rf_clf,
    wt_nostim_neuron_trace_df
)

# neuron feature importance of all behaviours
feature_importance.figure_wt_nostim_neuron_trace(
    wt_nostim_neuron_trace_df,
    wt_nostim_behaviours_feature_importance_df
)

# Neuron feature importance of revsus behaviour by logistic regression classifier
wt_nostim_revsus_behaviour_feature_importance_df = feature_importance.find_relevant_features_from_lr_clf_one_behaviour(
    wt_nostim_lr_clf,
    wt_nostim_neuron_trace_df,
    'revsus'
)

# neuron trace of revsus behaviour
feature_importance.figure_wt_nostim_neuron_trace(
    wt_nostim_neuron_trace_df,
    wt_nostim_revsus_behaviour_feature_importance_df
)

# Neuron feature importance of nostate behaviour by logistic regression classifier
wt_nostim_nostate_behaviour_feature_importance_df = feature_importance.find_relevant_features_from_lr_clf_one_behaviour(
    wt_nostim_lr_clf,
    wt_nostim_neuron_trace_df,
    'nostate'
)

# neuron trace of nostate behaviour
feature_importance.figure_wt_nostim_neuron_trace(
    wt_nostim_neuron_trace_df,
    wt_nostim_nostate_behaviour_feature_importance_df
)
