import mat73
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

wt_stim_dict = mat73.loadmat('../data/WT_Stim.mat')
wt_stim = wt_stim_dict.get('WT_Stim')
print(f"WT_Stim dict keys: {wt_stim.keys()}")


# process WT_Stim.mat data file
def process_wt_stim():
    index = 0
    neuron_ids = []
    identified_neuron_ids = []
    states = []
    neuron_trace_data = []

    behaviour_type_dict = {
        1: "fwd",
        2: "rev",
        3: "revsus",
        4: "turn"
    }

    for x, y in wt_stim.items():
        if x == 'IDs':
            max = 0
            for idx, v in enumerate(y):
                if len(v) > max:
                    max = len(v)
                    index = idx
                    neuron_ids = v
            for idx, v in enumerate(neuron_ids):
                if v is None:
                    neuron_ids[idx] = str(idx + 1)
                else:
                    neuron_ids[idx] = '/'.join(v)
                    identified_neuron_ids.append(neuron_ids[idx])
            print(f"Using C. elegans GCaMP data with the highest number of neurons monitored, "
                  f"detected neurons len: {len(neuron_ids)}, identified neurons len: {len(identified_neuron_ids)}, IDs: {identified_neuron_ids}")
        if x == 'States':
            states = y[index].astype(np.int64)
            states = [behaviour_type_dict.get(k) for k in states]
        if x == 'stimulus':
            stimulus = y[index]
            print(f"stimulus: {stimulus}")
        if x == 'traces':
            neuron_trace_data = y[index]

    return states, neuron_trace_data, neuron_ids, identified_neuron_ids


# get wt_stim data
def get_wt_stim_data():
    wt_stim_behaviours, wt_stim_neuron_trace_data, wt_stim_neuron_ids, wt_stim_identified_neuron_ids = process_wt_stim()

    wt_stim_behaviour_df = pd.DataFrame({'behaviour': wt_stim_behaviours})
    print(f"\n wt_stim behaviour data:")
    print(wt_stim_behaviour_df.info())
    print(wt_stim_behaviour_df)

    wt_stim_neuron_trace_df = pd.DataFrame(wt_stim_neuron_trace_data, columns=wt_stim_neuron_ids)
    wt_stim_neuron_trace_df = wt_stim_neuron_trace_df[wt_stim_identified_neuron_ids]
    print(f"wt_stim neuron trace data:")
    print(wt_stim_neuron_trace_df.info())
    print(wt_stim_neuron_trace_df)

    return wt_stim_neuron_trace_df, wt_stim_behaviour_df


wt_stim_neuron_trace_df, wt_stim_behaviour_df = get_wt_stim_data()


# data preprocessing
def pre_process_data(df):
    # Create the pipeline
    num_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())])

    # Fit the pipeline to the training data
    data_prepared = num_pipeline.fit_transform(df)

    return data_prepared


# get the preprocessed data and distinguish between training set and test set
def get_wt_stim_data_prepared():
    wt_stim_neuron_trace_train, wt_stim_neuron_trace_test, wt_stim_behaviour_train, wt_stim_behaviour_test = train_test_split(
        wt_stim_neuron_trace_df,
        wt_stim_behaviour_df,
        test_size=0.3,
        random_state=42
    )

    wt_stim_neuron_trace_train_prepared = pre_process_data(wt_stim_neuron_trace_train)
    print(f"wt_stim neuron trace train prepared:")
    print(wt_stim_neuron_trace_train_prepared.shape)
    print(wt_stim_neuron_trace_train_prepared)

    wt_stim_neuron_trace_test_prepared = pre_process_data(wt_stim_neuron_trace_test)
    print(f"wt_stim neuron trace test prepared:")
    print(wt_stim_neuron_trace_test_prepared.shape)
    print(wt_stim_neuron_trace_test_prepared)

    return wt_stim_neuron_trace_train_prepared, wt_stim_neuron_trace_test_prepared, wt_stim_behaviour_train, wt_stim_behaviour_test
