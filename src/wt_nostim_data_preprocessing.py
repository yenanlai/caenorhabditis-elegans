import mat73
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

wt_nostim_dict = mat73.loadmat('../data/WT_NoStim.mat')
wt_nostim = wt_nostim_dict.get('WT_NoStim')
print(f"WT_NoStim dict keys: {wt_nostim.keys()}")


# process WT_NoStim.mat data file
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


# get wt_nostim data
def get_wt_nostim_data():
    wt_nostim_behaviours, wt_nostim_neuron_trace_data, wt_nostim_neuron_ids, wt_nostim_identified_neuron_ids = process_wt_nostim()

    wt_nostim_behaviour_df = pd.DataFrame({'behaviour': wt_nostim_behaviours})
    print(f"\n wt_nostim behaviours data:")
    print(wt_nostim_behaviour_df.info())
    print(wt_nostim_behaviour_df)

    wt_nostim_neuron_trace_df = pd.DataFrame(wt_nostim_neuron_trace_data, columns=wt_nostim_neuron_ids)
    wt_nostim_neuron_trace_df = wt_nostim_neuron_trace_df[wt_nostim_identified_neuron_ids]
    print(f"\n wt_nostim neuron trace data:")
    print(wt_nostim_neuron_trace_df.info())
    print(wt_nostim_neuron_trace_df)

    return wt_nostim_neuron_trace_df, wt_nostim_behaviour_df


wt_nostim_neuron_trace_df, wt_nostim_behaviour_df = get_wt_nostim_data()


# data preprocessing
def pre_process_data(df):
    # Create the pipeline
    num_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())])

    # Fit the pipeline to the training data
    data_prepared = num_pipeline.fit_transform(df)

    return data_prepared


# get the preprocessed data and distinguish between training set and test set
def get_wt_nostim_data_prepared():
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
    print(f"\n wt_nostim neuron trace test prepared data:")
    print(wt_nostim_neuron_trace_test_prepared.shape)
    print(wt_nostim_neuron_trace_test_prepared)

    return wt_nostim_neuron_trace_train_prepared, wt_nostim_neuron_trace_test_prepared, wt_nostim_behaviour_train, wt_nostim_behaviour_test
