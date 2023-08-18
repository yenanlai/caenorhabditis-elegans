import mat73
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 1000)

wt_stim_dict = mat73.loadmat('../data/WT_Stim.mat')
wt_stim = wt_stim_dict.get('WT_Stim')
print(f"WT_Stim dict keys: {wt_stim.keys()}")


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
            print(f"states len: {len(states)}, value: {states}")
        if x == 'stimulus':
            stimulus = y[index]
            print(f"stimulus: {stimulus}")
        if x == 'traces':
            neuron_trace_data = y[index]
            print(f"neuron_trace_data len:{len(neuron_trace_data)}, value: {neuron_trace_data}")

    return states, neuron_trace_data, neuron_ids, identified_neuron_ids


wt_stim_behaviours, wt_stim_neuron_trace_data, wt_stim_neuron_ids, wt_stim_identified_neuron_ids = process_wt_stim()

wt_stim_behaviours_df = pd.DataFrame({'behaviour': wt_stim_behaviours})
print(f"wt_stim behaviours data:")
print(wt_stim_behaviours_df.info())
print(wt_stim_behaviours_df)

wt_stim_neuron_trace_df = pd.DataFrame(wt_stim_neuron_trace_data, columns=wt_stim_neuron_ids)
wt_stim_neuron_trace_df = wt_stim_neuron_trace_df[wt_stim_identified_neuron_ids]
print(f"wt_stim neuron trace data:")
print(wt_stim_neuron_trace_df.info())
print(wt_stim_neuron_trace_df)


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
        ax.set_xticks(np.arange(0, 1000, step=60))
        ax.legend(loc='upper right')
