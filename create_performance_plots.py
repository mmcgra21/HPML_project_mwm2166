import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd

# Iterate through files in outputs folder
data = dict()
data['optimizer'] = []
data['model_size'] = []
data['batch_size'] = []
data['wps'] = []
for filename in os.listdir("outputs"):
    names = filename.split(".")[0].split("_")
    specifier = names[1]
    model_size = int(names[2])
    batch_size = int(names[3])
    with open(f"outputs/{filename}", "r") as f:
        lines = f.readlines()
    wps = []
    for line in lines:
        if "wps=" in line:
            tmp = line.split("wps=")[1]
            if ', ' in tmp:
                wps.append(float(tmp.split(', ')[0]))
    wps = np.median(np.array(wps))
    if specifier == 'f':
        optimizer = 'Fairseq'
    elif specifier == 'fa':
        optimizer = 'Fairseq+Apex'
    elif specifier == 'l':
        optimizer = 'LightSeq2'
    # elif specifier == 'la':
    #     optimizer = 'LightSeq w/ Apex'
    else:
        optimizer = 'Unknown'
    data['optimizer'].append(optimizer)
    data['model_size'].append(model_size)
    data['batch_size'].append(batch_size)
    data['wps'].append(wps)
df = pd.DataFrame(data)

colors = sns.color_palette("colorblind")

# Normalize wps w.r.t. Fairseq wps at that batch size and model size
for model_size in df['model_size'].unique():
    for batch_size in df[df['model_size'] == model_size]['batch_size'].unique():
        df.loc[(df['model_size'] == model_size) & (df['batch_size'] == batch_size),'wps'] = df[(df['model_size'] == model_size) & (df['batch_size'] == batch_size)]['wps'].div(df[(df['optimizer'] == 'Fairseq') & (df['model_size'] == model_size) & (df['batch_size'] == batch_size)]['wps'].values[0])
        df['batch_size_enum'] = df['batch_size'].astype('category').cat.codes

# For each of the different model sizes, create plot of the speed up with respect to the batch size
for model_size in [6,12,18]:
    tmp = df[df['model_size'] == model_size]
    # normalize wps with respect to wps of Fairseq
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.lineplot(data=tmp, x='batch_size_enum', y='wps', hue='optimizer', ax=ax, hue_order=['Fairseq', 'Fairseq+Apex', 'LightSeq2'], palette='colorblind', style='optimizer', dashes={'Fairseq': (1, 2), 'Fairseq+Apex': (3, 2),  'LightSeq2': (1, 0)})
    # ax.set_title(f'{model_size}e{model_size}d on V100')
    ax.set_xlabel('Batch token size')
    ax.set_ylabel('Speedup')
    ax.get_legend().set_title("")
    ax.set_xticks(range(len(tmp['batch_size'].unique())))
    ax.set_xticklabels(np.sort(tmp['batch_size'].unique()))
    fig.savefig(f'plots/speedup_plot_{model_size}e{model_size}d.png')

# Same plots but side-by-side
pw = 4
ph = 3
n_cols = 3
n_rows = 1
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*pw, n_rows*ph))

# For each of the different model sizes, create plot of the speed up with respect to the batch size
for idx,model_size in enumerate([6,12,18]):
    tmp = df[df['model_size'] == model_size]
    # normalize wps with respect to wps of Fairseq
    sns.lineplot(data=tmp, x='batch_size_enum', y='wps', hue='optimizer', ax=axes[idx], hue_order=['Fairseq', 'Fairseq+Apex', 'LightSeq2'], palette='colorblind', style='optimizer', dashes={'Fairseq': (1, 2), 'Fairseq+Apex': (3, 2),  'LightSeq2': (1, 0)})
    # ax.set_title(f'{model_size}e{model_size}d on V100')
    axes[idx].set_xlabel('Batch token size')
    axes[idx].set_ylabel('Speedup')
    axes[idx].get_legend().set_title("")
    axes[idx].set_xticks(range(len(tmp['batch_size'].unique())))
    axes[idx].set_xticklabels(np.sort(tmp['batch_size'].unique()))
    axes[idx].legend(loc='upper right')
fig.savefig(f'plots/speedup_plots.png', bbox_inches="tight")